"""
guardrails.py — Day 6

Two guardrails:
  1. Score-based OOC safety net: refuses before LLM call if top chunk score < OOC_THRESHOLD.
     Note: for this corpus, domain-specific OOC questions score 0.66–0.79 because they
     use banking vocabulary that matches the domain. The primary OOC mechanism is LLM Rule 3.
     This threshold (0.50) only catches completely unrelated questions.

  2. Hallucination verifier: given a (claim, chunk_text) pair, checks whether the claim
     is directly supported by the chunk. Used to catch answers that cite a chunk but state
     something the chunk does not say.

Usage:
    python scripts/guardrails.py            # runs both measurements
    python scripts/guardrails.py --ooc      # OOC refusal rate only (fast, no LLM)
    python scripts/guardrails.py --hall     # hallucination detection only (~2 min)
"""

import sys
import os
import json
import argparse
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from sentence_transformers import SentenceTransformer
import chromadb
import ollama
from retrieve import retrieve

# --- Constants ---
CHROMA_DIR    = "data/chroma_db"
COLLECTION    = "finsight"
MODEL_NAME    = "BAAI/bge-small-en-v1.5"
EVAL_FILE     = "eval/eval_set.jsonl"
SEEDS_FILE    = "eval/hallucination_seeds.jsonl"
CHUNKS_FILE   = "data/chunks.jsonl"
OLLAMA_MODEL  = "llama3.1:8b"

# Score-based safety net threshold.
# OOC questions in this corpus score 0.66–0.79 due to domain overlap,
# so this threshold only catches truly unrelated queries.
OOC_THRESHOLD = 0.50

REFUSAL_PHRASE = (
    "I'm sorry, I don't have enough information to answer that reliably. "
    "Please contact our support team."
)


# ---------------------------------------------------------------------------
# Guardrail 1: Score-based OOC check
# ---------------------------------------------------------------------------

def is_ooc(chunks):
    """
    Returns True if the top retrieved chunk scores below OOC_THRESHOLD.
    This is a last-resort safety net. Primary OOC detection relies on LLM Rule 3.
    """
    if not chunks:
        return True
    # hybrid pipeline uses rerank_score; naive pipeline uses score
    # rerank_score is cross-encoder scale (unbounded) so can't threshold it —
    # fall back to 1.0 (treat as in-corpus) and let LLM Rule 3 handle OOC
    top_score = chunks[0].get("score", 1.0)
    return top_score < OOC_THRESHOLD


# ---------------------------------------------------------------------------
# Guardrail 2: Hallucination verifier
# ---------------------------------------------------------------------------

def load_chunk_text(chunk_id):
    """Look up chunk text from chunks.jsonl by chunk_id."""
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["chunk_id"] == chunk_id:
                return chunk["text"]
    return None


def verify_claim(claim, chunk_text):
    """
    Ask the LLM whether a claim is directly supported by the chunk text.
    Returns "SUPPORTED" or "UNSUPPORTED".

    Uses a strict prompt so the model does not infer or extrapolate —
    the same principle as Rule 7 in the system prompt.
    """
    prompt = f"""You are a fact verification assistant. Your only job is to check whether a specific claim is directly supported by the provided text.

Text:
{chunk_text}

Claim: {claim}

Rules:
- Answer SUPPORTED only if the claim is explicitly stated or directly follows from what the text says.
- Answer UNSUPPORTED if the claim contradicts the text, states a different number or condition, or if the text does not mention it at all.
- Answer with exactly one word: SUPPORTED or UNSUPPORTED. Nothing else.

Answer:"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"].strip().upper()

    if "UNSUPPORTED" in answer:
        return "UNSUPPORTED"
    elif "SUPPORTED" in answer:
        return "SUPPORTED"
    # Fallback: look for negative indicators in the response
    negative = ["no", "not", "false", "incorrect", "wrong", "contradict", "differ"]
    if any(w in answer.lower() for w in negative):
        return "UNSUPPORTED"
    return "SUPPORTED"


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------

def measure_ooc_refusal(collection, embed_model):
    """
    Checks score-based refusal rate on the 10 OOC eval questions.
    Note: this measures only the score-based safety net (threshold=0.50).
    Full LLM-based refusal rate is measured in run_eval.py on Day 7.
    """
    refused = 0
    total   = 0
    details = []

    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["category"] != "out_of_corpus":
                continue
            chunks = retrieve(item["question"], collection, embed_model)
            top_score = chunks[0]["score"] if chunks else 0.0
            refused_flag = is_ooc(chunks)
            if refused_flag:
                refused += 1
            details.append((item["question"][:65], top_score, refused_flag))
            total += 1

    print(f"\n  {'Question':<66} {'Score':>6}  {'Refused':>8}")
    print("  " + "-" * 82)
    for q, score, ref in details:
        print(f"  {q:<66} {score:>6.4f}  {'YES' if ref else 'NO':>8}")

    return refused / total if total else 0.0


def measure_hallucination_detection():
    """
    Runs the verifier on all seeded examples and measures detection rate.
    Reports both overall accuracy and the hallucination-specific detection rate.
    """
    results = []

    with open(SEEDS_FILE, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            chunk_text = load_chunk_text(item["chunk_id"])
            if not chunk_text:
                print(f"  Warning: chunk {item['chunk_id']} not found — skipping")
                continue

            verdict   = verify_claim(item["claim"], chunk_text)
            expected  = "UNSUPPORTED" if item["label"] == "hallucination" else "SUPPORTED"
            correct   = verdict == expected
            status    = "✓" if correct else "✗"

            print(f"  {status} [{item['id']}] expected={expected:<11} got={verdict}")
            print(f"      {item['claim'][:85]}")
            results.append({
                "id":       item["id"],
                "label":    item["label"],
                "expected": expected,
                "verdict":  verdict,
                "correct":  correct,
            })

    total      = len(results)
    correct    = sum(r["correct"] for r in results)
    hall_total = sum(1 for r in results if r["label"] == "hallucination")
    hall_caught = sum(1 for r in results if r["label"] == "hallucination" and r["verdict"] == "UNSUPPORTED")

    return {
        "total":                    total,
        "overall_accuracy":         correct / total if total else 0.0,
        "hallucination_total":      hall_total,
        "hallucination_caught":     hall_caught,
        "hallucination_detection":  hall_caught / hall_total if hall_total else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FinSight guardrail measurements")
    parser.add_argument("--ooc",  action="store_true", help="Score-based OOC refusal rate only")
    parser.add_argument("--hall", action="store_true", help="Hallucination detection only")
    args = parser.parse_args()

    run_ooc  = args.ooc  or not (args.ooc or args.hall)
    run_hall = args.hall or not (args.ooc or args.hall)

    if run_ooc:
        print("\n[ Guardrail 1: Score-based OOC Safety Net ]")
        print(f"  Threshold: {OOC_THRESHOLD}  (primary OOC mechanism is LLM Rule 3)")
        print("  Loading models...")
        embed_model = SentenceTransformer(MODEL_NAME)
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        collection  = client.get_collection(COLLECTION)

        rate = measure_ooc_refusal(collection, embed_model)
        print(f"\n  Score-based refusal rate: {rate:.1%}")
        print(f"  Note: OOC questions score 0.66–0.79 due to domain overlap.")
        print(f"  Full LLM-based refusal rate (target ≥ 95%) measured in run_eval.py Day 7.")

    if run_hall:
        print("\n[ Guardrail 2: Hallucination Detection ]")
        print(f"  Running verifier on {SEEDS_FILE}")
        print(f"  8 hallucinations + 2 correct claims — ~2 minutes with Ollama\n")

        metrics = measure_hallucination_detection()

        print(f"\n  Hallucination detection rate : {metrics['hallucination_detection']:.1%}"
              f"  ({metrics['hallucination_caught']}/{metrics['hallucination_total']})"
              f"  target ≥ 80%")
        print(f"  Overall accuracy             : {metrics['overall_accuracy']:.1%}"
              f"  ({sum(1 for _ in range(metrics['total']) if True)}/{metrics['total']})")
        status = "✅ PASS" if metrics["hallucination_detection"] >= 0.80 else "❌ FAIL"
        print(f"  {status}")


if __name__ == "__main__":
    main()
