"""
hybrid_retrieve.py — Day 5

Retrieval pipeline: BM25 + vector search → RRF fusion → cross-encoder reranking.

Usage:
    python scripts/hybrid_retrieve.py           # interactive mode
    python scripts/hybrid_retrieve.py --eval    # Hit@5 / MRR@10 comparison vs naive
"""

import sys
import os
import json
import time
import argparse
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from retrieve import retrieve

# --- Constants ---
CHROMA_DIR    = "data/chroma_db"
COLLECTION    = "finsight"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNKS_FILE   = "data/chunks.jsonl"
EVAL_FILE     = "eval/eval_set.jsonl"
TOP_K         = 5
CANDIDATE_K   = 20   # candidates fetched from each method before reranking
RRF_K         = 60   # RRF constant — prevents top ranks dominating


# --- BM25 ---

def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_bm25_index(chunks):
    """Tokenise chunk texts and build a BM25Okapi index."""
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def bm25_search(query, chunks, bm25, top_k=CANDIDATE_K):
    """Return top_k chunks ranked by BM25 keyword score."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {**chunks[i], "bm25_score": round(float(scores[i]), 4)}
        for i in ranked_indices
    ]


# --- RRF Fusion ---

def rrf_fusion(bm25_results, vector_results, k=RRF_K):
    """
    Combine two ranked lists using Reciprocal Rank Fusion.

    RRF score for a chunk = 1/(rank_in_bm25 + k) + 1/(rank_in_vector + k)
    Chunks that rank highly in both lists score highest.
    Raw scores are ignored — only rank positions matter.
    """
    scores = {}

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        if cid not in scores:
            scores[cid] = {"chunk": chunk, "rrf": 0.0}
        scores[cid]["rrf"] += 1.0 / (rank + k)

    for rank, chunk in enumerate(vector_results):
        cid = chunk["chunk_id"]
        if cid not in scores:
            scores[cid] = {"chunk": chunk, "rrf": 0.0}
        scores[cid]["rrf"] += 1.0 / (rank + k)

    ranked = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
    return [item["chunk"] for item in ranked]


# --- Cross-encoder reranking ---

def rerank(query, candidates, cross_encoder, top_k=TOP_K):
    """
    Score each (query, chunk) pair with a cross-encoder and return top_k.

    A cross-encoder reads the query and chunk together — more accurate than
    the bi-encoder (which encodes them separately) but too slow to run on
    all 516 chunks, so we only run it on the CANDIDATE_K fused results.
    """
    pairs  = [(query, c["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [
        {**chunk, "rerank_score": round(float(score), 4)}
        for score, chunk in ranked[:top_k]
    ]


# --- Full hybrid pipeline ---

def hybrid_retrieve(question, collection, embed_model, bm25, all_chunks, cross_encoder, top_k=TOP_K):
    """
    Full pipeline:
      1. Vector search  — semantic similarity (top CANDIDATE_K)
      2. BM25 search    — keyword matching   (top CANDIDATE_K)
      3. RRF fusion     — merge both ranked lists by rank position
      4. Cross-encoder  — rerank fused candidates, return top_k
    """
    vector_results = retrieve(question, collection, embed_model, n_results=CANDIDATE_K)
    bm25_results   = bm25_search(question, all_chunks, bm25, top_k=CANDIDATE_K)
    fused          = rrf_fusion(bm25_results, vector_results)
    return rerank(question, fused[:CANDIDATE_K], cross_encoder, top_k=top_k)


# --- HyDE pipeline ---

def generate_hypothetical_doc(question):
    """
    Ask Llama to write a short hypothetical policy passage that would answer
    the question. This passage uses the same vocabulary as real corpus chunks,
    so embedding it produces a better search vector than embedding the question.
    """
    import ollama as _ollama
    prompt = (
        "Write a short factual passage (2-3 sentences) from a neobanking policy "
        "document that directly answers the following question. Use specific numbers "
        "and policy language. Do not explain — just write the passage.\n\n"
        f"Question: {question}"
    )
    response = _ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


def hyde_retrieve(question, collection, embed_model, bm25, all_chunks, cross_encoder, top_k=TOP_K):
    """
    HyDE pipeline — same as hybrid_retrieve but the vector search uses an
    embedding of a hypothetical answer instead of the raw question.

    BM25 and cross-encoder still use the original question — only the
    embedding query is swapped.
    """
    hypothetical_doc   = generate_hypothetical_doc(question)
    vector_results     = retrieve(hypothetical_doc, collection, embed_model, n_results=CANDIDATE_K)
    bm25_results       = bm25_search(question, all_chunks, bm25, top_k=CANDIDATE_K)
    fused              = rrf_fusion(bm25_results, vector_results)
    return rerank(question, fused[:CANDIDATE_K], cross_encoder, top_k=top_k)


# --- Eval metrics ---

def measure_hit_at_k(eval_file, retrieve_fn, k=5):
    """
    Hit@k on in-corpus questions.
    A question is a hit if any ground-truth chunk appears in the top-k results.
    """
    hits = 0
    total = 0
    with open(eval_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["category"] != "in_corpus":
                continue
            results      = retrieve_fn(item["question"])
            retrieved_ids = [r["chunk_id"] for r in results[:k]]
            if any(gt in retrieved_ids for gt in item["ground_truth_chunks"]):
                hits += 1
            total += 1
    return hits / total if total else 0.0



def measure_mrr_at_k(eval_file, retrieve_fn, k=10):
    """
    MRR@k on in-corpus questions.
    Reciprocal rank of the first ground-truth chunk in the top-k results.
    """
    rrs = []
    with open(eval_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["category"] != "in_corpus":
                continue
            results      = retrieve_fn(item["question"])[:k]
            retrieved_ids = [r["chunk_id"] for r in results]
            rr = 0.0
            for rank, cid in enumerate(retrieved_ids, 1):
                if cid in item["ground_truth_chunks"]:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)
    return sum(rrs) / len(rrs) if rrs else 0.0


def run_eval_comparison(collection, embed_model, bm25, all_chunks, cross_encoder, include_hyde=False):
    """Run Hit@5 and MRR@10 for naive vs hybrid (vs HyDE) and print a delta table."""
    print("\nRunning retrieval comparison on 30 in-corpus eval questions...")
    print("(This will take ~60s for hybrid, plus ~3 min extra if HyDE is enabled)\n")

    def naive_fn(q):
        return retrieve(q, collection, embed_model, n_results=10)

    def hybrid_fn(q):
        return hybrid_retrieve(q, collection, embed_model, bm25, all_chunks, cross_encoder)

    def hyde_fn(q):
        return hyde_retrieve(q, collection, embed_model, bm25, all_chunks, cross_encoder)

    print("  Measuring naive vector retrieval...")
    naive_hit5  = measure_hit_at_k(EVAL_FILE, naive_fn,  k=5)
    naive_mrr10 = measure_mrr_at_k(EVAL_FILE, naive_fn,  k=10)

    print("  Measuring hybrid + reranking retrieval...")
    hybrid_hit5  = measure_hit_at_k(EVAL_FILE, hybrid_fn, k=5)
    hybrid_mrr10 = measure_mrr_at_k(EVAL_FILE, hybrid_fn, k=10)

    results = {
        "naive_hit5":   round(naive_hit5, 4),
        "naive_mrr10":  round(naive_mrr10, 4),
        "hybrid_hit5":  round(hybrid_hit5, 4),
        "hybrid_mrr10": round(hybrid_mrr10, 4),
    }

    if include_hyde:
        print("  Measuring HyDE retrieval (calls Ollama for each question)...")
        hyde_hit5  = measure_hit_at_k(EVAL_FILE, hyde_fn, k=5)
        hyde_mrr10 = measure_mrr_at_k(EVAL_FILE, hyde_fn, k=10)
        results["hyde_hit5"]  = round(hyde_hit5, 4)
        results["hyde_mrr10"] = round(hyde_mrr10, 4)

    print("\n" + "=" * 75)
    if include_hyde:
        print(f"{'Metric':<20} {'Naive':>12} {'Hybrid':>12} {'HyDE':>12} {'HyDE Delta':>10}")
        print("-" * 75)
        print(f"{'Hit@5':<20} {naive_hit5:>11.1%} {hybrid_hit5:>11.1%} {hyde_hit5:>11.1%} {hyde_hit5 - hybrid_hit5:>+9.1%}")
        print(f"{'MRR@10':<20} {naive_mrr10:>11.3f} {hybrid_mrr10:>11.3f} {hyde_mrr10:>11.3f} {hyde_mrr10 - hybrid_mrr10:>+9.3f}")
    else:
        print(f"{'Metric':<20} {'Naive (Day 2)':>15} {'Hybrid (Day 5)':>15} {'Delta':>8}")
        print("-" * 75)
        print(f"{'Hit@5':<20} {naive_hit5:>14.1%} {hybrid_hit5:>14.1%} {hybrid_hit5 - naive_hit5:>+7.1%}")
        print(f"{'MRR@10':<20} {naive_mrr10:>14.3f} {hybrid_mrr10:>14.3f} {hybrid_mrr10 - naive_mrr10:>+7.3f}")
    print("=" * 75)

    return results


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="FinSight hybrid retrieval")
    parser.add_argument("--eval", action="store_true",
                        help="Run Hit@5 / MRR@10 comparison against the eval set")
    parser.add_argument("--hyde", action="store_true",
                        help="Include HyDE in the eval comparison (adds ~3 min)")
    args = parser.parse_args()

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Loading cross-encoder reranker (first run downloads ~80MB)...")
    cross_encoder = CrossEncoder(RERANK_MODEL)

    print("Building BM25 index...")
    all_chunks = load_chunks()
    bm25       = build_bm25_index(all_chunks)
    print(f"  BM25 index built over {len(all_chunks)} chunks.")

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)
    print(f"  Chroma connected. {collection.count()} chunks indexed.\n")

    if args.eval:
        run_eval_comparison(collection, embed_model, bm25, all_chunks, cross_encoder,
                            include_hyde=args.hyde)
        return

    print("Hybrid search active (BM25 + vector + cross-encoder reranking).")
    print("Type a question and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue

            start   = time.time()
            results = hybrid_retrieve(question, collection, embed_model, bm25, all_chunks, cross_encoder)
            elapsed = (time.time() - start) * 1000

            print(f"\nTop {TOP_K} chunks  ({elapsed:.0f}ms)\n")
            print("=" * 80)

            for i, chunk in enumerate(results, 1):
                print(f"\n[{i}] {chunk['chunk_id']}  |  rerank score: {chunk.get('rerank_score', 'N/A')}")
                print(f"    source: {chunk['source']}  |  section: {chunk['section']}")
                print(f"\n{chunk['text'][:400]}...")
                print("\n" + "-" * 80)

            print()

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
