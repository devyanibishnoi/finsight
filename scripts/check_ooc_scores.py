"""
check_ooc_scores.py — diagnostic tool

Shows the top retrieval score for each OOC question in the eval set.
Used to understand why score-based OOC detection doesn't work for this
corpus — domain-specific questions score 0.66-0.79 even when the answer
is not in the corpus, because they share vocabulary with the documents.

Usage:
    python scripts/check_ooc_scores.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_DIR = "data/chroma_db"
COLLECTION = "finsight"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EVAL_FILE  = "eval/eval_set.jsonl"


def main():
    print("Loading embedding model...")
    model  = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)

    print(f"\n{'Score':>6}  Question")
    print("-" * 80)

    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["category"] != "out_of_corpus":
                continue
            emb   = model.encode([item["question"]], normalize_embeddings=True)[0]
            res   = collection.query(
                query_embeddings=[emb.tolist()],
                n_results=1,
                include=["distances"]
            )
            score = round(1 - res["distances"][0][0], 4)
            print(f"{score:>6.4f}  {item['question']}")

    print("\nNote: all scores are high (0.66-0.79) due to domain vocabulary overlap.")
    print("Score-based OOC detection is unreliable for this corpus.")
    print("Primary OOC mechanism is LLM Rule 3 in the system prompt.")


if __name__ == "__main__":
    main()
