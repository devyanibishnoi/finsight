"""
ablation.py — Day 5

Runs four retrieval ablations and prints a comparison table:
  1. Chunk size      : 256 / 512 / 1024 tokens
  2. Overlap         : 5% / 10% / 20% of chunk size
  3. Top-k           : k=3 / k=5 / k=10
  4. Embedding model : bge-small-en-v1.5 vs bge-base-en-v1.5

Results are printed to the terminal — copy them into WRITEUP.md.
Temporary Chroma collections are deleted after each experiment.

Usage:
    python scripts/ablation.py
"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(__file__))

from sentence_transformers import SentenceTransformer
import chromadb
import tiktoken

CORPUS_DIR  = "corpus"
EVAL_FILE   = "eval/eval_set.jsonl"
CHROMA_DIR  = "data/chroma_db"
ENCODING    = "cl100k_base"

enc = tiktoken.get_encoding(ENCODING)


# ---------------------------------------------------------------------------
# Chunking (inline — avoids modifying chunk.py)
# ---------------------------------------------------------------------------

def clean_text(text):
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if len(l.split()) >= 2 or l == '']
    return '\n'.join(lines).strip()


def chunk_text(text, doc_title, source, chunk_size, overlap):
    tokens = enc.encode(text)
    chunks = []
    step   = chunk_size - overlap

    for start in range(0, len(tokens), step):
        end          = start + chunk_size
        chunk_tokens = tokens[start:end]
        if len(chunk_tokens) < 20:
            break
        slug = re.sub(r'[^a-z0-9]+', '_', doc_title.lower()).strip('_')[:30]
        chunks.append({
            "chunk_id":  f"{slug}_{len(chunks):04d}",
            "text":      enc.decode(chunk_tokens).strip(),
            "source":    source,
            "doc_title": doc_title,
        })
    return chunks


def load_corpus(chunk_size, overlap):
    """Re-chunk the full corpus with given chunk_size and overlap."""
    import pdfplumber
    from pypdf import PdfReader

    all_chunks = []
    for source_folder in os.listdir(CORPUS_DIR):
        folder_path = os.path.join(CORPUS_DIR, source_folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in sorted(os.listdir(folder_path)):
            filepath = os.path.join(folder_path, filename)
            ext      = os.path.splitext(filename)[1].lower()
            doc_title = os.path.splitext(filename)[0]

            if ext == ".txt":
                with open(filepath, encoding="utf-8") as f:
                    text = f.read()
            elif ext == ".pdf":
                reader    = PdfReader(filepath)
                text      = ""
                for page in reader.pages:
                    pt = page.extract_text()
                    if pt:
                        text += pt + "\n"
                if len(text.strip()) < 100:
                    with pdfplumber.open(filepath) as pdf:
                        text = ""
                        for page in pdf.pages:
                            pt = page.extract_text()
                            if pt:
                                text += pt + "\n"
            else:
                continue

            chunks = chunk_text(clean_text(text), doc_title, source_folder, chunk_size, overlap)
            all_chunks.extend(chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Embed + store in a temporary Chroma collection
# ---------------------------------------------------------------------------

def embed_and_store(chunks, collection_name, model):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True,
                              show_progress_bar=False)

    BATCH = 100
    for i in range(0, len(chunks), BATCH):
        batch  = chunks[i:i + BATCH]
        embeds = embeddings[i:i + BATCH]
        collection.add(
            ids        = [c["chunk_id"] for c in batch],
            embeddings = embeds.tolist(),
            documents  = [c["text"]     for c in batch],
            metadatas  = [{"source": c["source"], "doc_title": c["doc_title"],
                           "section": "", "chunk_index": j}
                          for j, c in enumerate(batch)],
        )
    return collection


def delete_collection(collection_name):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Hit@5 and MRR@10 on the eval set
# ---------------------------------------------------------------------------

def vector_retrieve(question, collection, model, n_results):
    embedding = model.encode([question], normalize_embeddings=True)[0]
    results   = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return results["ids"][0]


def hit_at_k(collection, model, k=5):
    hits = 0
    total = 0
    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["category"] != "in_corpus":
                continue
            retrieved = vector_retrieve(item["question"], collection, model, n_results=k)
            if any(gt in retrieved for gt in item["ground_truth_chunks"]):
                hits += 1
            total += 1
    return hits / total if total else 0.0


def mrr_at_k(collection, model, k=10):
    rrs = []
    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["category"] != "in_corpus":
                continue
            retrieved = vector_retrieve(item["question"], collection, model, n_results=k)
            rr = 0.0
            for rank, cid in enumerate(retrieved, 1):
                if cid in item["ground_truth_chunks"]:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)
    return sum(rrs) / len(rrs) if rrs else 0.0


# ---------------------------------------------------------------------------
# Individual ablations
# ---------------------------------------------------------------------------

def ablation_chunk_size(base_model):
    print("\n[ Ablation 1: Chunk Size ]")
    print("  Chunk sizes: 256 / 512 / 1024 tokens (overlap = 10% of chunk size)")
    results = []

    for size in [256, 512, 1024]:
        overlap = size // 10
        name    = f"finsight_ablation_cs{size}"
        print(f"  → Chunking at {size} tokens (overlap={overlap})...", end=" ", flush=True)
        chunks = load_corpus(size, overlap)
        print(f"{len(chunks)} chunks. Embedding...", end=" ", flush=True)
        col = embed_and_store(chunks, name, base_model)
        h5  = hit_at_k(col, base_model, k=5)
        m10 = mrr_at_k(col, base_model, k=10)
        print(f"Hit@5={h5:.1%}  MRR@10={m10:.3f}")
        results.append((size, overlap, len(chunks), h5, m10))
        delete_collection(name)

    print()
    print(f"  {'Chunk Size':<12} {'Overlap':<10} {'# Chunks':<10} {'Hit@5':<10} {'MRR@10'}")
    print("  " + "-" * 55)
    for size, overlap, n, h5, m10 in results:
        marker = " ← current" if size == 512 else ""
        print(f"  {size:<12} {overlap:<10} {n:<10} {h5:<10.1%} {m10:.3f}{marker}")
    return results


def ablation_overlap(base_model):
    print("\n[ Ablation 2: Overlap ]")
    print("  Chunk size fixed at 512. Overlap: 5% / 10% / 20%")
    results = []

    for pct, overlap in [(5, 25), (10, 50), (20, 100)]:
        name = f"finsight_ablation_ov{pct}"
        print(f"  → Overlap {pct}% ({overlap} tokens)...", end=" ", flush=True)
        chunks = load_corpus(512, overlap)
        print(f"{len(chunks)} chunks. Embedding...", end=" ", flush=True)
        col = embed_and_store(chunks, name, base_model)
        h5  = hit_at_k(col, base_model, k=5)
        m10 = mrr_at_k(col, base_model, k=10)
        print(f"Hit@5={h5:.1%}  MRR@10={m10:.3f}")
        results.append((pct, overlap, len(chunks), h5, m10))
        delete_collection(name)

    print()
    print(f"  {'Overlap %':<12} {'Tokens':<10} {'# Chunks':<10} {'Hit@5':<10} {'MRR@10'}")
    print("  " + "-" * 55)
    for pct, overlap, n, h5, m10 in results:
        marker = " ← current" if pct == 10 else ""
        print(f"  {pct}%{'':<10} {overlap:<10} {n:<10} {h5:<10.1%} {m10:.3f}{marker}")
    return results


def ablation_top_k(base_model):
    print("\n[ Ablation 3: Top-k ]")
    print("  Using current 512-token chunks. Testing k=3 / k=5 / k=10")

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("finsight")
    results    = []

    for k in [3, 5, 10]:
        h5  = hit_at_k(collection, base_model, k=k)
        m10 = mrr_at_k(collection, base_model, k=10)
        marker = " ← current" if k == 5 else ""
        print(f"  → k={k}: Hit@{k}={h5:.1%}  MRR@10={m10:.3f}{marker}")
        results.append((k, h5, m10))

    print()
    print(f"  {'k':<10} {'Hit@k':<10} {'MRR@10'}")
    print("  " + "-" * 30)
    for k, h5, m10 in results:
        marker = " ← current" if k == 5 else ""
        print(f"  {k:<10} {h5:<10.1%} {m10:.3f}{marker}")
    return results


def ablation_embedding_model():
    print("\n[ Ablation 4: Embedding Model ]")
    print("  bge-small-en-v1.5 (33M params) vs bge-base-en-v1.5 (109M params)")
    print("  Note: bge-base download is ~430MB on first run.\n")
    results = []

    for model_name, label in [
        ("BAAI/bge-small-en-v1.5", "bge-small (33M)"),
        ("BAAI/bge-base-en-v1.5",  "bge-base  (109M)"),
    ]:
        name = f"finsight_ablation_{'small' if 'small' in model_name else 'base'}"
        print(f"  → Loading {label}...", end=" ", flush=True)
        model = SentenceTransformer(model_name)
        print(f"Chunking + embedding...", end=" ", flush=True)
        chunks = load_corpus(512, 50)
        col    = embed_and_store(chunks, name, model)
        h5     = hit_at_k(col, model, k=5)
        m10    = mrr_at_k(col, model, k=10)
        print(f"Hit@5={h5:.1%}  MRR@10={m10:.3f}")
        results.append((label, h5, m10))
        delete_collection(name)

    print()
    print(f"  {'Model':<25} {'Hit@5':<10} {'MRR@10'}")
    print("  " + "-" * 45)
    for label, h5, m10 in results:
        marker = " ← current" if "small" in label else ""
        print(f"  {label:<25} {h5:<10.1%} {m10:.3f}{marker}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("  FinSight — Retrieval Ablation Study")
    print("=" * 62)
    print("\nLoading base embedding model (bge-small-en-v1.5)...")
    base_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    ablation_chunk_size(base_model)
    ablation_overlap(base_model)
    ablation_top_k(base_model)
    ablation_embedding_model()

    print("\n" + "=" * 62)
    print("  Done. Copy results into WRITEUP.md experiments table.")
    print("=" * 62)


if __name__ == "__main__":
    main()
