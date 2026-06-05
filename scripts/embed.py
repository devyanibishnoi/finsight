import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

CHUNKS_FILE  = "data/chunks.jsonl"
CHROMA_DIR   = "data/chroma_db"
COLLECTION   = "finsight"
MODEL_NAME   = "BAAI/bge-small-en-v1.5"
BATCH_SIZE   = 64


def main():
    # Load chunks
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # Load embedding model
    print(f"\nLoading embedding model: {MODEL_NAME}")
    print("(First run will download ~130MB of model weights, cached after that)")
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings for all chunks
    texts = [c["text"] for c in chunks]
    print(f"\nGenerating embeddings for {len(texts)} chunks in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True   # required for cosine similarity with bge models
    )
    print(f"Embeddings shape: {embeddings.shape}")  # should be (516, 384)

    # Connect to Chroma
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if present so re-runs are clean
    try:
        client.delete_collection(COLLECTION)
        print(f"\nDeleted existing collection '{COLLECTION}' (clean re-run)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}  # cosine similarity as distance metric
    )

    # Add chunks to Chroma in batches
    # Chroma has an internal limit per add call so we batch at 100
    print(f"\nStoring chunks in Chroma at {CHROMA_DIR}...")
    CHROMA_BATCH = 100
    for i in range(0, len(chunks), CHROMA_BATCH):
        batch        = chunks[i : i + CHROMA_BATCH]
        batch_embeds = embeddings[i : i + CHROMA_BATCH]

        collection.add(
            ids        = [c["chunk_id"] for c in batch],
            embeddings = batch_embeds.tolist(),
            documents  = [c["text"] for c in batch],
            metadatas  = [
                {
                    "source":      c["source"],
                    "doc_title":   c["doc_title"],
                    "section":     c["section"],
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
        )
        print(f"  Stored chunks {i + 1} to {min(i + CHROMA_BATCH, len(chunks))}")

    print(f"\n✅ Done! {len(chunks)} chunks embedded and stored.")
    print(f"   Collection : '{COLLECTION}'")
    print(f"   Location   : {CHROMA_DIR}")
    print(f"   Dimensions : 384 (bge-small-en-v1.5)")
    print(f"   Metric     : cosine similarity")


if __name__ == "__main__":
    main()
