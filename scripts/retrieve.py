import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

CHROMA_DIR  = "data/chroma_db"
COLLECTION  = "finsight"
MODEL_NAME  = "BAAI/bge-small-en-v1.5"
TOP_K       = 5


def retrieve(question, collection, model, n_results=TOP_K):
    # Embed the question using the same model and settings as embed.py
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    # Query Chroma for top-k most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Chroma returns cosine distance (0 = identical, 2 = opposite)
    # Convert to similarity score (1 = identical, -1 = opposite) for readability
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id":  results["ids"][0][i],
            "text":      results["documents"][0][i],
            "source":    results["metadatas"][0][i]["source"],
            "doc_title": results["metadatas"][0][i]["doc_title"],
            "section":   results["metadatas"][0][i]["section"],
            "score":     round(1 - results["distances"][0][i], 4),
        })

    return chunks


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)

    print(f"Connected to Chroma. {collection.count()} chunks indexed.")
    print("Type a question and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue

            start   = time.time()
            results = retrieve(question, collection, model)
            elapsed = (time.time() - start) * 1000  # ms

            print(f"\nTop {TOP_K} chunks  ({elapsed:.0f}ms)\n")
            print("=" * 80)

            for i, chunk in enumerate(results, 1):
                print(f"\n[{i}] {chunk['chunk_id']}  |  score: {chunk['score']}")
                print(f"    source: {chunk['source']}  |  doc: {chunk['doc_title']}")
                print(f"    section: {chunk['section']}")
                print(f"\n{chunk['text']}")
                print("\n" + "-" * 80)

            print()

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
