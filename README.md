# FinSight

A production-grade RAG system for neobanking customer support. Ask natural-language questions about policies, regulations, and procedures — FinSight retrieves the most relevant chunks from your corpus and generates grounded answers with inline citations.

No LangChain. No LlamaIndex. Everything written from scratch.

---

## What's included

The repo ships with a working example corpus:

| Source | Format | Description |
|---|---|---|
| RBI Master Directions (KYC, PPI, Digital Lending, UPI) | PDF | Public regulatory documents from rbi.org.in |
| Synthetic internal policy docs | TXT | 15 fictional neobank policy documents |
| Synthetic product notes | TXT | 10 fictional product team edge-case notes |

**You can swap in your own documents** — drop PDFs or TXTs into `corpus/` subdirectories and re-run the ingestion scripts.

---

## Prerequisites

- Python 3.11
- [Ollama](https://ollama.com) with `llama3.1:8b` pulled: `ollama pull llama3.1:8b`
- Tesseract OCR (`brew install tesseract` on Mac) — only needed to re-run corpus ingestion on scanned PDFs
- AWS Bedrock credentials if using Claude for generation — see `.env.example`

---

## Setup

```bash
# 1. Clone the repo and create a virtual environment
python3.11 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add credentials (only needed for Claude/Bedrock and RAGAS eval)
cp .env.example .env
# Edit .env and fill in your values

# 4. Chunk the corpus
python scripts/chunk.py

# 5. Embed and build the Chroma vector store
python scripts/embed.py
```

After step 5, the vector store is ready and the app can run.

---

## Run the app

Make sure Ollama is running in a separate terminal:

```bash
ollama serve
```

Start the Flask server:

```bash
python app/app.py
```

Open `http://localhost:5001` in your browser.

---

## Using your own documents

Add your documents to any subdirectory under `corpus/`, then re-run chunking and embedding:

```bash
python scripts/chunk.py   # re-ingests all files in corpus/
python scripts/embed.py   # rebuilds the vector store
```

Supported formats: PDF (text-based or scanned via OCR) and TXT.

---

## Other scripts

```bash
# Interactive retrieval — naive vector search
python scripts/retrieve.py

# Interactive retrieval — hybrid BM25 + vector + reranking
python scripts/hybrid_retrieve.py

# Interactive generation — choose ollama or claude
python scripts/generate.py

# Retrieval ablation study (chunk size, overlap, top-k, embedding model)
python scripts/ablation.py

# Guardrail measurements (OOC refusal rate + hallucination detection)
python scripts/guardrails.py
```

---

## Project structure

```
finsight/
├── corpus/
│   ├── rbi/                 # RBI Master Directions (PDF, public)
│   ├── internal_policies/   # Synthetic policy docs (TXT)
│   └── product_notes/       # Synthetic product notes (TXT)
├── data/                    # Generated — not committed
│   ├── chroma_db/           # Vector store (built by embed.py)
│   └── chunks.jsonl         # Chunked corpus (built by chunk.py)
├── scripts/
│   ├── chunk.py             # Corpus ingestion and chunking
│   ├── embed.py             # Embedding generation + Chroma storage
│   ├── retrieve.py          # Naive vector retrieval
│   ├── hybrid_retrieve.py   # BM25 + vector + cross-encoder reranking + HyDE
│   ├── generate.py          # LLM generation with citations
│   ├── guardrails.py        # OOC safety net + hallucination verifier
│   ├── ablation.py          # Retrieval ablation experiments
│   └── check_ooc_scores.py  # Diagnostic for OOC score distribution
├── app/
│   ├── app.py               # Flask backend
│   └── templates/
│       └── index.html       # Frontend
├── requirements.txt
└── .env.example
```

---

## Eval results (example corpus)

Run against 50 questions (30 in-corpus, 10 OOC, 10 adversarial) with Llama 3.1 8B and Claude 3 Haiku via AWS Bedrock.

| Metric | Target | Llama | Llama HyDE | Claude | Claude HyDE |
|---|---|---|---|---|---|
| Hit@5 | ≥ 80% | 100.0% ✅ | 100.0% ✅ | 100.0% ✅ | 100.0% ✅ |
| MRR@10 | ≥ 0.65 | 0.878 ✅ | 0.883 ✅ | 0.878 ✅ | 0.883 ✅ |
| OOC Refusal Rate | ≥ 95% | 100.0% ✅ | 100.0% ✅ | 100.0% ✅ | 100.0% ✅ |
| Citation Accuracy | ≥ 90% | 92.0% ✅ | 90.0% ✅ | 90.0% ✅ | 88.7% ❌ |
| Faithfulness (RAGAS) | ≥ 0.85 | 0.615 ❌ | 0.633 ❌ | 0.673 ❌ | 0.677 ❌ |
| Answer Relevance (RAGAS) | ≥ 0.80 | 0.835 ✅ | 0.809 ✅ | 0.867 ✅ | 0.888 ✅ |
| p95 Latency | < 3s | 17.5s ❌ | 25.3s ❌ | 4.3s ❌ | 9.4s ❌ |
| Cost per Query | Logged | $0.00 | $0.00 | $0.00305 | ~$0.00305 |

Faithfulness is below target for all runs — both models add background context not directly in the retrieved chunks. The fix is a post-generation verification step. HyDE improves MRR marginally (+0.005) but roughly doubles latency — not worth it at this corpus size.
