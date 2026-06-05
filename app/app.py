"""
app.py — Flask backend for FinSight

Serves the UI and exposes a single /ask endpoint that runs the full
hybrid retrieval + generation pipeline and logs every query.

Run from project root:
    python app/app.py
"""

import sys
import os
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify, render_template

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

os.chdir(ROOT)

# Load .env before anything else so HF_TOKEN is set before model downloads
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

from hybrid_retrieve import hybrid_retrieve, build_bm25_index, load_chunks  # type: ignore
from guardrails import is_ooc, REFUSAL_PHRASE  # type: ignore
from generate import generate_ollama, generate_claude  # type: ignore

# --- Paths ---
CHROMA_DIR  = str(ROOT / "data" / "chroma_db")
DB_PATH     = str(ROOT / "data" / "queries.db")
TRACES_FILE = str(ROOT / "data" / "traces.jsonl")

EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Load models once at startup ---
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading cross-encoder reranker...")
cross_encoder = CrossEncoder(RERANK_MODEL)

print("Building BM25 index...")
all_chunks = load_chunks()
bm25 = build_bm25_index(all_chunks)

client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("finsight")

print(f"Ready. {collection.count()} chunks indexed.\n")

# --- Flask app ---
app = Flask(__name__, template_folder="templates")


# --- Logging ---

CACHE_TTL_SECONDS = 3600  # cache hits valid for 1 hour


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT,
            question   TEXT,
            answer     TEXT,
            model      TEXT,
            latency_ms INTEGER,
            chunk_ids  TEXT,
            session_id TEXT,
            cached     INTEGER DEFAULT 0
        )
    """)
    # Add columns to existing DB if they don't exist yet
    for col, typedef in [("session_id", "TEXT"), ("cached", "INTEGER DEFAULT 0"), ("chunks_json", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE queries ADD COLUMN {col} {typedef}")
        except Exception:
            pass
    conn.commit()
    conn.close()


def get_cached(question, model):
    """Return cached answer+chunks if the same question+model was asked within the TTL."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        """SELECT answer, chunks_json FROM queries
           WHERE question = ? AND model = ? AND cached = 0
             AND (julianday('now') - julianday(timestamp)) * 86400 < ?
           ORDER BY id DESC LIMIT 1""",
        (question, model, CACHE_TTL_SECONDS)
    ).fetchone()
    conn.close()
    return row  # (answer, chunks_json) or None


def log_query(question, answer, model, latency_ms, chunks, session_id=None, cached=0):
    chunk_ids  = ",".join(c["chunk_id"] for c in chunks)
    chunks_json = json.dumps([
        {"chunk_id": c["chunk_id"], "source": c["source"],
         "section": c["section"], "score": round(c.get("rerank_score", c.get("score", 0)), 4)}
        for c in chunks
    ])

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO queries
           (timestamp, question, answer, model, latency_ms, chunk_ids, session_id, cached, chunks_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now(timezone.utc).isoformat(), question, answer, model,
         latency_ms, chunk_ids, session_id, cached, chunks_json)
    )
    conn.commit()
    conn.close()

    # JSONL trace
    trace = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "question":   question,
        "answer":     answer,
        "model":      model,
        "latency_ms": latency_ms,
        "chunks":     [
            {"chunk_id": c["chunk_id"], "source": c["source"],
             "section": c["section"], "score": c.get("rerank_score", c.get("score", 0))}
            for c in chunks
        ],
    }
    with open(TRACES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")


# --- Query rewriting ---

FOLLOWUP_TRIGGERS = {
    "it", "that", "this", "they", "them", "those", "these", "its",
    "what about", "how about", "and what", "and how", "also", "same",
    "similar", "different", "else", "instead", "other", "another"
}

def looks_like_followup(question):
    """
    Returns True if the question is likely a follow-up — short or contains
    reference words that only make sense in context.
    Standalone questions skip the rewriter entirely.
    """
    words = question.lower().split()
    if len(words) <= 6:
        return True
    q_lower = question.lower()
    return any(trigger in q_lower for trigger in FOLLOWUP_TRIGGERS)


def rewrite_question(question, history):
    """
    Rewrites follow-up questions into standalone queries using conversation history.
    Only runs when history exists AND the question looks like a follow-up.
    Standalone questions are returned unchanged without calling Ollama.
    """
    if not history or not looks_like_followup(question):
        return question

    history_text = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer'][:300]}"
        for turn in history[-3:]  # last 3 turns is enough context
    )

    prompt = f"""Given this conversation history, rewrite the latest question as a fully self-contained question a search engine can understand. If it already stands alone, return it completely unchanged. Return ONLY the rewritten question — no explanation, no prefix, no quotes.

Conversation history:
{history_text}

Latest question: {question}"""

    import ollama as _ollama
    response = _ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}]
    )
    rewritten = response["message"]["content"].strip()
    return rewritten if rewritten else question


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data            = request.get_json(silent=True) or {}
    question        = data.get("question", "").strip()
    model           = data.get("model", "ollama")
    compliance_mode = bool(data.get("compliance_mode", False))
    session_id      = data.get("session_id") or None
    history         = data.get("history", [])  # list of {question, answer} dicts

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # --- Rewrite follow-up questions into standalone queries ---
        rewritten = rewrite_question(question, history)

        # --- Cache check (always use original question as key; skip for compliance mode) ---
        if not compliance_mode:
            cached_row = get_cached(question, model)
            if cached_row:
                cached_answer, cached_chunks_json = cached_row
                cached_chunks = json.loads(cached_chunks_json) if cached_chunks_json else []
                log_query(question, cached_answer, model, 0, [], session_id=session_id, cached=1)
                return jsonify({
                    "answer":     cached_answer,
                    "latency_ms": 0,
                    "model":      model,
                    "cached":     True,
                    "rewritten":  rewritten if rewritten != question else None,
                    "chunks":     cached_chunks,
                })

        t0 = time.time()

        chunks = hybrid_retrieve(
            rewritten, collection, embed_model, bm25, all_chunks, cross_encoder
        )

        # Score-based OOC safety net (threshold 0.50)
        if is_ooc(chunks):
            answer = REFUSAL_PHRASE
        elif model == "claude":
            answer, _ = generate_claude(question, chunks, compliance_mode=compliance_mode)
        else:
            answer, _ = generate_ollama(question, chunks, compliance_mode=compliance_mode)

        latency_ms = round((time.time() - t0) * 1000)

        log_query(question, answer, model, latency_ms, chunks, session_id=session_id, cached=0)

        return jsonify({
            "answer":     answer,
            "latency_ms": latency_ms,
            "model":      model,
            "cached":     False,
            "rewritten":  rewritten if rewritten != question else None,
            "chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "source":   c["source"],
                    "section":  c["section"],
                    "score":    round(c.get("rerank_score", c.get("score", 0)), 4),
                }
                for c in chunks
            ],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify([])
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT question, timestamp FROM queries
           WHERE session_id = ? AND cached = 0
           ORDER BY id DESC LIMIT 5""",
        (session_id,)
    ).fetchall()
    conn.close()
    return jsonify([{"question": r[0], "timestamp": r[1]} for r in rows])


if __name__ == "__main__":
    init_db()
    print("http://localhost:5001\n")
    app.run(debug=False, port=5001)
