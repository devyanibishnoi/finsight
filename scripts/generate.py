import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from sentence_transformers import SentenceTransformer
import chromadb
import ollama
from retrieve import retrieve
from guardrails import is_ooc, REFUSAL_PHRASE

CHROMA_DIR   = "data/chroma_db"
COLLECTION   = "finsight"
MODEL_NAME   = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are FinSight, a customer support assistant for a neobanking platform.

You answer customer and support agent questions strictly using the context passages provided below. Each passage is labelled with its chunk ID.

RULES — follow these exactly, without exception:

1. CITATIONS: Every factual claim in your answer must be followed by the chunk ID in square brackets. Example: "The daily UPI limit for a Tier 1 account is ₹1 lakh [upi_limit_policy_0002]." If a sentence draws from two chunks, cite both.

2. CONTEXT ONLY: Use only the information in the provided passages. Do not use your training knowledge about banking, RBI regulations, or anything else — even if you are confident it is correct. If the answer is not in the passages, do not guess.

3. REFUSAL: If the topic of the question is not covered in the provided passages — even if you know the answer from your training knowledge — respond with this exact phrase and nothing else. Do not add any explanation, context, or additional sentences before or after it:
   "I'm sorry, I don't have enough information to answer that reliably. Please contact our support team."

4. TONE: Be professional, clear, and direct. No filler phrases like "Great question!" or "Certainly!". Answer the question and stop.

5. FORMAT: Use short paragraphs or bullet points for multi-part answers. Do not write walls of text.

6. PRECISION: Quote all amounts, limits, timeframes, and percentages exactly as they appear in the source passage. Do not round, approximate, or paraphrase numerical values.

7. NO INFERENCE: Do not infer, extrapolate, or combine information across passages to reach a conclusion that no single passage explicitly states. If the answer requires a logical leap beyond what is written, treat it as unanswered and use the refusal phrase.

8. URGENT ESCALATION: Add the line "If this is happening right now, please call our 24/7 support line immediately or freeze your card through the app." ONLY when the customer is personally reporting or suspecting an incident — e.g. "I think someone used my card", "there's a charge I didn't make", "my account was accessed", "I received a phishing message", "my card was stolen". Do NOT add it for informational or policy questions — e.g. "how do conversion rates work", "what is the transaction limit", "what does RBI say about X". The trigger is the customer describing something happening to them, not asking how something works.
"""

COMPLIANCE_PROMPT = """You are FinSight operating in COMPLIANCE MODE for a neobanking platform.

You answer customer and support agent questions strictly using the context passages provided below. Each passage is labelled with its chunk ID.

RULES — follow these exactly, without exception:

1. CITATIONS: Every factual claim in your answer must be followed by the chunk ID in square brackets. Example: "The daily UPI limit for a Tier 1 account is ₹1 lakh [upi_limit_policy_0002]." If a sentence draws from two chunks, cite both.

2. CONTEXT ONLY: Use only the information in the provided passages. Do not use outside knowledge under any circumstances — even if you are certain it is correct.

3. REFUSAL: If the topic of the question is not covered in the provided passages, respond with this exact phrase and nothing else:
   "I'm sorry, I don't have enough information to answer that reliably. Please contact our support team."

4. VERBATIM: Quote the source text verbatim where possible. Do not paraphrase regulatory or policy language — use the exact words from the chunk.

5. NO INFERENCE: Do not infer, interpret, or extrapolate. If the chunk says X, say X and nothing more. Do not draw conclusions that require any logical step beyond what is explicitly written.

6. REGULATORY QUESTIONS: Do not answer questions about legal obligations, regulatory requirements, or compliance matters unless the exact answer is stated word-for-word in the provided chunks.

7. ALL OR NOTHING: If any part of a question cannot be answered directly from the chunks, refuse the entire question. Do not partially answer.

8. URGENT ESCALATION: Add the line "If this is happening right now, please call our 24/7 support line immediately or freeze your card through the app." ONLY when the customer is personally reporting or suspecting an incident — e.g. "I think someone used my card", "there's a charge I didn't make", "my account was accessed", "I received a phishing message", "my card was stolen". Do NOT add it for informational or policy questions — e.g. "how do conversion rates work", "what is the transaction limit", "what does RBI say about X". The trigger is the customer describing something happening to them, not asking how something works.
"""


def format_context(chunks):
    """Format retrieved chunks into labelled passages for the LLM."""
    passages = []
    for chunk in chunks:
        passages.append(f"[{chunk['chunk_id']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(passages)


def generate_ollama(question, chunks, compliance_mode=False):
    """Send question + retrieved context to Llama 3.1 8B via Ollama."""
    context = format_context(chunks)
    user_message = f"Context passages:\n\n{context}\n\nQuestion: {question}"
    prompt = COMPLIANCE_PROMPT if compliance_mode else SYSTEM_PROMPT

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": user_message},
        ]
    )
    return response["message"]["content"], 0.0


def generate_claude(question, chunks, compliance_mode=False):
    """Send question + retrieved context to Claude Haiku via AWS Bedrock."""
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()

    context = format_context(chunks)
    user_message = f"Context passages:\n\n{context}\n\nQuestion: {question}"
    prompt = COMPLIANCE_PROMPT if compliance_mode else SYSTEM_PROMPT

    client = anthropic.AnthropicBedrock(
        aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        aws_region=os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"),
    )
    message = client.messages.create(
        model="apac.anthropic.claude-3-haiku-20240307-v1:0",
        max_tokens=1024,
        system=prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    # Log token usage for cost tracking
    input_tokens  = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    cost = (input_tokens * 0.80 + output_tokens * 4.00) / 1_000_000
    print(f"  [tokens] in: {input_tokens}  out: {output_tokens}  cost: ${cost:.5f}")

    return message.content[0].text, round(cost, 6)


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)

    print(f"Connected to Chroma. {collection.count()} chunks indexed.")
    print("\nWhich model? Type 'ollama' or 'claude' (claude requires API key).")
    backend = input("Model [ollama]: ").strip().lower() or "ollama"

    if backend == "claude":
        generate_fn = generate_claude
        print("Using Claude Haiku 4.5")
    else:
        generate_fn = generate_ollama
        print("Using Llama 3.1 8B (Ollama)")

    print("\nType a question and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue

            # Retrieve
            t0     = time.time()
            chunks = retrieve(question, collection, model)
            t1     = time.time()

            # Score-based OOC safety net (threshold=0.50)
            # Primary OOC mechanism is LLM Rule 3 in the system prompt
            if is_ooc(chunks):
                print(f"\n{'=' * 80}")
                print(REFUSAL_PHRASE)
                print(f"[guardrail: top score {chunks[0]['score'] if chunks else 0:.4f} < 0.50]")
                print(f"{'=' * 80}\n")
                continue

            # Generate
            answer, _ = generate_fn(question, chunks)
            t2        = time.time()

            retrieval_ms  = (t1 - t0) * 1000
            generation_ms = (t2 - t1) * 1000

            print(f"\n{'=' * 80}")
            print(answer)
            print(f"{'=' * 80}")
            print(f"\nRetrieval: {retrieval_ms:.0f}ms  |  Generation: {generation_ms:.0f}ms")
            print("\nChunks used:")
            for chunk in chunks:
                print(f"  [{chunk['chunk_id']}]  score: {chunk['score']}  |  {chunk['source']}")
            print()

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
