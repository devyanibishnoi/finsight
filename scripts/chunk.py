import os
import json
import re
import tiktoken
import pdfplumber
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

CORPUS_DIR    = "corpus"
OUTPUT_FILE   = "data/chunks.jsonl"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
ENCODING      = "cl100k_base"

enc = tiktoken.get_encoding(ENCODING)


def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if len(l.split()) >= 2 or l == '']
    return '\n'.join(lines).strip()


def split_into_chunks(text, doc_title, source):
    tokens = enc.encode(text)
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(tokens), step):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]

        if len(chunk_tokens) < 20:
            break

        slug = re.sub(r'[^a-z0-9]+', '_', doc_title.lower()).strip('_')[:30]

        chunks.append({
            "chunk_id":    f"{slug}_{len(chunks):04d}",
            "text":        enc.decode(chunk_tokens).strip(),
            "source":      source,
            "doc_title":   doc_title,
            "chunk_index": len(chunks),
        })

    return chunks


def assign_sections(chunks):
    """
    Post-process chunks from a single document to add a 'section' field.
    Scans the first 3 lines of each chunk for a heading pattern.
    Keeps a running current_section so chunks inherit the last-seen heading.
    Called per-document so sections don't bleed across documents.
    """
    heading_pattern = re.compile(
        r'^(?:'
        r'Chapter\s+[IVXLCDM\d]+(?:[:\s]\s*[A-Za-z][A-Za-z ,\-\(\)]{2,50})?'   # Chapter I / Chapter IV: Title (no dots, no newline bleed)
        r'|\d{1,2}\.\s+[A-Z][A-Z &\/\(\)\-]{5,50}'                            # 1. OVERVIEW — space not \s, stops at newline
        r'|\d{1,2}\.\d{1,2}\.?\s+[A-Z][A-Za-z &\/\-\(\)]{2,35}$'             # 2.1 Sub-heading — end-of-line anchor
        r'|[A-Z][A-Z \/\-&\(\)]{8,60}:'                                       # ALL CAPS HEADING: — space not \s
        r')',
        re.MULTILINE
    )

    current_section = "General"
    for chunk in chunks:
        # Strip table-of-contents leader lines (contain 4+ dots) before scanning
        # Scan the full chunk text so headings mid-chunk are not missed
        clean_text = '\n'.join(
            line for line in chunk['text'].split('\n')
            if '....' not in line
        )
        match = heading_pattern.search(clean_text)
        if match:
            raw = match.group(0).strip()
            raw = re.sub(r'\s+', ' ', raw)
            current_section = raw[:80]
        chunk['section'] = current_section
    return chunks


def process_pdf(filepath, source):
    # try pypdf first
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    # if pypdf got nothing, fall back to pdfplumber
    if len(full_text.strip()) < 100:
        print(f"  pypdf returned empty — trying pdfplumber...")
        full_text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

    # if pdfplumber also got nothing, fall back to OCR
    if len(full_text.strip()) < 100:
        print(f"  pdfplumber returned empty — trying OCR (this may take a few minutes)...")
        full_text = ""
        images = convert_from_path(filepath, dpi=300)
        for i, image in enumerate(images):
            print(f"    OCR page {i+1}/{len(images)}...")
            gray = image.convert("L")
            page_text = pytesseract.image_to_string(gray, config="--oem 3 --psm 3")
            if page_text:
                full_text += page_text + "\n"
        print(f"  OCR complete — extracted {len(full_text.strip())} characters")

    doc_title = os.path.splitext(os.path.basename(filepath))[0]
    return split_into_chunks(clean_text(full_text), doc_title, source)


def process_txt(filepath, source):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    doc_title = os.path.splitext(os.path.basename(filepath))[0]
    return split_into_chunks(clean_text(text), doc_title, source)


def main():
    all_chunks = []

    for source_folder in os.listdir(CORPUS_DIR):
        folder_path = os.path.join(CORPUS_DIR, source_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in sorted(os.listdir(folder_path)):
            filepath = os.path.join(folder_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            print(f"Processing: {filepath}")

            if ext == ".pdf":
                chunks = process_pdf(filepath, source=source_folder)
            elif ext == ".txt":
                chunks = process_txt(filepath, source=source_folder)
            else:
                continue

            chunks = assign_sections(chunks)
            print(f"  → {len(chunks)} chunks")
            all_chunks.extend(chunks)

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f"\n✅ Done! {len(all_chunks)} total chunks → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
