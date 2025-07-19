import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

# Defaults (fallback if not provided as CLI args)
DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data/processed/combined.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data/processed/chunked_documents.jsonl"
)

CHUNK_SIZE = 1000  # characters

def clean_html(text):
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'&#xA;|&nbsp;', ' ', text)  # replace common HTML entities
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_PATH

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    chunks = []
    for item in tqdm(data, desc="Chunking documents"):
        qid = item["id"]
        source = item.get("source", "unknown")

        text = clean_html(item.get("title", "") + "\n\n" + item.get("body", ""))
        chunked = chunk_text(text)

        for i, c in enumerate(chunked):
            chunks.append({
                "id": qid,
                "chunk_id": i,
                "chunk": c,
                "source": source
            })

    with open(output_path, "w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(chunk) + "\n")

    print(f"✅ Saved {len(chunks)} chunks to {output_path}")

if __name__ == "__main__":
    main()
