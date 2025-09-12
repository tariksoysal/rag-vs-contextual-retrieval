"""Generate LLM summaries for document chunks to build a contextual index."""

import json
import os
import multiprocessing
import time
from pathlib import Path
from tqdm import tqdm
from ollama import Client

# Input chunks come from the basic chunking step; we will write new summarized
# chunks suitable for the contextual retrieval pipeline.
INPUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data/processed/chunked_documents_train.jsonl"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data/processed/chunked_contextual_train.jsonl"
)
MODEL_NAME = "gemma3:latest"
NUM_WORKERS = 8

client = Client()

def summarize(text):
    """Ask the Ollama model for a search‑optimized summary of ``text``."""
    prompt = f"""
You are a helpful assistant. Your job is to generate a search-optimized summary of a passage.

Passage:
{text.strip()}

Summary (one paragraph, suitable for document search):
"""
    response = client.chat(model=MODEL_NAME, messages=[
        {"role": "user", "content": prompt.strip()}
    ])
    return response['message']['content'].strip()

def process_chunk(line):
    """Summarize a single JSONL line representing a chunk."""
    try:
        item = json.loads(line)
        summary = summarize(item["chunk"])
        item["chunk"] = summary
        return json.dumps(item)
    except Exception:
        return None

def already_processed_chunks(output_file: Path):
    """Return a set of (id, chunk_id) pairs already written to ``output_file``."""
    if not output_file.exists():
        return set()
    with open(output_file, "r", encoding="utf-8") as f:
        return set((json.loads(line)["id"], json.loads(line).get("chunk_id", 0)) for line in f)

def main():
    processed_chunks = already_processed_chunks(OUTPUT_PATH)

    # Filter out chunks we've already summarized so the script can be resumed
    with open(INPUT_PATH, "r", encoding="utf-8") as infile:
        lines = [
            line for line in infile
            if (json.loads(line)["id"], json.loads(line).get("chunk_id", 0)) not in processed_chunks
        ]

    print(f"🚀 Generating summaries for {len(lines)} chunks with {NUM_WORKERS} workers...")

    with multiprocessing.Pool(NUM_WORKERS) as pool, open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for result in tqdm(pool.imap(process_chunk, lines), total=len(lines), desc="⏱️ Chunking", dynamic_ncols=True):
            if result:
                out.write(result + "\n")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    elapsed = end - start
    print(f"\n✅ Finished in {elapsed / 60:.2f} minutes.")
