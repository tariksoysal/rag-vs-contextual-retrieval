import json
import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_FILE = "data/processed/chunked_documents.jsonl"
OUTPUT_FILE = "data/processed/chunked_contextual.jsonl"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3"
MAX_WORKERS = 6  # Adjust based on your system's capabilities

PROMPT_TEMPLATE = """You are summarizing a passage for use in document search. Here is a chunk from a longer document. Write a short summary that helps situate this chunk in its original context:

{chunk}

Summary:"""

def get_summary(chunk_text):
    prompt = PROMPT_TEMPLATE.format(chunk=chunk_text)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return ""

# Step 1: Load existing output (if any) to avoid recomputing
processed_ids = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                processed_ids.add((item["id"], item["chunk_id"]))
            except:
                continue

# Step 2: Load all input chunks
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

# Step 3: Filter only those that haven't been processed
pending_chunks = [c for c in chunks if (c["id"], c["chunk_id"]) not in processed_ids]

print(f"✅ Skipping {len(chunks) - len(pending_chunks)} already processed chunks.")
print(f"🚀 Generating summaries for {len(pending_chunks)} chunks with {MAX_WORKERS} workers...")

# Step 4: Parallel summarization
with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_summary, item["chunk"]): item
            for item in pending_chunks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating summaries"):
            item = futures[future]
            summary = future.result()
            enriched = f"{summary}\n\n{item['chunk']}".strip()
            enriched_entry = {
                "id": item["id"],
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "chunk": enriched
            }
            outfile.write(json.dumps(enriched_entry) + "\n")

print(f"✅ All enriched contextual chunks written to {OUTPUT_FILE}")
