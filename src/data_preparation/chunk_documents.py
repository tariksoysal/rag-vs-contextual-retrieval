# src/data_preparation/chunk_documents.py

import json
import os
from tqdm import tqdm

INPUT_FILE = "data/processed/combined.jsonl"
OUTPUT_FILE = "data/processed/chunked_documents.jsonl"
CHUNK_SIZE = 300  # Approximate number of words per chunk

def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]

os.makedirs("data/processed", exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in tqdm(infile, desc="Chunking documents"):
        item = json.loads(line)
        full_text = f"{item['title']} {item['body']}"
        chunks = chunk_text(full_text)

        for idx, chunk in enumerate(chunks):
            chunked_entry = {
                "id": item["id"],
                "chunk": chunk,
                "chunk_id": idx,
                "source": item["source"]  # keep track of origin (cs, p, ds)
            }
            outfile.write(json.dumps(chunked_entry) + "\n")

print(f"✅ Saved chunked output to {OUTPUT_FILE}")
