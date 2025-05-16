# src/data_preparation/chunk_documents.py

import json
import os

INPUT_PATH = 'data/processed/stackexchange_cs.jsonl'
OUTPUT_PATH = 'data/processed/chunked_documents.jsonl'
CHUNK_SIZE = 150  # words
OVERLAP = 30      # optional overlap for better context

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_documents(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            item = json.loads(line)
            full_text = item['title'] + '\n' + item['body']
            chunks = chunk_text(full_text, CHUNK_SIZE, OVERLAP)
            for i, chunk in enumerate(chunks):
                chunk_entry = {
                    'id': f"{item['id']}_{i}",
                    'chunk': chunk,
                    'tags': item['tags'],
                    'label': item['label']
                }
                outfile.write(json.dumps(chunk_entry) + '\n')

if __name__ == '__main__':
    process_documents(INPUT_PATH, OUTPUT_PATH)

