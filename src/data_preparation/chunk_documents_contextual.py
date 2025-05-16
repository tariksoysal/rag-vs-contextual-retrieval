import json
import os

import re

def clean_html(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'&#x[A-Fa-f0-9]+;', ' ', text)
    text = re.sub(r'&nbsp;|&quot;|&amp;', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


INPUT_PATH = 'data/processed/stackexchange_cs.jsonl'
OUTPUT_PATH = 'data/processed/chunked_contextual.jsonl'
CHUNK_SIZE = 150
OVERLAP = 30

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
            full_text = clean_html(item['body'])
            chunks = chunk_text(full_text, CHUNK_SIZE, OVERLAP)
            for i, chunk in enumerate(chunks):
                enriched_chunk = f"Title: {item['title']}\n\n{chunk}"
                chunk_entry = {
                    'id': f"{item['id']}_{i}",
                    'chunk': enriched_chunk,
                    'tags': item['tags'],
                    'label': item['label']
                }
                outfile.write(json.dumps(chunk_entry) + '\n')

if __name__ == '__main__':
    process_documents(INPUT_PATH, OUTPUT_PATH)

