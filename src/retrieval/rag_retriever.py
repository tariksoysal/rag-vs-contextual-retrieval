# src/retrieval/rag_retriever.py

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_PATH = 'data/processed/chunked_documents.jsonl'
INDEX_PATH = 'data/processed/faiss_rag.index'
DOCS_PATH = 'data/processed/rag_docs.jsonl'  # To map FAISS IDs to original chunks

EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

def embed_chunks(chunks, model):
    return model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def main():
    model = SentenceTransformer(EMBED_MODEL_NAME)
    chunks, metadata = [], []

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            chunks.append(item['chunk'])
            metadata.append({
                'id': item['id'],
                'text': item['chunk'],
                'tags': item['tags'],
                'label': item['label']
            })

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks, model)

    print("Building FAISS index...")
    index = build_index(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')

    print(f"Index saved to {INDEX_PATH}")
    print(f"Chunk metadata saved to {DOCS_PATH}")

if __name__ == '__main__':
    main()

