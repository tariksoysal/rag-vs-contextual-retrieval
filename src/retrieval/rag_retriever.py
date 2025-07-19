# src/retrieval/rag_retriever.py

import json
import os
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

BASE_PATH = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_PATH / "data/processed/chunked_documents_train.jsonl"
INDEX_PATH = BASE_PATH / "data/processed/faiss_rag.index"
DOCS_PATH = BASE_PATH / "data/processed/rag_docs.jsonl"

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
                'id': item.get('id'),
                'text': item['chunk'],
                'source': item.get('source', 'unknown'),
                'tags': item.get('tags', None),
                'label': item.get('label', None)
            })

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks, model)

    print("Building FAISS index...")
    index = build_index(embeddings)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with open(DOCS_PATH, 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')

    print(f"✅ FAISS index saved to {INDEX_PATH}")
    print(f"📄 Chunk metadata saved to {DOCS_PATH}")

if __name__ == '__main__':
    main()
