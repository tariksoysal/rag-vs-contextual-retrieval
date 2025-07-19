# src/retrieval/query_rag.py

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.generation.generate_answers import generate_answer

import faiss
import json
from sentence_transformers import SentenceTransformer

INDEX_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data/processed/faiss_rag.index"
)
DOCS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data/processed/rag_docs.jsonl"
)
EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = 10
OLLAMA_MODEL = "gemma3:latest"  # Or change to llama3.2:latest if needed

def load_documents(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def search_index(query, model, index, docs, k=TOP_K):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    results = [docs[i] for i in indices[0]]
    return results, distances[0]

def main():
    print("Loading index and model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    docs = load_documents(DOCS_PATH)

    while True:
        query = input("\n🔎 Enter your query (or type 'exit'): ").strip()
        if query.lower() == 'exit':
            break

        results, distances = search_index(query, model, index, docs)
        print(f"\nTop {TOP_K} retrieved chunks:")
        for i, (res, dist) in enumerate(zip(results, distances)):
            print(f"\n#{i+1} | Distance: {dist:.2f}")
            print(res['text'][:300] + "...\n" + "-"*50)

        print("\n💡 Generating answer with Ollama...")
        answer = generate_answer(query, results, model=OLLAMA_MODEL)
        print("\n🧠 Answer:\n")
        print(answer)

if __name__ == '__main__':
    main()

