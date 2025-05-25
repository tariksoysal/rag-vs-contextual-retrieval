from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

INDEX_PATH = 'data/processed/faiss_rag.index'
DOCS_PATH = 'data/processed/rag_docs.jsonl'
EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Load index + model
model = SentenceTransformer(EMBED_MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

# Load metadata
with open(DOCS_PATH, 'r', encoding='utf-8') as f:
    docs = [json.loads(line) for line in f]

def retrieve(query, top_k=5):
    vec = model.encode([query], convert_to_numpy=True)
    scores, indices = index.search(vec, top_k)
    return [docs[i] for i in indices[0]]

# Example query
query = "What are the differences between PCA and L1 regularization?"
results = retrieve(query)

for i, r in enumerate(results, 1):
    print(f"\n--- Result {i} ---\n{r['text']}\n")
