import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INDEX_PATH = "data/processed/faiss_rag.index"
DOCS_PATH = "data/processed/rag_docs.jsonl"
EVAL_PATH = "data/processed/eval_questions.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# Load FAISS index and metadata
index = faiss.read_index(INDEX_PATH)
with open(DOCS_PATH, "r", encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]
doc_lookup = {(doc["id"], doc.get("chunk_id", 0)): i for i, doc in enumerate(docs)}

# Load eval questions
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_data = [json.loads(line) for line in f]

model = SentenceTransformer(MODEL_NAME)

recall_at_1 = 0
recall_at_k = 0
mrr = 0

for example in tqdm(eval_data, desc=f"Evaluating Recall@1 + Recall@{TOP_K} + MRR@{TOP_K}"):
    vec = model.encode([example["question"]], convert_to_numpy=True)
    scores, indices = index.search(vec, TOP_K)

    gold_ids = set((example["id"], cid) for cid in example["relevant_chunk_ids"])
    retrieved = [(docs[i]["id"], docs[i].get("chunk_id", 0)) for i in indices[0]]

    # Recall@1
    if retrieved[0] in gold_ids:
        recall_at_1 += 1

    # Recall@K
    if gold_ids & set(retrieved):
        recall_at_k += 1

    # MRR@K
    for rank, result in enumerate(retrieved, 1):
        if result in gold_ids:
            mrr += 1 / rank
            break

total = len(eval_data)
print(f"\n🎯 Recall@1:   {recall_at_1 / total:.4f} ({recall_at_1}/{total})")
print(f"🎯 Recall@{TOP_K}: {recall_at_k / total:.4f} ({recall_at_k}/{total})")
print(f"📈 MRR@{TOP_K}:    {mrr / total:.4f}")


print("\n🔍 DEBUG: Show example retrievals")

for i in range(3):
    ex = eval_data[i]
    question = ex["question"]
    gold_ids = set((ex["id"], cid) for cid in ex["relevant_chunk_ids"])

    print(f"\n🧠 Q{i+1}: {question[:80]}...")
    print(f"Expected: {gold_ids}")

    vec = model.encode([question], convert_to_numpy=True)
    scores, indices = index.search(vec, TOP_K)

    print("Retrieved:")
    for rank, idx in enumerate(indices[0], 1):
        doc = docs[idx]
        print(f"{rank}. ID: {doc['id']} | Chunk: {doc.get('chunk_id', 0)}")
