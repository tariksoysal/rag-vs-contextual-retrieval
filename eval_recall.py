import json
import os
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

LOG_PATH = "evaluation_logs.jsonl"
TOP_K = 5
SIM_THRESHOLD = 0.7  # for semantic similarity match
KEYWORD_THRESHOLD = 0.5  # percent of keywords matched

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def jaccard(a, b):
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    return len(a_set & b_set) / len(a_set | b_set)

def evaluate_entry(entry):
    if "gold_answer" not in entry or not entry["gold_answer"].strip():
        return None

    gold = entry["gold_answer"]
    chunks = entry["chunks"][:TOP_K]

    # --- Keyword Match ---
    keyword_hits = 0
    for chunk in chunks:
        score = jaccard(gold, chunk)
        if score >= KEYWORD_THRESHOLD:
            keyword_hits += 1

    # --- Embedding Match ---
    gold_emb = model.encode(gold, convert_to_tensor=True)
    chunk_embs = model.encode(chunks, convert_to_tensor=True)
    cos_scores = util.cos_sim(gold_emb, chunk_embs)[0]

    semantic_hits = sum(1 for score in cos_scores if score >= SIM_THRESHOLD)

    return {
        "query": entry["query"],
        "mode": entry["mode"],
        "recall@k_keyword": round(keyword_hits / TOP_K, 2),
        "recall@k_semantic": round(semantic_hits / TOP_K, 2),
        "miss@k_keyword": round(1 - keyword_hits / TOP_K, 2),
        "miss@k_semantic": round(1 - semantic_hits / TOP_K, 2)
    }

def main():
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    results = [evaluate_entry(e) for e in entries if e.get("gold_answer")]
    results = [r for r in results if r is not None]

    grouped = defaultdict(list)
    for r in results:
        grouped[r["query"]].append(r)

    for query, runs in grouped.items():
        print(f"\n🟡 Query: {query}")
        for run in runs:
            print(f"  Mode: {run['mode']}")
            print(f"    Keyword Recall@{TOP_K}: {run['recall@k_keyword']} (miss: {run['miss@k_keyword']})")
            print(f"    Semantic Recall@{TOP_K}: {run['recall@k_semantic']} (miss: {run['miss@k_semantic']})")

if __name__ == "__main__":
    main()
