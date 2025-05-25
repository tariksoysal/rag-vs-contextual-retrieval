import json
import random
from collections import defaultdict

INPUT_PATH = "data/processed/chunked_documents.jsonl"
TRAIN_OUT = "data/processed/chunked_documents_train.jsonl"
EVAL_OUT = "data/processed/eval_questions.jsonl"

SPLIT_RATIO = 0.1  # 10% for evaluation

# Group chunks by question ID
by_id = defaultdict(list)
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        by_id[item["id"]].append(item)

# Sample 10% of question IDs for evaluation
all_ids = list(by_id.keys())
eval_ids = set(random.sample(all_ids, int(len(all_ids) * SPLIT_RATIO)))

with open(TRAIN_OUT, "w", encoding="utf-8") as train_f, \
     open(EVAL_OUT, "w", encoding="utf-8") as eval_f:

    for qid, chunks in by_id.items():
        if qid in eval_ids:
            eval_entry = {
                "question": chunks[0]["chunk"].split('\n')[-1][:300],  # crude title guess
                "relevant_chunk_ids": [c["chunk_id"] for c in chunks],
                "id": qid
            }
            eval_f.write(json.dumps(eval_entry) + "\n")
        else:
            for c in chunks:
                train_f.write(json.dumps(c) + "\n")

print(f"✅ Wrote {len(eval_ids)} questions to eval set.")
