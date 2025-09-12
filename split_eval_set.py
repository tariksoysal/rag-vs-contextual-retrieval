"""Split processed chunks into train and evaluation sets.

The script groups all chunked documents by their original question ID and then
randomly selects a portion of those IDs for evaluation.  Chunks belonging to the
selected IDs are written to ``eval_questions.jsonl`` as evaluation entries,
while the remainder are saved for training.
"""

import json
import random
from collections import defaultdict

INPUT_PATH = "data/processed/chunked_documents.jsonl"
TRAIN_OUT = "data/processed/chunked_documents_train.jsonl"
EVAL_OUT = "data/processed/eval_questions.jsonl"

SPLIT_RATIO = 0.1  # 10% for evaluation

# Group chunks by question ID so we can split on full questions rather than
# individual chunks.
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
            # Build a minimal evaluation entry consisting of the question and
            # IDs of relevant chunks.
            eval_entry = {
                "question": chunks[0]["chunk"].split('\n')[-1][:300],
                "relevant_chunk_ids": [c["chunk_id"] for c in chunks],
                "id": qid
            }
            eval_f.write(json.dumps(eval_entry) + "\n")
        else:
            for c in chunks:
                train_f.write(json.dumps(c) + "\n")

print(f"✅ Wrote {len(eval_ids)} questions to eval set.")
