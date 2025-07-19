import json
import random
from pathlib import Path

INPUT = (
    Path(__file__).resolve().parent.parent.parent
    / "data/processed/combined.jsonl"
)
TRAIN_OUT = (
    Path(__file__).resolve().parent.parent.parent
    / "data/processed/train_questions.jsonl"
)
EVAL_OUT = (
    Path(__file__).resolve().parent.parent.parent
    / "data/processed/eval_questions_raw.jsonl"
)

random.seed(42)

with open(INPUT, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)
split_idx = int(0.9 * len(data))
train, eval = data[:split_idx], data[split_idx:]

with open(TRAIN_OUT, "w", encoding="utf-8") as f:
    for item in train:
        f.write(json.dumps(item) + "\n")

with open(EVAL_OUT, "w", encoding="utf-8") as f:
    for item in eval:
        f.write(json.dumps(item) + "\n")

print(f"✅ Wrote {len(train)} train and {len(eval)} eval posts.")
