import json
from collections import defaultdict
from pathlib import Path

EVAL_CHUNKED = (
    Path(__file__).resolve().parent.parent.parent
    / "data/processed/chunked_documents_eval.jsonl"
)
EVAL_OUTPUT = (
    Path(__file__).resolve().parent.parent.parent
    / "data/processed/eval_questions.jsonl"
)

# Group chunks by question ID
by_question = defaultdict(list)

with open(EVAL_CHUNKED, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        by_question[item["id"]].append(item)

# Save one eval question per post
with open(EVAL_OUTPUT, "w", encoding="utf-8") as out:
    for qid, chunks in by_question.items():
        question = chunks[0]["chunk"].split('\n')[-1][:300]
        entry = {
            "question": question,
            "id": qid,
            "relevant_chunk_ids": [c["chunk_id"] for c in chunks]
        }
        out.write(json.dumps(entry) + "\n")

print(f"✅ Created {len(by_question)} entries in eval_questions.jsonl")
