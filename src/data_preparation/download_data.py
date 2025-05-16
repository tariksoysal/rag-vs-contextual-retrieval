# src/data_preparation/download_data.py

from datasets import load_dataset
import json
import os

OUTPUT_PATH = "data/raw/stackoverflow.jsonl"

def save_dataset(dataset, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            if example["answers"]:
                best_answer = max(example["answers"], key=lambda a: a["score"])
                entry = {
                    "title": example["question"]["title"],
                    "body": example["question"]["body"],
                    "tags": example["question"]["tags"],
                    "answer": best_answer["text"],
                    "answer_score": best_answer["score"]
                }
                f.write(json.dumps(entry) + "\n")

def main():
    print("Downloading AskUbuntu dataset...")
    dataset = load_dataset("stackoverflow-qa")["train"]
    print(f"Loaded {len(dataset)} entries.")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    save_dataset(dataset, OUTPUT_PATH)
    print(f"Saved cleaned dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

