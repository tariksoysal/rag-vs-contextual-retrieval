# src/data_preparation/preprocess.py

import pandas as pd
import json
import os

DATASETS = {
    "cs": "data/raw/cs.tsv",
    "politics": "data/raw/p.tsv",
    "datascience": "data/raw/ds.tsv"
}

os.makedirs("data/processed", exist_ok=True)

all_entries = []

for domain, path in DATASETS.items():
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["title", "body"])  # Drop incomplete entries
    for _, row in df.iterrows():
        entry = {
            "id": int(row["id"]),
            "title": str(row["title"]),
            "body": str(row["body"]),
            "tags": str(row.get("tags", "")),
            "label": int(row.get("label", -1)),
            "source": domain
        }
        all_entries.append(entry)

with open("data/processed/combined.jsonl", "w", encoding="utf-8") as f:
    for item in all_entries:
        f.write(json.dumps(item) + "\n")

print(f"✅ Combined {len(all_entries)} entries into data/processed/combined.jsonl")
