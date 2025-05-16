# src/data_preparation/preprocess.py

import pandas as pd
import json
import os

INPUT_PATH = 'data/raw/cs.tsv'
OUTPUT_PATH = 'data/processed/stackexchange_cs.jsonl'

def preprocess_and_save(input_path, output_path):
    df = pd.read_csv(input_path, sep='\t')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            entry = {
                'id': row['id'],
                'title': row['title'],
                'body': row['body'],
                'tags': row['tags'],
                'label': int(row['label'])  # Convert to int for safety
            }
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    preprocess_and_save(INPUT_PATH, OUTPUT_PATH)

