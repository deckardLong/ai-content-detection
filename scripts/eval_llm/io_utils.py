import csv
from pathlib import Path

INPUT_FILE = Path('../eval/eval_samples.csv')
OUTPUT_FILE = Path('../eval/eval_results.csv')

def load_samples():
    with open(INPUT_FILE, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def save_results(rows: list[dict]):
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)