"""Pick N = syllables kept per article so both encoders read the same text.

For each N: share of articles whose first N syllables fit in PhoBERT's 256 tokens (after word
segmentation), and how many BamiBERT tokens the same text needs. Choose the largest N with
PhoBERT fit >= ~99%, then set bamibert max_length >= its p99.

    python comparison/scripts/choose_truncation.py
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import yaml
from transformers.utils import logging as hf_logging

from src.data.loading import load_dataframe
from src.data.segmentation import segment_all
from src.models.encoders import load_tokenizer

hf_logging.set_verbosity_error()
CANDIDATES = [150, 175, 200, 225, 250, 275, 300]
PHOBERT_MAX = 256

cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
df = load_dataframe(cfg["data_path"], cfg["text_col"])
pho_tok = load_tokenizer(cfg["encoders"]["phobert"])
bami_tok = load_tokenizer(cfg["encoders"]["bamibert"])

# Segment the full text once, then cut in syllable space (a word "a_b" counts as 2 syllables)
seg = [s.split() for s in segment_all(df["text"].tolist())]

def first_syllables(words, n):
    out, count = [], 0
    for w in words:
        k = len(w.split("_"))
        if count + k > n:
            break
        out.append(w)
        count += k
    return " ".join(out)

rows = []
for n in CANDIDATES:
    pho_len = np.array([len(x) for x in pho_tok([first_syllables(w, n) for w in seg])["input_ids"]])
    bami_len = np.array([len(x) for x in bami_tok([" ".join(t.split()[:n]) for t in df["text"]])["input_ids"]])
    rows.append({
        "N syllables": n,
        "PhoBERT fits 256 (%)": 100 * np.mean(pho_len <= PHOBERT_MAX),
        "PhoBERT tokens p50": int(np.median(pho_len)),
        "BamiBERT tokens p50": int(np.median(bami_len)),
        "BamiBERT tokens p99": int(np.percentile(bami_len, 99)),
        "BamiBERT tokens max": int(bami_len.max()),
    })

print(pd.DataFrame(rows).round(2).to_markdown(index=False))
