"""Detect AI-written Vietnamese text with BamiBERT (frozen encoder) + the combined CNN||BiLSTM head.

    python comparison/scripts/predict.py --text "Nội dung bài viết ..."
    python comparison/scripts/predict.py --file bai_viet.txt
    python comparison/scripts/predict.py --csv data.csv --column text --out ket_qua.csv

The encoder runs once per text; every requested head reads the same features. Preprocessing matches
training: HTML/URL cleaning -> first 225 syllables -> BamiBERT tokenizer (max 512 tokens).
Models are read from models/<run>/bamibert+<head>.pt (run = frozen_equal_text by default).
Default head: cnn_bilstm (train it with scripts/train_hybrid.py). To compare several heads on the
same texts: --heads cnn bilstm cnn_bilstm.
Output: P(AI) of each head (>= 0.5 means AI).
"""
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
import torch
import yaml
from transformers.utils import logging as hf_logging

from src.data.loading import clean_text, truncate_syllables
from src.models.classifier import EncoderClassifier
from src.models.encoders import load_encoder, load_tokenizer

hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="one text given directly")
    src.add_argument("--file", help="a .txt file holding one article")
    src.add_argument("--csv", help="a CSV with many articles")
    ap.add_argument("--column", default="text", help="text column of --csv (default: text)")
    ap.add_argument("--out", help="save results of --csv to this file")
    ap.add_argument("--heads", nargs="+", default=["cnn_bilstm"],
                    choices=["cnn", "bilstm", "cnn_bilstm"], help="heads to run (default: cnn_bilstm)")
    ap.add_argument("--run", help="models/<run>/ to read (default: from config, frozen_equal_text)")
    ap.add_argument("--batch-size", type=int, default=16)
    return ap.parse_args()

def load_heads(run, names):
    heads, metas = {}, {}
    for h in names:
        path = Path("models") / run / f"bamibert+{h}.pt"
        if not path.exists():
            cmd = ("python comparison/scripts/train_hybrid.py" if h == "cnn_bilstm"
                   else f"python comparison/scripts/train.py --encoder bamibert --head {h}")
            sys.exit(f"Missing {path}. Train it first:  {cmd}")
        meta = torch.load(path, map_location="cpu", weights_only=False)
        if meta["scope"] != "head":
            sys.exit(f"{path} is a fine-tuned model; this script only handles frozen-encoder heads.")
        model = EncoderClassifier(None, meta["head_type"], meta["head_cfg"],
                                  meta["num_classes"], meta["dropout"])
        model.head.load_state_dict(meta["state_dict"])
        heads[h], metas[h] = model.head.to(device).eval(), meta
    return heads, metas

@torch.no_grad()
def predict_texts(texts, encoder, tokenizer, heads, max_length, n_syllables, batch_size):
    prepared = [truncate_syllables(clean_text(t), n_syllables) for t in texts]
    probs = {h: [] for h in heads}
    for i in range(0, len(prepared), batch_size):
        enc = tokenizer(prepared[i:i + batch_size], truncation=True, padding="max_length",
                        max_length=max_length, return_tensors="pt").to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            hidden = encoder(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"]).last_hidden_state
            for h, head in heads.items():
                logits = head(hidden, enc["attention_mask"])
                probs[h].extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
    return probs

def main():
    args = parse_args()
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    run = args.run or f"{cfg['mode']}_{cfg['run_name']}"
    heads, metas = load_heads(run, args.heads)
    meta = metas[args.heads[0]]
    max_length, n_syllables = meta["max_length"], meta.get("truncate_syllables")

    if args.text:
        texts = [args.text]
    elif args.file:
        texts = [Path(args.file).read_text(encoding="utf-8")]
    else:
        texts = pd.read_csv(args.csv, encoding="utf-8-sig")[args.column].astype(str).tolist()

    short = [i for i, t in enumerate(texts) if n_syllables and len(t.split()) < n_syllables]
    if short:
        print(f"Note: {len(short)} text(s) have fewer than {n_syllables} syllables; "
              f"the model was trained on {n_syllables}-syllable inputs, so results are less reliable.")

    print(f"device={device}  run={run}  encoder={meta['encoder_name']}  "
          f"max_length={max_length}  truncate_syllables={n_syllables}")
    tokenizer = load_tokenizer(meta["encoder_name"])
    encoder = load_encoder(meta["encoder_name"]).to(device).eval()
    probs = predict_texts(texts, encoder, tokenizer, heads, max_length, n_syllables, args.batch_size)

    result = pd.DataFrame({"text": [t[:60].replace("\n", " ") for t in texts]})
    for h in args.heads:
        result[f"p_ai_{h}"] = probs[h]
        result[f"label_{h}"] = ["AI" if p >= 0.5 else "Human" for p in probs[h]]
    print(result.drop(columns="text").round(4).to_string() if len(texts) > 20
          else result.round(4).to_string())
    if args.out:
        full = pd.DataFrame({"text": texts, **{c: result[c] for c in result.columns if c != "text"}})
        full.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"saved {args.out}")

if __name__ == "__main__":
    main()
