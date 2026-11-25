"""Train BamiBERT (frozen features) + the combined CNN||BiLSTM head and export the model file.

    python comparison/scripts/train_hybrid.py
    python comparison/scripts/train_hybrid.py --epochs 10

Same split (data/subsets, 70/15/15 by group), hyperparameters and truncation as scripts/train.py,
so the result is directly comparable with bamibert+cnn.pt and bamibert+bilstm.pt.
Output: comparison/models/<run>/bamibert+cnn_bilstm.pt  (<run> = frozen_equal_text).
Resume: re-run the same command; an interrupted model continues from its last finished epoch.
"""
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import torch
import yaml

from run_all import device, fit, run_id
from src.data.features import extract_features, feature_cache_path
from src.data.loaders import encode_texts
from src.data.loading import load_dataframe
from src.data.notebook_split import notebook_split

ENC_KEY, HEAD = "bamibert", "cnn_bilstm"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    cfg["heads"] = {**cfg["heads"], **cfg["hybrid_head"]}      # make the hybrid head visible to fit()
    mode = "frozen"
    mcfg = dict(cfg[mode])
    if args.epochs:
        mcfg["epochs"] = args.epochs
    run = run_id(cfg, mode)

    out_dir = Path("models") / run
    ckpt_dir = Path("results") / run / "train_ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{ENC_KEY}+{HEAD}.pt"
    if out_file.exists():
        print(f"{out_file} already exists. Delete it to train again.")
        return

    df = load_dataframe(cfg["data_path"], cfg["text_col"])
    split = notebook_split(df)
    print(f"device={device}  run={run}  train={len(split['train'])}  val={len(split['val'])}  "
          f"test={len(split['test'])}  epochs={mcfg['epochs']}  "
          f"truncate_syllables={cfg.get('truncate_syllables')}")

    data = encode_texts(ENC_KEY, cfg, df)
    max_length = data.max_length
    data, extract_seconds = extract_features(
        cfg["encoders"][ENC_KEY], data, device, feature_cache_path(ENC_KEY, data))
    print(f"[{ENC_KEY}] feature extraction: {extract_seconds:.1f}s (all {len(df)} texts, cached)")

    name = f"{ENC_KEY}+{HEAD}"
    print(f"[{name}]")
    ckpt = ckpt_dir / f"{name}.pt"
    model, _, info = fit(ENC_KEY, HEAD, cfg, mcfg, True, data, split, cfg["seed"], ckpt)

    torch.save({
        "encoder_key": ENC_KEY, "encoder_name": cfg["encoders"][ENC_KEY],
        "head_type": HEAD, "head_cfg": cfg["heads"][HEAD],
        "num_classes": cfg["num_classes"], "dropout": cfg["dropout"],
        "max_length": max_length, "truncate_syllables": cfg.get("truncate_syllables"),
        "mode": mode, "run": run, "scope": "head",
        "state_dict": model.head.state_dict(),
        "val_macro_f1": info["best_val_macro_f1"], "best_epoch": info["best_epoch"],
        "history": info["history"], "train_seconds": info["train_seconds"],
        "extract_seconds": extract_seconds, "train_config": mcfg,
    }, out_file)
    ckpt.unlink(missing_ok=True)
    print(f"  saved {out_file}  (best epoch {info['best_epoch']}, "
          f"val_macro_f1={info['best_val_macro_f1']:.4f}, train {info['train_seconds']:.1f}s)")

if __name__ == "__main__":
    main()
