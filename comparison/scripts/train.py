"""Train on the notebook split (data/subsets, 70/15/15 by group) and export the model file.

    python comparison/scripts/train.py --encoder bamibert --head cnn     # one model
    python comparison/scripts/train.py --all                             # all 4 models
    python comparison/scripts/train.py --all --mode finetune

Output: comparison/models/<run>/<encoder>+<head>.pt, with <run> = <mode>_<run_name> (e.g. frozen_equal_text).
The file also holds the per-epoch train/val/test loss history and the training time
(plot with scripts/plot_losses.py).
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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["phobert", "bamibert"])
    ap.add_argument("--head", choices=["cnn", "bilstm"])
    ap.add_argument("--all", action="store_true", help="train all encoder x head combinations")
    ap.add_argument("--mode", choices=["frozen", "finetune"])
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    if not args.all and not (args.encoder and args.head):
        ap.error("give --encoder and --head, or --all")
    return args

def main():
    args = parse_args()
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    mode = args.mode or cfg["mode"]
    mcfg = dict(cfg[mode])
    if args.epochs:
        mcfg["epochs"] = args.epochs
    frozen = mode == "frozen"
    run = run_id(cfg, mode)

    encoders = list(cfg["encoders"]) if args.all else [args.encoder]
    heads = list(cfg["heads"]) if args.all else [args.head]

    df = load_dataframe(cfg["data_path"], cfg["text_col"])
    split = notebook_split(df)
    print(f"device={device}  run={run}  train={len(split['train'])}  val={len(split['val'])}  "
          f"test={len(split['test'])}  epochs={mcfg['epochs']}  "
          f"truncate_syllables={cfg.get('truncate_syllables')}  max_length={cfg['max_length']}")

    out_dir = Path("models") / run
    ckpt_dir = Path("results") / run / "train_ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for enc_key in encoders:
        todo = [h for h in heads if not (out_dir / f"{enc_key}+{h}.pt").exists()]
        if not todo:
            continue
        data = encode_texts(enc_key, cfg, df)
        max_length, extract_seconds = data.max_length, 0.0
        if frozen:
            data, extract_seconds = extract_features(
                cfg["encoders"][enc_key], data, device, feature_cache_path(enc_key, data))
            print(f"[{enc_key}] feature extraction: {extract_seconds:.1f}s (all {len(df)} texts, cached)")

        for head_type in todo:
            name = f"{enc_key}+{head_type}"
            print(f"[{name}]")
            ckpt = ckpt_dir / f"{name}.pt"
            model, _, info = fit(enc_key, head_type, cfg, mcfg, frozen, data, split, cfg["seed"], ckpt)

            # frozen: the encoder is the unchanged pretrained one -> only the head needs saving
            torch.save({
                "encoder_key": enc_key, "encoder_name": cfg["encoders"][enc_key],
                "head_type": head_type, "head_cfg": cfg["heads"][head_type],
                "num_classes": cfg["num_classes"], "dropout": cfg["dropout"],
                "max_length": max_length, "truncate_syllables": cfg.get("truncate_syllables"),
                "mode": mode, "run": run,
                "scope": "head" if frozen else "full",
                "state_dict": (model.head if frozen else model).state_dict(),
                "val_macro_f1": info["best_val_macro_f1"], "best_epoch": info["best_epoch"],
                "history": info["history"], "train_seconds": info["train_seconds"],
                "extract_seconds": extract_seconds, "train_config": mcfg,
            }, out_dir / f"{name}.pt")
            ckpt.unlink()
            print(f"  saved {out_dir / f'{name}.pt'}  (best epoch {info['best_epoch']}, "
                  f"val_macro_f1={info['best_val_macro_f1']:.4f}, train {info['train_seconds']:.1f}s)")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        del data

if __name__ == "__main__":
    main()
