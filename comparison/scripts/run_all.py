"""Compare {PhoBERT, BamiBERT} x {CNN, BiLSTM} with group k-fold cross-validation.

    python comparison/scripts/run_all.py                    # mode from config (default: frozen)
    python comparison/scripts/run_all.py --mode finetune
    python comparison/scripts/run_all.py --limit 200 --epochs 1 --folds 2   # smoke test

Resume: just re-run the same command. Finished folds are skipped, and an interrupted fold
continues from its last finished epoch (results/<mode>/ckpt/). Delete results/<mode>/ to start over.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from transformers import get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

from src.data.features import extract_features
from src.data.loaders import encode_texts, make_loaders
from src.data.loading import load_dataframe, subsample
from src.data.splits import make_folds
from src.evaluation.metrics import compute_metrics
from src.models.classifier import EncoderClassifier
from src.training.seed import set_seed
from src.training.trainer import predict, train_epoch

hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frozen", "finetune"])
    ap.add_argument("--limit", type=int, help="use a stratified subset of N samples (smoke test)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--folds", type=int)
    return ap.parse_args()

def save_checkpoint(path, **state):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)    # atomic: an interrupted save never corrupts the previous checkpoint

def fit(enc_key, head_type, cfg, mcfg, frozen, data, split, seed, ckpt_path):
    """Train on split['train'], early-stop on split['val']; return (best model, test loader, best val macro-F1)."""
    set_seed(seed)
    model = EncoderClassifier(
        None if frozen else cfg["encoders"][enc_key], head_type, cfg["heads"][head_type],
        cfg["num_classes"], cfg["dropout"]
    ).to(device)
    train_l, val_l, test_l = make_loaders(data, split, mcfg["batch_size"], seed)

    param_groups = [{"params": model.head.parameters(), "lr": mcfg["lr_head"]}]
    if not frozen:
        param_groups.append({"params": model.encoder.parameters(), "lr": mcfg["lr_encoder"]})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = None
    if not frozen:
        total = mcfg["epochs"] * len(train_l)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(mcfg["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    criterion = nn.CrossEntropyLoss()

    # Early stopping on val macro-F1
    start, best_val, best_state, wait, stopped = 0, -1.0, None, 0, False

    # Resume from the last finished epoch (same weights, optimizer, scheduler and RNG state)
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["mcfg"] == mcfg:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            if scheduler is not None:
                scheduler.load_state_dict(ck["scheduler"])
            train_l.generator.set_state(ck["loader_rng"])
            torch.set_rng_state(ck["cpu_rng"])
            if device == "cuda":
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
            start, best_val, best_state, wait, stopped = (
                ck["epoch"], ck["best_val"], ck["best_state"], ck["wait"], ck["stopped"])
            print(f"    resumed after epoch {start} (best val_macro_f1={best_val:.4f})")
        else:
            print("    config changed since checkpoint -> training from scratch")

    for epoch in range(start, mcfg["epochs"]):
        if stopped:
            break
        loss = train_epoch(model, train_l, optimizer, criterion, device, scaler, scheduler)
        y, prob = predict(model, val_l, device)
        val_f1 = f1_score(y, (prob >= 0.5).astype(int), average="macro")
        print(f"    epoch {epoch + 1:2d}  loss={loss:.4f}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val:
            best_val, wait = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            stopped = wait >= mcfg["patience"]

        save_checkpoint(
            ckpt_path, mcfg=mcfg, epoch=epoch + 1, best_val=best_val, best_state=best_state,
            wait=wait, stopped=stopped, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), scheduler=scheduler.state_dict() if scheduler is not None else None,
            loader_rng=train_l.generator.get_state(), cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else None,
        )

    model.load_state_dict(best_state)
    return model, test_l, best_val

def run_one(enc_key, head_type, cfg, mcfg, frozen, data, split, seed, ckpt_path):
    model, test_l, _ = fit(enc_key, head_type, cfg, mcfg, frozen, data, split, seed, ckpt_path)
    y, prob = predict(model, test_l, device)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return y, prob

def summarize(fold_metrics, oof):
    keys = fold_metrics[0].keys()
    return {
        "mean": {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys},
        "std": {k: float(np.std([m[k] for m in fold_metrics], ddof=1)) if len(fold_metrics) > 1 else 0.0
                for k in keys},
        "oof": oof,          # metrics on all out-of-fold predictions pooled
        "folds": fold_metrics,
    }

def main():
    args = parse_args()
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    mode = args.mode or cfg["mode"]
    mcfg = dict(cfg[mode])
    if args.epochs:
        mcfg["epochs"] = args.epochs
    frozen = mode == "frozen"
    n_folds = args.folds or cfg["n_folds"]

    df = load_dataframe(cfg["data_path"], cfg["text_col"])
    if args.limit:
        df = subsample(df, args.limit, cfg["seed"])
    out = Path("results") / (f"{mode}_smoke" if args.limit else mode)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    folds = make_folds(df, n_folds, cfg["seed"], cache_path=out / "folds.json")
    types = df["generated_type"].values.astype(str)
    print(f"device={device}  mode={mode}  n={len(df)}  folds={n_folds}  "
          f"{df['generated_type'].value_counts().to_dict()}")

    results = {}
    for enc_key in cfg["encoders"]:
        pred_file = lambda name, k: out / "preds" / f"{name}_fold{k}.npz"
        names = [f"{enc_key}+{h}" for h in cfg["heads"]]
        data = None
        if not all(pred_file(n, k).exists() for n in names for k in range(n_folds)):
            data = encode_texts(enc_key, cfg, df)
            if frozen:
                feat_path = Path("results/cache") / f"features_{enc_key}_L{data.max_length}_n{len(df)}.npy"
                data = extract_features(cfg["encoders"][enc_key], data, device, feat_path)

        for head_type in cfg["heads"]:
            name = f"{enc_key}+{head_type}"
            fold_metrics, oof = [], []
            for k, split in enumerate(folds):
                f = pred_file(name, k)
                if not f.exists():
                    print(f"[{name}] fold {k + 1}/{n_folds}")
                    ckpt = out / "ckpt" / f"{name}_fold{k}.pt"
                    y, prob = run_one(enc_key, head_type, cfg, mcfg, frozen, data, split,
                                      cfg["seed"] + k, ckpt)
                    np.savez(f, idx=split["test"], y=y, prob=prob,
                             pred=(prob >= 0.5).astype(int), type=types[split["test"]])
                    ckpt.unlink()    # fold finished -> its predictions are the resume point
                d = np.load(f, allow_pickle=True)
                fold_metrics.append(compute_metrics(d["y"], d["pred"], d["prob"], d["type"]))
                oof.append(d)

            oof_metrics = compute_metrics(*(np.concatenate([d[c] for d in oof]) for c in ("y", "pred", "prob", "type")))
            results[name] = summarize(fold_metrics, oof_metrics)
            m, s = results[name]["mean"], results[name]["std"]
            print(f"{name}: acc={m['accuracy']:.4f}±{s['accuracy']:.4f}  "
                  f"macro_f1={m['macro_f1']:.4f}±{s['macro_f1']:.4f}  auc={m['roc_auc']:.4f}")

        del data

    with open(out / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump({"mode": mode, "n": len(df), "n_folds": n_folds, "config": mcfg,
                   "results": results}, fp, indent=2)
    print(f"Saved {out / 'metrics.json'}")

if __name__ == "__main__":
    main()
