"""Compare {PhoBERT, BamiBERT} x {CNN, BiLSTM} with group k-fold cross-validation.

    python comparison/scripts/run_all.py                    # mode from config (default: frozen)
    python comparison/scripts/run_all.py --mode finetune
    python comparison/scripts/run_all.py --limit 200 --epochs 2 --folds 2   # smoke test

Output: results/<run>/ with <run> = <mode>_<run_name> (run_name from config, e.g. frozen_equal_text)
Resume: just re-run the same command. Finished folds are skipped, and an interrupted fold
continues from its last finished epoch (results/<run>/ckpt/). Delete results/<run>/ to start over.
"""
import argparse, json, os, sys, time
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

from src.data.features import extract_features, feature_cache_path
from src.data.loaders import encode_texts, make_loaders
from src.data.loading import load_dataframe, subsample
from src.data.splits import make_folds
from src.evaluation.metrics import compute_metrics
from src.models.classifier import EncoderClassifier
from src.training.seed import set_seed
from src.training.trainer import predict, train_epoch

hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"
HISTORY_KEYS = ("train_loss", "val_loss", "test_loss", "val_f1")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frozen", "finetune"])
    ap.add_argument("--limit", type=int, help="use a stratified subset of N samples (smoke test)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--folds", type=int)
    return ap.parse_args()

def run_id(cfg, mode, smoke=False):
    name = f"{mode}_{cfg['run_name']}" if cfg.get("run_name") else mode
    return name + ("_smoke" if smoke else "")

def log_loss(y, prob, eps=1e-7):
    """Cross-entropy from P(AI); equals nn.CrossEntropyLoss for 2-class softmax outputs."""
    p = np.clip(prob, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def save_checkpoint(path, **state):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)    # atomic: an interrupted save never corrupts the previous checkpoint

def fit(enc_key, head_type, cfg, mcfg, frozen, data, split, seed, ckpt_path):
    """Train on split['train'], select the epoch with the best val macro-F1.

    Test loss is recorded every epoch for the loss curves only - it never influences training
    or model selection. Returns (best model, test loader, info dict).
    """
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
    patience = mcfg.get("patience")    # None -> train all epochs

    start, best_val, best_epoch, best_state, wait, stopped = 0, -1.0, 0, None, 0, False
    history = {k: [] for k in HISTORY_KEYS}
    train_seconds = 0.0

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
            start, best_val, best_epoch, best_state, wait, stopped = (
                ck["epoch"], ck["best_val"], ck["best_epoch"], ck["best_state"], ck["wait"], ck["stopped"])
            history, train_seconds = ck["history"], ck["train_seconds"]
            print(f"    resumed after epoch {start} (best val_macro_f1={best_val:.4f})")
        else:
            print("    config changed since checkpoint -> training from scratch")

    for epoch in range(start, mcfg["epochs"]):
        if stopped:
            break
        t0 = time.perf_counter()
        loss = train_epoch(model, train_l, optimizer, criterion, device, scaler, scheduler)
        y_val, p_val = predict(model, val_l, device)
        train_seconds += time.perf_counter() - t0          # train + validation (model selection)
        y_test, p_test = predict(model, test_l, device)     # curves only

        val_f1 = f1_score(y_val, (p_val >= 0.5).astype(int), average="macro")
        for k, v in zip(HISTORY_KEYS, (loss, log_loss(y_val, p_val), log_loss(y_test, p_test), val_f1)):
            history[k].append(float(v))
        print(f"    epoch {epoch + 1:2d}  train_loss={loss:.4f}  val_loss={history['val_loss'][-1]:.4f}  "
              f"test_loss={history['test_loss'][-1]:.4f}  val_macro_f1={val_f1:.4f}  "
              f"({time.perf_counter() - t0:.1f}s)")

        if val_f1 > best_val:
            best_val, best_epoch, wait = val_f1, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            stopped = patience is not None and wait >= patience

        save_checkpoint(
            ckpt_path, mcfg=mcfg, epoch=epoch + 1, best_val=best_val, best_epoch=best_epoch,
            best_state=best_state, wait=wait, stopped=stopped, history=history,
            train_seconds=train_seconds, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), scheduler=scheduler.state_dict() if scheduler is not None else None,
            loader_rng=train_l.generator.get_state(), cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else None,
        )

    model.load_state_dict(best_state)
    info = {"best_val_macro_f1": best_val, "best_epoch": best_epoch,
            "epochs_run": len(history["train_loss"]), "train_seconds": train_seconds, "history": history}
    return model, test_l, info

def run_one(enc_key, head_type, cfg, mcfg, frozen, data, split, seed, ckpt_path):
    model, test_l, info = fit(enc_key, head_type, cfg, mcfg, frozen, data, split, seed, ckpt_path)
    t0 = time.perf_counter()
    y, prob = predict(model, test_l, device)
    info["test_seconds"] = time.perf_counter() - t0
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return y, prob, info

def summarize(fold_metrics, oof, fold_times):
    keys = fold_metrics[0].keys()
    return {
        "mean": {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys},
        "std": {k: float(np.std([m[k] for m in fold_metrics], ddof=1)) if len(fold_metrics) > 1 else 0.0
                for k in keys},
        "oof": oof,          # metrics on all out-of-fold predictions pooled
        "folds": fold_metrics,
        "time": {
            "train_seconds_per_fold": float(np.mean([t["train_seconds"] for t in fold_times])),
            "train_seconds_total": float(np.sum([t["train_seconds"] for t in fold_times])),
            "test_seconds_per_fold": float(np.mean([t["test_seconds"] for t in fold_times])),
            "epochs_run": float(np.mean([t["epochs_run"] for t in fold_times])),
            "best_epoch": float(np.mean([t["best_epoch"] for t in fold_times])),
        },
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
    out = Path("results") / run_id(cfg, mode, smoke=bool(args.limit))
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    folds = make_folds(df, n_folds, cfg["seed"], cache_path=out / "folds.json")
    types = df["generated_type"].to_numpy(dtype=str)    # plain numpy strings: npz loads without pickle
    print(f"device={device}  run={out.name}  n={len(df)}  folds={n_folds}  epochs={mcfg['epochs']}  "
          f"truncate_syllables={cfg.get('truncate_syllables')}  max_length={cfg['max_length']}")

    results, extract_seconds = {}, {}
    if (out / "metrics.json").exists():    # keep extraction times when a resumed run skips encoding
        extract_seconds = json.load(open(out / "metrics.json", encoding="utf-8")).get("extract_seconds", {})
    pred_file = lambda name, k: out / "preds" / f"{name}_fold{k}.npz"
    for enc_key in cfg["encoders"]:
        names = [f"{enc_key}+{h}" for h in cfg["heads"]]
        data = None
        if not all(pred_file(n, k).exists() for n in names for k in range(n_folds)):
            data = encode_texts(enc_key, cfg, df)
            max_length = data.max_length
            if frozen:
                data, extract_seconds[enc_key] = extract_features(
                    cfg["encoders"][enc_key], data, device, feature_cache_path(enc_key, data))
                print(f"[{enc_key}] feature extraction: {extract_seconds[enc_key]:.1f}s "
                      f"({len(df)} texts, max_length={max_length})")

        for head_type in cfg["heads"]:
            name = f"{enc_key}+{head_type}"
            fold_metrics, oof, fold_times = [], [], []
            for k, split in enumerate(folds):
                f = pred_file(name, k)
                if not f.exists():
                    print(f"[{name}] fold {k + 1}/{n_folds}")
                    ckpt = out / "ckpt" / f"{name}_fold{k}.pt"
                    y, prob, info = run_one(enc_key, head_type, cfg, mcfg, frozen, data, split,
                                            cfg["seed"] + k, ckpt)
                    np.savez(f, idx=split["test"], y=y, prob=prob,
                             pred=(prob >= 0.5).astype(int), type=types[split["test"]],
                             train_seconds=info["train_seconds"], test_seconds=info["test_seconds"],
                             epochs_run=info["epochs_run"], best_epoch=info["best_epoch"],
                             **{f"hist_{h}": np.array(info["history"][h]) for h in HISTORY_KEYS})
                    ckpt.unlink()    # fold finished -> its predictions are the resume point
                    print(f"  fold {k + 1} done: train {info['train_seconds']:.1f}s "
                          f"({info['epochs_run']} epochs, best epoch {info['best_epoch']}), "
                          f"test {info['test_seconds']:.2f}s")
                d = np.load(f)
                fold_metrics.append(compute_metrics(d["y"], d["pred"], d["prob"], d["type"]))
                fold_times.append({k2: float(d[k2]) for k2 in
                                   ("train_seconds", "test_seconds", "epochs_run", "best_epoch")})
                oof.append(d)

            oof_metrics = compute_metrics(*(np.concatenate([d[c] for d in oof]) for c in ("y", "pred", "prob", "type")))
            results[name] = summarize(fold_metrics, oof_metrics, fold_times)
            m, s, t = results[name]["mean"], results[name]["std"], results[name]["time"]
            print(f"{name}: acc={m['accuracy']:.4f}±{s['accuracy']:.4f}  f1={m['f1_ai']:.4f}±{s['f1_ai']:.4f}  "
                  f"auc={m['roc_auc']:.4f}  train={t['train_seconds_per_fold']:.1f}s/fold  "
                  f"test={t['test_seconds_per_fold']:.2f}s/fold")

        del data

    with open(out / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump({"mode": mode, "run": out.name, "n": len(df), "n_folds": n_folds, "config": mcfg,
                   "truncate_syllables": cfg.get("truncate_syllables"), "max_length": cfg["max_length"],
                   "extract_seconds": extract_seconds, "results": results}, fp, indent=2)
    print(f"Saved {out / 'metrics.json'}")

if __name__ == "__main__":
    main()
