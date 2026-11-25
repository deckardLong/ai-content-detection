"""5-fold CV for a frozen encoder + the combined CNN||BiLSTM head, on the SAME folds and protocol
as the 4-model comparison (scripts/run_all.py), plus McNemar vs the four existing models.

    python comparison/scripts/cv_hybrid.py                          # BamiBERT (default)
    python comparison/scripts/cv_hybrid.py --encoder phobert
    python comparison/scripts/cv_hybrid.py --encoder phobert --retime   # re-measure time only
    python comparison/scripts/cv_hybrid.py --folds 2 --epochs 2 --limit 200      # smoke test

Reads    results/<run>/folds.json and results/<run>/preds/*.npz   (made by run_all.py)
Writes   results/<run>/preds_hybrid/<encoder>+cnn_bilstm_fold<k>.npz
         results/<run>/hybrid_metrics[_phobert].json      (same layout as metrics.json, one model)
         results/<run>/significance_hybrid[_phobert].csv  (hybrid vs each of the 4 models, Holm over those 4)
  --retime: results/<run>/preds_hybrid_retime/ and hybrid_timing_<encoder>.json only
The existing metrics.json / significance.csv / preds/ are NOT touched.
Resume: re-run the same command; finished folds are skipped.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

from run_all import HISTORY_KEYS, device, run_id, run_one, summarize
from src.data.features import extract_features, feature_cache_path
from src.data.loaders import encode_texts
from src.data.loading import load_dataframe, subsample
from src.data.splits import make_folds
from src.evaluation.metrics import compute_metrics
from src.evaluation.significance import mcnemar_pair

HEAD = "cnn_bilstm"
ENC_KEY = "bamibert"   # overridden by --encoder
NAME = f"{ENC_KEY}+{HEAD}"

def load_oof(files):
    """Pooled out-of-fold labels and predictions in dataset order."""
    ds = [np.load(f, allow_pickle=True) for f in sorted(files)]
    idx = np.concatenate([d["idx"] for d in ds])
    order = np.argsort(idx)
    return (np.concatenate([d["y"] for d in ds])[order], np.concatenate([d["pred"] for d in ds])[order])

def main():
    global ENC_KEY, NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="bamibert", choices=["bamibert", "phobert"])
    ap.add_argument("--retime", action="store_true",
                    help="re-measure time only: write to preds_hybrid_retime/ and hybrid_timing_<enc>.json, "
                         "leave existing results and the McNemar table untouched")
    ap.add_argument("--limit", type=int, help="stratified subset of N samples (smoke test)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--folds", type=int)
    args = ap.parse_args()
    ENC_KEY = args.encoder
    NAME = f"{ENC_KEY}+{HEAD}"
    suffix = "" if ENC_KEY == "bamibert" else f"_{ENC_KEY}"

    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    cfg["heads"] = {**cfg["heads"], **cfg["hybrid_head"]}
    mode = "frozen"
    mcfg = dict(cfg[mode])
    if args.epochs:
        mcfg["epochs"] = args.epochs
    n_folds = args.folds or cfg["n_folds"]
    smoke = bool(args.limit)

    df = load_dataframe(cfg["data_path"], cfg["text_col"])
    if smoke:
        df = subsample(df, args.limit, cfg["seed"])
    out = Path("results") / run_id(cfg, mode, smoke=smoke)
    pdir = "preds_hybrid_retime" if args.retime else "preds_hybrid"
    (out / pdir).mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    folds = make_folds(df, n_folds, cfg["seed"], cache_path=out / "folds.json")   # same folds as run_all
    types = df["generated_type"].to_numpy(dtype=str)
    print(f"device={device}  run={out.name}  n={len(df)}  folds={n_folds}  epochs={mcfg['epochs']}  "
          f"model={NAME}")

    pred_file = lambda k: out / pdir / f"{NAME}_fold{k}.npz"
    data, extract_seconds = None, None
    if not all(pred_file(k).exists() for k in range(n_folds)):
        data = encode_texts(ENC_KEY, cfg, df)
        data, extract_seconds = extract_features(
            cfg["encoders"][ENC_KEY], data, device, feature_cache_path(ENC_KEY, data))
        print(f"[{ENC_KEY}] features ready ({extract_seconds:.1f}s, cached)")

    fold_metrics, oof, fold_times = [], [], []
    for k, split in enumerate(folds):
        f = pred_file(k)
        if not f.exists():
            print(f"[{NAME}] fold {k + 1}/{n_folds}")
            ckpt = out / "ckpt" / f"{NAME}_fold{k}.pt"
            y, prob, info = run_one(ENC_KEY, HEAD, cfg, mcfg, True, data, split, cfg["seed"] + k, ckpt)
            np.savez(f, idx=split["test"], y=y, prob=prob, pred=(prob >= 0.5).astype(int),
                     type=types[split["test"]], train_seconds=info["train_seconds"],
                     test_seconds=info["test_seconds"], epochs_run=info["epochs_run"],
                     best_epoch=info["best_epoch"],
                     **{f"hist_{h}": np.array(info["history"][h]) for h in HISTORY_KEYS})
            ckpt.unlink(missing_ok=True)
            print(f"  fold {k + 1} done: train {info['train_seconds']:.1f}s "
                  f"({info['epochs_run']} epochs, best epoch {info['best_epoch']}), "
                  f"test {info['test_seconds']:.2f}s")
        d = np.load(f)
        fold_metrics.append(compute_metrics(d["y"], d["pred"], d["prob"], d["type"]))
        fold_times.append({c: float(d[c]) for c in
                           ("train_seconds", "test_seconds", "epochs_run", "best_epoch")})
        oof.append(d)

    oof_metrics = compute_metrics(*(np.concatenate([d[c] for d in oof]) for c in ("y", "pred", "prob", "type")))
    res = summarize(fold_metrics, oof_metrics, fold_times)
    m, s, t = res["mean"], res["std"], res["time"]
    print(f"{NAME}: acc={m['accuracy']:.4f}±{s['accuracy']:.4f}  f1={m['f1_ai']:.4f}±{s['f1_ai']:.4f}  "
          f"auc={m['roc_auc']:.4f}  train={t['train_seconds_per_fold']:.1f}s/fold")
    if args.retime:
        with open(out / f"hybrid_timing_{ENC_KEY}.json", "w", encoding="utf-8") as fp:
            json.dump({"model": NAME, "time": t, "per_fold": fold_times}, fp, indent=2)
        print("retime done ->", out / f"hybrid_timing_{ENC_KEY}.json")
        return
    with open(out / f"hybrid_metrics{suffix}.json", "w", encoding="utf-8") as fp:
        json.dump({"run": out.name, "n": len(df), "n_folds": n_folds, "config": mcfg,
                   "extract_seconds": extract_seconds, "results": {NAME: res}}, fp, indent=2)

    # McNemar: hybrid vs each existing model (Holm over these 4 comparisons)
    y_h, p_h = load_oof((out / "preds_hybrid").glob(f"{NAME}_fold*.npz"))
    rows = []
    for other in [f"{e}+{h}" for e in cfg["encoders"] for h in ("cnn", "bilstm")]:
        files = list((out / "preds").glob(f"{other}_fold*.npz"))
        if len(files) < n_folds:
            print(f"skip {other}: only {len(files)} fold files in {out / 'preds'}")
            continue
        y_o, p_o = load_oof(files)
        assert np.array_equal(y_h, y_o), "label mismatch -> folds differ"
        rows.append({"model_a": NAME, "model_b": other, **mcnemar_pair(y_h, p_h, p_o)})
    if rows:
        _, p_adj, _, _ = multipletests([r["p_value"] for r in rows], alpha=0.05, method="holm")
        for r, pa in zip(rows, p_adj):
            r["p_holm"], r["significant"] = float(pa), bool(pa < 0.05)
        sig = pd.DataFrame(rows).round(4)
        print(sig.to_markdown(index=False))
        sig.to_csv(out / f"significance_hybrid{suffix}.csv", index=False)

if __name__ == "__main__":
    main()
