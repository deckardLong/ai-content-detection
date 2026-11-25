"""Evaluate exported models on the notebook test set (as in notebooks/04_evaluation.ipynb).

    python comparison/scripts/evaluate.py --model models/frozen_equal_text/bamibert+cnn.pt
    python comparison/scripts/evaluate.py --all                        # run from config
    python comparison/scripts/evaluate.py --all --run frozen           # an older run

The model runs end-to-end on raw test text (tokenizer -> encoder -> head), so this also checks
that the exported file works on its own. Test time = encoder + head on the 766 test texts.
Output: comparison/results/eval/<run>/<name>/
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from torch.utils.data import DataLoader, Subset
from transformers.utils import logging as hf_logging

from src.data.loaders import encode_texts
from src.data.loading import load_dataframe
from src.data.notebook_split import notebook_split
from src.evaluation.metrics import compute_metrics
from src.models.classifier import EncoderClassifier
from src.training.trainer import predict

hf_logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path):
    meta = torch.load(path, map_location="cpu", weights_only=False)
    model = EncoderClassifier(meta["encoder_name"], meta["head_type"], meta["head_cfg"],
                              meta["num_classes"], meta["dropout"])
    (model.head if meta["scope"] == "head" else model).load_state_dict(meta["state_dict"])
    return model.to(device).eval(), meta

def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["Human", "AI"]); ax.set_yticks([0, 1], ["Human", "AI"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Label")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_roc(curves, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Test Set", fontweight="bold")
    ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="path to an exported .pt file")
    ap.add_argument("--all", action="store_true", help="evaluate every model in models/<mode>/")
    ap.add_argument("--run", help="models/<run>/ to evaluate with --all (default: from config, "
                                  "e.g. frozen_equal_text)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    run = args.run or (f"{cfg['mode']}_{cfg['run_name']}" if cfg.get("run_name") else cfg["mode"])
    paths = sorted(Path(f"models/{run}").glob("*.pt")) if args.all else [Path(args.model)]
    if not paths or not all(p.exists() for p in paths):
        ap.error(f"no model file found: {paths}")
    df = load_dataframe(cfg["data_path"], cfg["text_col"])
    test_idx = notebook_split(df)["test"]
    test_df = df.iloc[test_idx].reset_index(drop=True)
    types = test_df["generated_type"].values

    summary, curves = {}, {}
    for path in paths:
        model, meta = load_model(path)
        name = path.stem
        run_dir = Path("results/eval") / meta.get("run", meta["mode"])
        out = run_dir / name
        out.mkdir(parents=True, exist_ok=True)

        # Same truncation / token budget as in training (stored in the model file)
        enc_cfg = {**cfg, "max_length": {meta["encoder_key"]: meta["max_length"]},
                   "truncate_syllables": meta.get("truncate_syllables")}
        dataset = encode_texts(meta["encoder_key"], enc_cfg, df)
        loader = DataLoader(Subset(dataset, test_idx), batch_size=32)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y, prob = predict(model, loader, device)    # end-to-end: encoder + head on tokenized text
        if device == "cuda":
            torch.cuda.synchronize()
        test_seconds = time.perf_counter() - t0
        pred = (prob >= 0.5).astype(int)

        print(f"\n===== {name} ({meta.get('run', meta['mode'])}) - test set: {len(y)} samples =====")
        print(f"  test time (encoder + head): {test_seconds:.2f}s = {1000 * test_seconds / len(y):.2f} ms/sample"
              f" | train time: {meta.get('train_seconds', float('nan')):.1f}s"
              f" + feature extraction {meta.get('extract_seconds', 0.0):.1f}s")
        print(classification_report(y, pred, target_names=["Human", "AI"], digits=4))
        for t in ["human", "rewrited", "generated"]:
            sel = types == t
            print(f"  {t:10s}: {(pred[sel] == y[sel]).mean():.2%} ({sel.sum()} samples)")

        summary[name] = compute_metrics(y, pred, prob, types)
        summary[name].update({
            "train_seconds": float(meta.get("train_seconds", float("nan"))),
            "extract_seconds": float(meta.get("extract_seconds", 0.0)),
            "test_seconds": test_seconds, "test_ms_per_sample": 1000 * test_seconds / len(y),
            "best_epoch": meta.get("best_epoch"),
        })
        plot_confusion(confusion_matrix(y, pred), f"Confusion Matrix - {name}", out / "confusion_matrix.png")
        fpr, tpr, _ = roc_curve(y, prob)
        curves[name] = (fpr, tpr, summary[name]["roc_auc"])
        plot_roc({name: curves[name]}, out / "roc_curve.png")
        test_df.assign(prediction=pred, prob_ai=prob).drop(columns=["text", "original_text"]) \
            .to_csv(out / "test_df_evaluated.csv", index=False, encoding="utf-8-sig")
        with open(out / "metrics.json", "w", encoding="utf-8") as fp:
            json.dump(summary[name], fp, indent=2)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(paths) > 1:
        out = run_dir
        table = pd.DataFrame(summary).T.round(4)
        print("\n" + table[["accuracy", "precision_ai", "recall_ai", "f1_ai", "roc_auc",
                            "acc_human", "acc_rewrited", "acc_generated",
                            "train_seconds", "test_seconds"]].to_markdown())
        table.to_csv(out / "summary.csv")
        plot_roc(curves, out / "roc_curves_all.png")

if __name__ == "__main__":
    main()
