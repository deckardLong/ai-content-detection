"""Three analysis plots tied to the paper's questions.

    python comparison/scripts/plot_analysis.py

1. score_distribution.png : distribution of P(AI) by text type (out-of-fold, every text predicted once)
2. tsne_features.png      : t-SNE of mean-pooled frozen features, PhoBERT vs BamiBERT, coloured by type
3. error_overlap.png      : which samples only one encoder gets right, per text type (CNN and BiLSTM heads)

Reads results/<run>/preds/*.npz (from run_all.py) and the cached features. Output: results/<run>/plots/
"""
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data.features import extract_features, feature_cache_path
from src.data.loaders import encode_texts
from src.data.loading import load_dataframe

TYPES = [("human", "Human", "#4C78A8"), ("generated", "AI-Generated", "#E45756"),
         ("rewrited", "AI-Rewritten", "#F58518")]
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_preds(run_dir, name, n_folds, n):
    prob, pred, y, typ = np.zeros(n), np.zeros(n, int), np.zeros(n, int), np.empty(n, dtype=object)
    for k in range(n_folds):
        d = np.load(run_dir / "preds" / f"{name}_fold{k}.npz", allow_pickle=True)
        prob[d["idx"]], pred[d["idx"]], y[d["idx"]], typ[d["idx"]] = d["prob"], d["pred"], d["y"], d["type"]
    return prob, pred, y, typ

# ---------------------------------------------------------------- 1) score distribution
def plot_scores(run_dir, out, n_folds, n):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    bins = np.linspace(0, 1, 21)
    for r, name in enumerate(["bamibert+cnn", "phobert+cnn"]):
        prob, pred, y, typ = load_preds(run_dir, name, n_folds, n)
        wrong = pred != y
        conf_wrong = np.maximum(prob, 1 - prob)[wrong]
        print(f"{name}: errors={wrong.sum()}  confident errors (>=0.9 on the wrong side)="
              f"{(conf_wrong >= 0.9).sum()} ({(conf_wrong >= 0.9).mean():.1%})  "
              f"median confidence of errors={np.median(conf_wrong):.3f}")
        for c, (t, label, color) in enumerate(TYPES):
            ax = axes[r, c]
            sel = typ == t
            ax.hist(prob[sel], bins=bins, color=color, edgecolor="white")
            ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
            err = wrong[sel].mean()
            ax.set_title(f"{label} ({name})\nsai {err:.1%} ({int(wrong[sel].sum())}/{int(sel.sum())})", fontsize=10)
            ax.set_xlabel("P(AI)")
            if c == 0:
                ax.set_ylabel("Số văn bản")
            ax.set_yscale("log")
    fig.suptitle("Phân bố xác suất P(AI) theo loại văn bản (dự đoán out-of-fold, trục tung log)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "score_distribution.png", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- 2) t-SNE of frozen features
def pooled_features(cfg, df, enc_key):
    data = encode_texts(enc_key, cfg, df)
    feats, _ = extract_features(cfg["encoders"][enc_key], data, device, feature_cache_path(enc_key, data))
    mask = feats.attention_mask.numpy().astype(np.float32)
    out = np.zeros((len(df), feats.features.shape[2]), dtype=np.float32)
    for i in range(0, len(df), 250):
        block = np.asarray(feats.features[i:i + 250]).astype(np.float32)
        m = mask[i:i + 250][:, :, None]
        out[i:i + 250] = (block * m).sum(1) / m.sum(1)
    return out

def plot_tsne(cfg, df, out):
    typ = df["generated_type"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (enc_key, title) in zip(axes, (("phobert", "PhoBERT"), ("bamibert", "BamiBERT"))):
        x = pooled_features(cfg, df, enc_key)
        x = PCA(n_components=50, random_state=42).fit_transform(x)
        z = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(x)
        for t, label, color in TYPES:
            sel = typ == t
            ax.scatter(z[sel, 0], z[sel, 1], s=5, alpha=0.5, color=color, label=label)
        ax.set_title(f"{title} (đặc trưng đóng băng, trung bình theo token)", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(markerscale=3, loc="best")
    fig.suptitle("t-SNE đặc trưng của hai encoder, tô màu theo loại văn bản", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "tsne_features.png", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- 3) error overlap per type
def plot_overlap(run_dir, out, n_folds, n):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    cats = [("Chỉ BamiBERT đúng", "#2CA02C"), ("Chỉ PhoBERT đúng", "#9467BD"), ("Cả hai sai", "#7F7F7F")]
    for ax, head in zip(axes, ("cnn", "bilstm")):
        pb, yb, tb = load_preds(run_dir, f"bamibert+{head}", n_folds, n)[1:]
        pp, yp, _ = load_preds(run_dir, f"phobert+{head}", n_folds, n)[1:]
        assert (yb == yp).all()
        okb, okp = pb == yb, pp == yp
        counts = {}
        for t, label, _ in TYPES:
            s = tb == t
            counts[label] = [int((okb & ~okp & s).sum()), int((~okb & okp & s).sum()), int((~okb & ~okp & s).sum())]
            print(f"{head} {label:12s}: only BamiBERT={counts[label][0]}  only PhoBERT={counts[label][1]}  "
                  f"both wrong={counts[label][2]}  (n={int(s.sum())})")
        x = np.arange(len(TYPES))
        w = 0.26
        for j, (cat, color) in enumerate(cats):
            vals = [counts[label][j] for _, label, _ in TYPES]
            bars = ax.bar(x + (j - 1) * w, vals, w, color=color, label=cat)
            ax.bar_label(bars, fontsize=9, padding=2)
        ax.set_xticks(x, [label for _, label, _ in TYPES])
        ax.set_title(f"Head {head.upper() if head == 'cnn' else 'BiLSTM'}", fontweight="bold")
        ax.set_ylabel("Số văn bản")
    axes[0].legend()
    fig.suptitle("Hai encoder bất đồng trên những văn bản nào? (out-of-fold, 5 000 văn bản)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "error_overlap.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- compact versions used in the paper

PAPER_TXT = {
    "vi": {"wrong": "{}: sai {:.1%}", "xlabel": "Xác suất AI mô hình gán cho văn bản", "ylog": "Số văn bản (thang log)",
           "only_bami": "Chỉ BamiBERT đúng", "only_pho": "Chỉ PhoBERT đúng", "both_wrong": "Cả hai sai", "count": "Số văn bản", "count_short": "Số văn bản (log)"},
    "en": {"wrong": "{}: error rate {:.1%}", "xlabel": "P(AI) assigned by the model", "ylog": "Number of texts (log scale)",
           "only_bami": "Only BamiBERT correct", "only_pho": "Only PhoBERT correct", "both_wrong": "Both wrong", "count": "Number of texts", "count_short": "Texts (log)"},
}
def plot_paper_figures(run_dir, out, n_folds, n, lang="vi"):
    T = PAPER_TXT[lang]; sfx = "" if lang == "vi" else "_en"
    # (a) P(AI) of the main model, one row
    prob, pred, y, typ = load_preds(run_dir, "bamibert+cnn", n_folds, n)
    rew = typ == "rewrited"
    print(f"rewritten texts: wrong={int((pred[rew] != y[rew]).sum())}  with P(AI)<0.05: {int((prob[rew] < 0.05).sum())}  "
          f"with P(AI)<0.1: {int((prob[rew] < 0.1).sum())}")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, (t, label, color) in zip(axes, TYPES):
        sel = typ == t
        ax.hist(prob[sel], bins=bins, color=color, edgecolor="white")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_title(T["wrong"].format(label, (pred[sel] != y[sel]).mean()), fontsize=11, fontweight="bold")
        ax.set_xlabel(T["xlabel"])
    axes[0].set_ylabel(T["ylog"])
    fig.tight_layout()
    fig.savefig(out / f"paper_score_distribution{sfx}.png", dpi=200)
    plt.close(fig)

    # (b) encoder disagreement, CNN head only, single column
    pb, yb = load_preds(run_dir, "bamibert+cnn", n_folds, n)[1:3]
    pp = load_preds(run_dir, "phobert+cnn", n_folds, n)[1]
    okb, okp = pb == yb, pp == yb
    cats = [(T["only_bami"], "#2CA02C"), (T["only_pho"], "#9467BD"), (T["both_wrong"], "#7F7F7F")]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    x, w = np.arange(len(TYPES)), 0.26
    for j, (cat, color) in enumerate(cats):
        vals = []
        for t, _, _ in TYPES:
            s = typ == t
            vals.append([int((okb & ~okp & s).sum()), int((~okb & okp & s).sum()), int((~okb & ~okp & s).sum())][j])
        ax.bar_label(ax.bar(x + (j - 1) * w, vals, w, color=color, label=cat), fontsize=9, padding=2)
    ax.set_xticks(x, [label for _, label, _ in TYPES])
    ax.set_ylabel(T["count"])
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"paper_error_overlap{sfx}.png", dpi=200)
    plt.close(fig)
def plot_scores_vertical(run_dir, out, n_folds, n, lang="vi"):
    """Same content as the wide score figure but stacked vertically so it fits one column."""
    T = PAPER_TXT[lang]; sfx = "" if lang == "vi" else "_en"
    prob, pred, y, typ = load_preds(run_dir, "bamibert+cnn", n_folds, n)
    fig, axes = plt.subplots(3, 1, figsize=(3.6, 5.0), sharex=True)
    bins = np.linspace(0, 1, 21)
    for ax, (t, label, color) in zip(axes, TYPES):
        sel = typ == t
        ax.hist(prob[sel], bins=bins, color=color, edgecolor="white")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_title(T["wrong"].format(label, (pred[sel] != y[sel]).mean()), fontsize=9, fontweight="bold", pad=3)
        ax.set_ylabel(T["count_short"], fontsize=8)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel(T["xlabel"], fontsize=8)
    fig.tight_layout(h_pad=0.8)
    fig.savefig(out / f"paper_score_distribution_v{sfx}.png", dpi=250)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["scores", "tsne", "overlap", "paper"])
    ap.add_argument("--lang", choices=["vi", "en"], default="vi")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    run_dir = Path("results") / f"{cfg['mode']}_{cfg['run_name']}"
    out = run_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    df = load_dataframe(cfg["data_path"], cfg["text_col"])
    n, n_folds = len(df), cfg["n_folds"]
    if args.only in (None, "scores"):
        plot_scores(run_dir, out, n_folds, n)
    if args.only in (None, "overlap"):
        plot_overlap(run_dir, out, n_folds, n)
    if args.only in (None, "paper"):
        plot_paper_figures(run_dir, out, n_folds, n, args.lang)
        plot_scores_vertical(run_dir, out, n_folds, n, args.lang)
    if args.only in (None, "tsne"):
        plot_tsne(cfg, df, out)
    print(f"saved to {out}")

if __name__ == "__main__":
    main()
