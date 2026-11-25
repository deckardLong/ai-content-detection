"""Loss curves (train / val / test per epoch) for the 4 models.

    python comparison/scripts/plot_losses.py                 # models trained by train.py (notebook split)
    python comparison/scripts/plot_losses.py --source cv     # 5-fold CV: mean over folds, band = ±1 std

Output (results/<run>/plots/):
    loss_<source>_all.png         train | val | test loss, one colour per model
    loss_<source>_<model>.png     one model: train, val and test loss together
Test loss is recorded for these curves only; the kept weights are chosen on val macro-F1.
"""
import argparse, os
from collections import defaultdict
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import torch
import yaml

ORDER = ["phobert+cnn", "phobert+bilstm", "bamibert+cnn", "bamibert+bilstm",
         "phobert+cnn_bilstm", "bamibert+cnn_bilstm"]   # last two: combined head (dashed)
DISPLAY = {"phobert": "PhoBERT", "bamibert": "BamiBERT", "cnn": "CNN", "bilstm": "BiLSTM", "cnn_bilstm": "CNN‖BiLSTM"}
# Categorical slots 1-4 in fixed order (validated: CVD-separable on the light surface)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd9", "#6b6b6b"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SPLITS = [("train_loss", "Train loss"), ("val_loss", "Validation loss"), ("test_loss", "Test loss")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})

def display_name(name):
    enc, head = name.split("+")
    return f"{DISPLAY[enc]} + {DISPLAY[head]}"

def load_train(run):
    """{model: {split: (mean, None), 'best_epoch': int}} from models/<run>/*.pt"""
    curves = {}
    for p in Path(f"models/{run}").glob("*.pt"):
        m = torch.load(p, map_location="cpu", weights_only=False)
        if "history" not in m:
            continue
        curves[p.stem] = {k: (np.array(m["history"][k]), None) for k, _ in SPLITS}
        curves[p.stem]["best_epoch"] = m["best_epoch"]
    return curves

def load_cv(run):
    """Mean ± std over folds per epoch from results/<run>/preds/*_fold*.npz"""
    parts = defaultdict(list)
    files = (list(Path(f"results/{run}/preds").glob("*_fold*.npz"))
             + list(Path(f"results/{run}/preds_hybrid").glob("*_fold*.npz")))   # combined-head models
    for f in files:
        d = np.load(f)
        if "hist_train_loss" in d:
            parts[f.stem.rsplit("_fold", 1)[0]].append(d)
    curves = {}
    for name, ds in parts.items():
        n = min(len(d["hist_train_loss"]) for d in ds)       # epochs every fold reached
        curves[name] = {}
        for k, _ in SPLITS:
            a = np.stack([d[f"hist_{k}"][:n] for d in ds])
            curves[name][k] = (a.mean(0), a.std(0, ddof=1) if len(ds) > 1 else None)
        curves[name]["best_epoch"] = None
        curves[name]["n_folds"] = len(ds)
    return curves

def label_ends(ax, ends, min_gap=0.055):
    """Direct labels at the right end of each line, nudged apart so they never overlap."""
    lo, hi = ax.get_ylim()
    ys = sorted(((y - lo) / (hi - lo), text) for y, text in ends)
    placed = []
    for y, text in ys:
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap
        placed.append((y, text))
    for y, text in placed:
        ax.annotate(text, xy=(1.01, y), xycoords="axes fraction", va="center", fontsize=8.5, color=INK)


LANGS = {
    "vi": {"note_best": "  (chấm tròn = epoch được chọn theo val macro-F1)", "note_std": "  (vùng mờ = ±1 std giữa các fold)",
           "best_epoch": "epoch được chọn: {}", "title_train": "train.py, tập chia giống notebook",
           "title_cv": "{}-fold CV, trung bình các fold"},
    "en": {"note_best": "  (dot = epoch selected by validation macro-F1)", "note_std": "  (shaded area = ±1 std across folds)",
           "best_epoch": "selected epoch: {}", "title_train": "train.py, notebook split",
           "title_cv": "{}-fold CV, mean over folds"},
}
TXT = dict(LANGS["vi"])
def plot_all(curves, title, path):
    names = [n for n in ORDER if n in curves]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    for ax, (key, label) in zip(axes, SPLITS):
        ends = []
        for name in names:
            color = SERIES[ORDER.index(name)]
            mean, std = curves[name][key]
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=color, linewidth=2, label=display_name(name),
                    linestyle="--" if name.endswith("cnn_bilstm") else "-")
            if std is not None:
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
            best = curves[name]["best_epoch"]
            if best and key != "train_loss":
                ax.plot(best, mean[best - 1], "o", markersize=8, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
            ends.append((mean[-1], display_name(name)))
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_xlim(1, max(len(curves[n][key][0]) for n in names))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(bottom=0)
        label_ends(ax, ends)
    handles, labels = axes[0].get_legend_handles_labels()
    note = TXT["note_best"] if any(curves[n]["best_epoch"] for n in names) else TXT["note_std"]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.905, title + note, ha="center", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 0.97, 0.88))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_one(name, c, title, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ends = []
    for (key, label), color in zip(SPLITS, SERIES):
        mean, std = c[key]
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, color=color, linewidth=2, label=label)
        if std is not None:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ends.append((mean[-1], label))
    if c["best_epoch"]:
        ax.axvline(c["best_epoch"], color=INK_2, linewidth=1, linestyle="--")
        ax.annotate(TXT["best_epoch"].format(c["best_epoch"]), xy=(c["best_epoch"], 1.0),
                    xycoords=("data", "axes fraction"), xytext=(4, -12), textcoords="offset points",
                    fontsize=8.5, color=INK_2)
    ax.set_title(f"{display_name(name)} - {title}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_xlim(1, len(c["train_loss"][0])); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.14))
    label_ends(ax, ends)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["train", "cv"], default="train")
    ap.add_argument("--lang", choices=["vi", "en"], default="vi", help="language of the figure text")
    ap.add_argument("--run", default=f"{cfg['mode']}_{cfg['run_name']}" if cfg.get("run_name") else cfg["mode"])
    args = ap.parse_args()
    TXT.update(LANGS[args.lang])
    sfx = '' if args.lang == 'vi' else '_en'

    curves = load_train(args.run) if args.source == "train" else load_cv(args.run)
    if not curves:
        raise SystemExit(f"No loss history found for run '{args.run}' ({args.source}) - "
                         f"run {'train.py' if args.source == 'train' else 'run_all.py'} first")
    title = (TXT["title_train"] if args.source == "train"
             else TXT["title_cv"].format(next(iter(curves.values()))["n_folds"]))
    out = Path(f"results/{args.run}/plots")
    out.mkdir(parents=True, exist_ok=True)

    plot_all(curves, title, out / f"loss_{args.source}_all{sfx}.png")
    for name in curves:
        plot_one(name, curves[name], title, out / f"loss_{args.source}_{name}{sfx}.png")
    print(f"Saved {len(curves) + 1} figures to {out}/")

if __name__ == "__main__":
    main()
