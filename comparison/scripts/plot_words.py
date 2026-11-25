"""Word analysis plots (punctuation and numbers removed, words only).

    python comparison/scripts/plot_words.py                       # BamiBERT + CNN predictions
    python comparison/scripts/plot_words.py --model phobert+cnn

1. top20_words.png          : 20 most frequent words of Human vs AI texts
2. wordcloud_errors.png     : word clouds of misclassified texts (Human -> AI and AI -> Human)
3. *_nostop.png             : the same, with common Vietnamese function words removed

Words come from the text the model reads: HTML/URL cleaning -> first N syllables (config) ->
underthesea word segmentation (compound words stay together, e.g. "công nghệ").
Misclassified = out-of-fold predictions of the chosen model, pooled over the 5 folds, so every text is
predicted once by a model that never saw it.
Output: results/<run>/plots/
"""
import argparse, os, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from wordcloud import WordCloud

from src.data.loading import load_dataframe, truncate_syllables
from src.data.segmentation import segment_all

FONT = r"C:\Windows\Fonts\arial.ttf"      # has Vietnamese glyphs (needed by WordCloud)
STOP = set("""và của là các có trong được cho một những với này để khi không đã người từ như về
cũng đó nhiều các theo ra tại sẽ còn nên hơn rất bị vào lại sau trên đang nhưng hoặc nếu mà
đến thì ở bạn chúng tôi họ nó mới hay vì do qua đây việc cả mỗi vẫn bằng trước giữa phải
nhất đều ra năm tới nào làm lên""".split())

def words_of(text):
    """Lower-case words of a segmented text; drop punctuation and pure numbers."""
    out = []
    for tok in text.split():
        tok = tok.strip("_").lower()
        if re.search(r"[^\W\d_]", tok):                       # has at least one letter
            tok = re.sub(r"[^\w_]|[\d]", "", tok).strip("_")
            if tok:
                out.append(tok.replace("_", " "))
    return out

def count(texts, stop=False):
    c = Counter(w for t in texts for w in words_of(t))
    if stop:
        c = Counter({w: n for w, n in c.items() if w not in STOP and len(w) > 1})
    return c

def plot_top20(cnt_h, cnt_a, path, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=False)
    for ax, cnt, name, color in ((axes[0], cnt_h, "Human", "#4C78A8"), (axes[1], cnt_a, "AI", "#E45756")):
        top = cnt.most_common(20)[::-1]
        ax.barh([w for w, _ in top], [n for _, n in top], color=color)
        for y, (_, n) in enumerate(top):
            ax.text(n, y, f" {n:,}", va="center", fontsize=8)
        ax.set_title(f"{name}{title_suffix}", fontweight="bold")
        ax.set_xlabel("Số lần xuất hiện")
        ax.margins(x=0.12)
    fig.suptitle("20 từ xuất hiện nhiều nhất", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_clouds(cnt_fp, cnt_fn, n_fp, n_fn, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, cnt, title in ((axes[0], cnt_fp, f"Human bị dự đoán thành AI ({n_fp} văn bản)"),
                           (axes[1], cnt_fn, f"AI bị dự đoán thành Human ({n_fn} văn bản)")):
        wc = WordCloud(width=1100, height=700, background_color="white", font_path=FONT,
                       max_words=120, collocations=False, colormap="viridis", prefer_horizontal=0.9)
        wc.generate_from_frequencies(cnt)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bamibert+cnn", help="model whose out-of-fold predictions define the errors")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    run = f"{cfg['mode']}_{cfg['run_name']}"
    out = Path("results") / run / "plots"
    out.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(cfg["data_path"], cfg["text_col"])
    texts = [truncate_syllables(t, cfg.get("truncate_syllables")) for t in df["text"]]
    seg = segment_all(texts)                                   # cached (same texts as PhoBERT training)
    y = df["label"].to_numpy()

    # --- 1) top-20 words of Human vs AI
    h, a = [seg[i] for i in np.where(y == 0)[0]], [seg[i] for i in np.where(y == 1)[0]]
    ch, ca = count(h), count(a)
    plot_top20(ch, ca, out / "top20_words.png")
    plot_top20(count(h, True), count(a, True), out / "top20_words_nostop.png", " (bỏ từ chức năng)")
    print(f"top20: Human {sum(ch.values()):,} words, AI {sum(ca.values()):,} words")

    # --- 2) word clouds of misclassified texts (pooled out-of-fold predictions)
    pred = np.full(len(df), -1)
    for k in range(cfg["n_folds"]):
        d = np.load(out.parent / "preds" / f"{args.model}_fold{k}.npz", allow_pickle=True)
        pred[d["idx"]] = d["pred"]
    assert (pred >= 0).all(), "missing predictions"
    fp = np.where((y == 0) & (pred == 1))[0]                   # human predicted as AI
    fn = np.where((y == 1) & (pred == 0))[0]                   # AI predicted as human
    print(f"{args.model}: accuracy={(pred == y).mean():.4f}  human->AI={len(fp)}  AI->human={len(fn)}")
    for stop, suffix in ((False, ""), (True, "_nostop")):
        plot_clouds(count([seg[i] for i in fp], stop), count([seg[i] for i in fn], stop),
                    len(fp), len(fn), out / f"wordcloud_errors{suffix}.png")
    for i_type, name in ((1, "AI-Generated"), (2, "AI-Rewritten")):
        pass
    types = df["generated_type"].to_numpy()
    for t in ("generated", "rewrited"):
        print(f"  AI->human among {t}: {int(((types[fn]) == t).sum())}")
    print(f"saved to {out}")

if __name__ == "__main__":
    main()
