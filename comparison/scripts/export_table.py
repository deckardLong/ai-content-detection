"""Export the main results table (Markdown + CSV + LaTeX), best value per column in bold.

    python comparison/scripts/export_table.py                  # 5-fold CV, mean ± std (main table)
    python comparison/scripts/export_table.py --source test    # notebook test set (766 samples)
    python comparison/scripts/export_table.py --run frozen     # an older run (default: from config)

Source: results/<run>/metrics.json (run_all.py) or results/eval/<run>/<name>/metrics.json (evaluate.py)
Output: main_table.{md,csv,tex} + time_table.{md,csv,tex} in results/<run>/ or results/eval/<run>/
"""
import argparse, json, os
from pathlib import Path
import pandas as pd
import yaml

os.chdir(Path(__file__).resolve().parents[1])

ORDER = ["phobert+cnn", "phobert+bilstm", "bamibert+cnn", "bamibert+bilstm"]
DISPLAY = {"phobert": "PhoBERT", "bamibert": "BamiBERT", "cnn": "CNN", "bilstm": "BiLSTM"}

# Binary task: Precision/Recall/F1 of the AI class (positive label = 1)
COLS = {"Accuracy": "accuracy", "Precision": "precision_ai", "Recall": "recall_ai", "F1": "f1_ai"}

def display_name(name):
    enc, head = name.split("+")
    return f"{DISPLAY[enc]} + {DISPLAY[head]}"

def ordered(names):
    return [n for n in ORDER if n in names] + sorted(n for n in names if n not in ORDER)

def encoder_word(run):
    return "đóng băng" if run.startswith("frozen") else "fine-tune"

def load_cv(run):
    path = Path(f"results/{run}/metrics.json")
    if not path.exists():
        raise SystemExit(f"No 5-fold CV results in {path.parent}/ - run scripts/run_all.py first, "
                         f"or use --source test for the train.py/evaluate.py results")
    r = json.load(open(path, encoding="utf-8"))
    res = r["results"]
    mean = {display_name(n): {c: res[n]["mean"][k] for c, k in COLS.items()} for n in ordered(res)}
    std = {display_name(n): {c: res[n]["std"][k] for c, k in COLS.items()} for n in ordered(res)}
    caption = (f"Kết quả {r['n_folds']}-fold cross-validation trên {r['n']} mẫu (mean $\\pm$ std), "
               f"encoder {encoder_word(run)}")

    times = {}
    for n in ordered(res):
        t = res[n].get("time")
        if t:
            times[display_name(n)] = {
                "Trích đặc trưng (s)": r.get("extract_seconds", {}).get(n.split("+")[0], float("nan")),
                "Train / fold (s)": t["train_seconds_per_fold"],
                f"Train {r['n_folds']} fold (s)": t["train_seconds_total"],
                "Test / fold (s)": t["test_seconds_per_fold"],
                "Epoch tốt nhất (TB)": t["best_epoch"],
            }
    return pd.DataFrame(mean).T, pd.DataFrame(std).T, caption, Path(f"results/{run}"), \
        (pd.DataFrame(times).T if times else None)

def load_test(run):
    out = Path(f"results/eval/{run}")
    files = {p.parent.name: p for p in out.glob("*/metrics.json")}
    if not files:
        raise SystemExit(f"No results in {out}/ - run evaluate.py --all --run {run} first")
    rows, times = {}, {}
    for n in ordered(files):
        m = json.load(open(files[n], encoding="utf-8"))
        rows[display_name(n)] = {c: m[k] for c, k in COLS.items()}
        if "test_seconds" in m:
            times[display_name(n)] = {
                "Trích đặc trưng (s)": m["extract_seconds"],
                "Train (s)": m["train_seconds"],
                "Test (s)": m["test_seconds"],
                "Test (ms/mẫu)": m["test_ms_per_sample"],
                "Epoch tốt nhất": m["best_epoch"],
            }
    caption = f"Kết quả trên tập test, encoder {encoder_word(run)}"
    return pd.DataFrame(rows).T[list(COLS)], None, caption, out, \
        (pd.DataFrame(times).T if times else None)

def render(mean, std, digits, bold, pm):
    best = mean.round(digits).max()    # ties at the shown precision are all bolded
    cells = {}
    for c in mean.columns:
        col = {}
        for n in mean.index:
            v = f"{mean.at[n, c]:.{digits}f}"
            if round(mean.at[n, c], digits) == best[c]:
                v = bold(v)
            if std is not None:
                v += f" {pm} {std.at[n, c]:.{digits}f}"
            col[n] = v
        cells[c] = col
    return pd.DataFrame(cells)[list(mean.columns)]

def to_latex(table, caption, label):
    cols = list(table.columns)
    lines = [
        r"\begin{table}[ht]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{l{'c' * len(cols)}}}", r"\hline",
        "Mô hình & " + " & ".join(cols) + r" \\", r"\hline",
    ]
    lines += [f"{n} & " + " & ".join(table.loc[n]) + r" \\" for n in table.index]
    lines += [r"\hline", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cv", "test"], default="cv")
    ap.add_argument("--run", help="results/<run>/ (default: from config, e.g. frozen_equal_text)")
    ap.add_argument("--digits", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    run = args.run or (f"{cfg['mode']}_{cfg['run_name']}" if cfg.get("run_name") else cfg["mode"])

    mean, std, caption, out, times = (load_cv if args.source == "cv" else load_test)(run)

    md = render(mean, std, args.digits, lambda v: f"**{v}**", "±")
    tex = render(mean, std, args.digits, lambda v: rf"\textbf{{{v}}}", r"$\pm$")
    csv = mean.round(args.digits)
    if std is not None:
        csv = csv.join(std.round(args.digits).add_suffix(" (std)"))

    md_text = md.rename_axis("Mô hình").to_markdown()
    (out / "main_table.md").write_text(md_text + "\n", encoding="utf-8")
    csv.rename_axis("Mô hình").to_csv(out / "main_table.csv", encoding="utf-8-sig")
    (out / "main_table.tex").write_text(
        to_latex(tex, caption, f"tab:{run}-{args.source}"), encoding="utf-8")
    print(md_text)
    print(f"\nSaved {out / 'main_table.md'}, main_table.csv, main_table.tex")

    if times is not None:
        t = times.astype(float).round(2)
        t_md = t.rename_axis("Mô hình").to_markdown()
        (out / "time_table.md").write_text(t_md + "\n", encoding="utf-8")
        t.rename_axis("Mô hình").to_csv(out / "time_table.csv", encoding="utf-8-sig")
        t_tex = t.map(lambda v: f"{v:.2f}" if v == v else "-")
        (out / "time_table.tex").write_text(to_latex(
            t_tex, "Thời gian huấn luyện và kiểm thử (giây). Đặc trưng của mỗi encoder được trích một "
                   "lần và dùng chung cho cả CNN và BiLSTM", f"tab:{run}-{args.source}-time"),
            encoding="utf-8")
        print("\n" + t_md + f"\n\nSaved {out / 'time_table.md'}, time_table.csv, time_table.tex")

if __name__ == "__main__":
    main()
