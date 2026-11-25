"""Extra paper tables from a finished 5-fold run: accuracy by text type + McNemar significance.

    python comparison/scripts/export_extra_tables.py [--run frozen_equal_text]

Output (results/<run>/): type_table.{md,tex}, significance_table.{md,tex}
Needs: metrics.json (run_all.py) and significance.csv (compare_significance.py).
"""
import argparse, json, os
from pathlib import Path
import pandas as pd
import yaml

from export_table import DISPLAY, ordered, display_name, render, to_latex   # also chdir()s to comparison/

TYPE_COLS = {"Human": "acc_human", "AI-generated": "acc_generated", "AI-rewritten": "acc_rewrited",
             "ROC-AUC": "roc_auc"}

def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--digits", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    run = args.run or f"{cfg['mode']}_{cfg['run_name']}"
    out = Path(f"results/{run}")

    r = json.load(open(out / "metrics.json", encoding="utf-8"))["results"]
    mean = pd.DataFrame({display_name(n): {c: r[n]["mean"][k] for c, k in TYPE_COLS.items()}
                         for n in ordered(r)}).T
    std = pd.DataFrame({display_name(n): {c: r[n]["std"][k] for c, k in TYPE_COLS.items()}
                        for n in ordered(r)}).T
    md = render(mean, std, args.digits, lambda v: f"**{v}**", "±")
    tex = render(mean, std, args.digits, lambda v: rf"\textbf{{{v}}}", r"$\pm$")
    (out / "type_table.md").write_text(md.rename_axis("Mô hình").to_markdown() + "\n", encoding="utf-8")
    (out / "type_table.tex").write_text(to_latex(
        tex, "Độ chính xác theo loại văn bản và ROC-AUC (mean $\\pm$ std trên 5 fold)",
        f"tab:{run}-bytype"), encoding="utf-8")
    print(md.rename_axis("Mô hình").to_markdown())

    sig = pd.read_csv(out / "significance.csv")
    rows = []
    for _, s in sig.iterrows():
        rows.append({"Mô hình A": display_name(s.model_a), "Mô hình B": display_name(s.model_b),
                     "Chỉ A đúng": int(s.a_only_correct), "Chỉ B đúng": int(s.b_only_correct),
                     "$p_{\\mathrm{Holm}}$": fmt_p(s.p_holm),
                     "Ý nghĩa": "Có" if s.significant else "Không"})
    sig_t = pd.DataFrame(rows)
    (out / "significance_table.md").write_text(sig_t.to_markdown(index=False) + "\n", encoding="utf-8")
    cols = list(sig_t.columns)
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{Kiểm định McNemar trên dự đoán out-of-fold gộp (hiệu chỉnh Holm)}",
             rf"\label{{tab:{run}-significance}}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{llcccc}", r"\hline", " & ".join(cols) + r" \\", r"\hline"]
    lines += [" & ".join(str(v) for v in row) + r" \\" for row in sig_t.values]
    lines += [r"\hline", r"\end{tabular}%", r"}", r"\end{table}"]
    (out / "significance_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + sig_t.to_markdown(index=False))
    print(f"\nSaved type_table / significance_table (.md, .tex) to {out}/")

if __name__ == "__main__":
    main()
