"""Results table (mean ± std over folds).  python comparison/scripts/compare_models.py --mode frozen"""
import argparse, json, os
from pathlib import Path
import pandas as pd

os.chdir(Path(__file__).resolve().parents[1])

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="frozen")
mode = ap.parse_args().mode

r = json.load(open(f"results/{mode}/metrics.json", encoding="utf-8"))["results"]
main_cols = ["accuracy", "macro_f1", "precision_ai", "recall_ai", "f1_ai",
             "precision_human", "recall_human", "f1_human", "roc_auc"]
type_cols = sorted(k for k in next(iter(r.values()))["mean"] if k.startswith("acc_"))

def table(cols):
    return pd.DataFrame({name: {c: f"{v['mean'][c]:.4f} ± {v['std'][c]:.4f}" for c in cols}
                         for name, v in r.items()}).T

main, by_type = table(main_cols), table(type_cols)
print("## Overall\n" + main.to_markdown() + "\n\n## Accuracy by text type\n" + by_type.to_markdown())
main.join(by_type).to_csv(f"results/{mode}/comparison.csv")
