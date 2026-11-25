"""McNemar on pooled out-of-fold predictions (every sample once).  python comparison/scripts/compare_significance.py --mode frozen"""
import argparse, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from src.evaluation.significance import compare_all

import yaml
cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
ap = argparse.ArgumentParser()
ap.add_argument("--mode", "--run", dest="mode",
                default=f"{cfg['mode']}_{cfg['run_name']}" if cfg.get("run_name") else cfg["mode"],
                help="results/<run>/ (default: from config)")
mode = ap.parse_args().mode

parts = defaultdict(list)
for f in sorted(Path(f"results/{mode}/preds").glob("*_fold*.npz")):
    parts[f.stem.rsplit("_fold", 1)[0]].append(np.load(f, allow_pickle=True))

preds = {}
for name, ds in parts.items():
    idx = np.concatenate([d["idx"] for d in ds])
    order = np.argsort(idx)
    preds[name] = {"y": np.concatenate([d["y"] for d in ds])[order],
                   "pred": np.concatenate([d["pred"] for d in ds])[order]}

df = pd.DataFrame(compare_all(preds)).round(4)
print(df.to_markdown(index=False))
df.to_csv(f"results/{mode}/significance.csv", index=False)
