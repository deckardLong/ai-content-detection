import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def make_folds(df, n_splits, seed, val_size=0.1, cache_path='results/cache/folds.json'):
    """K-fold stratified by generated_type and grouped by group_id.

    Every sample is in test exactly once, and an original/rewrite pair always stays on the
    same side. Val (~val_size of all data) is split from the train part, also by group.
    """
    strat = df['generated_type'].values
    groups = df['group_id'].values

    cache_file = Path(cache_path)
    if cache_file.exists():
        s = json.loads(cache_file.read_text())
        if (s['n'], s['seed'], s['n_splits']) == (len(df), seed, n_splits):
            return [{k: np.array(v) for k, v in f.items()} for f in s['folds']]

    n_inner = max(2, round((1 - 1 / n_splits) / val_size))
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_val, test in outer.split(np.zeros(len(df)), strat, groups):
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        tr, va = next(inner.split(train_val, strat[train_val], groups[train_val]))
        fold = {'train': train_val[tr], 'val': train_val[va], 'test': test}
        g = {k: set(groups[v]) for k, v in fold.items()}
        assert not (g['train'] & g['val'] or g['train'] & g['test'] or g['val'] & g['test']), 'Group leakage'
        folds.append(fold)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        'n': len(df), 'seed': seed, 'n_splits': n_splits,
        'folds': [{k: v.tolist() for k, v in f.items()} for f in folds]}))
    return folds
