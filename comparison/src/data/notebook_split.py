import numpy as np
import pandas as pd

def notebook_split(df, subsets_dir='../data/subsets'):
    """The 70/15/15 group split from notebooks/03_model_training.ipynb (data/subsets/*.csv),
    mapped back to df rows by cleaned text. Val/test are taken as saved; train = all other rows."""
    pos = {t: i for i, t in enumerate(df['text'])}
    split = {}
    for name in ('val', 'test'):
        sub = pd.read_csv(f'{subsets_dir}/{name}_df.csv', encoding='utf-8-sig')
        idx = sub['cleaned_text'].map(pos)
        assert idx.notna().all(), f'{name}_df.csv has rows not found in the dataset'
        split[name] = idx.astype(int).values
    split['train'] = np.setdiff1d(np.arange(len(df)), np.concatenate([split['val'], split['test']]))

    groups = {k: set(df['group_id'].values[v]) for k, v in split.items()}
    assert not (groups['train'] & groups['test'] or groups['train'] & groups['val']), 'Group leakage'
    return split
