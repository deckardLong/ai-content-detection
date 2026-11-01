import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

def clean_text(text):
    # Same cleaning as src/preprocessing/cleaner.py (used to train the original model)
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_groups(df):
    """A rewritten article shares the group of its original human article (matched by url,
    falling back to original_text), so a pair never ends up split across train and test."""
    group = np.arange(len(df))
    human = df['generated_type'] == 'human'
    by_url = dict(zip(df.loc[human, 'url'], df.index[human]))
    by_text = dict(zip(df.loc[human, 'text'].str.strip(), df.index[human]))

    for i in df.index[df['generated_type'] == 'rewrited']:
        src = by_url.get(df.at[i, 'url'])
        if src is None and isinstance(df.at[i, 'original_text'], str):
            src = by_text.get(df.at[i, 'original_text'].strip())
        if src is not None:
            group[i] = src
    return group

def load_dataframe(path: str, text_col: str = 'text') -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    df = df.dropna(subset=[text_col, 'label']).reset_index(drop=True)
    df['text'] = df[text_col].astype(str)
    df['label'] = df['label'].astype(int)
    df['group_id'] = build_groups(df)   # before cleaning: matches raw original_text
    df['text'] = df['text'].map(clean_text)
    return df

def subsample(df, n, seed):
    """Small stratified subset (by generated_type) for smoke tests."""
    frac = min(1.0, n / len(df))
    return (df.groupby('generated_type', group_keys=False)
              .sample(frac=frac, random_state=seed)
              .reset_index(drop=True))
