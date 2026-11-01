import torch
from torch.utils.data import DataLoader, Subset
from .dataset import TextDataset
from .segmentation import segment_all
from ..models.encoders import load_tokenizer

PHOBERT_MAX_LEN = 256

def encode_texts(enc_key, cfg, df):
    """Tokenize the whole dataset for one encoder (PhoBERT needs word segmentation first)."""
    tokenizer = load_tokenizer(cfg['encoders'][enc_key])

    texts = df['text'].tolist()
    max_len = cfg['max_length']
    if enc_key == 'phobert':
        texts = segment_all(texts)
        max_len = min(max_len, PHOBERT_MAX_LEN)

    return TextDataset(texts, df['label'].tolist(), tokenizer, max_len)

def make_loaders(dataset, split, batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(Subset(dataset, split['train']), batch_size=batch_size,
                       shuffle=True, generator=g)
    val = DataLoader(Subset(dataset, split['val']), batch_size=batch_size)
    test = DataLoader(Subset(dataset, split['test']), batch_size=batch_size)
    return train, val, test
