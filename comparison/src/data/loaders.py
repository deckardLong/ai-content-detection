import torch
from torch.utils.data import DataLoader, Subset
from .dataset import TextDataset
from .loading import truncate_syllables
from .segmentation import segment_all
from ..models.encoders import load_tokenizer

PHOBERT_MAX_LEN = 256

def encoder_max_length(enc_key, cfg):
    """max_length is either one int for all encoders or a {encoder: int} mapping."""
    m = cfg['max_length']
    m = m[enc_key] if isinstance(m, dict) else m
    return min(m, PHOBERT_MAX_LEN) if enc_key == 'phobert' else m

def encode_texts(enc_key, cfg, df):
    """Tokenize the whole dataset for one encoder (PhoBERT needs word segmentation first).
    Texts are first cut to the same N syllables for every encoder (cfg['truncate_syllables'])."""
    tokenizer = load_tokenizer(cfg['encoders'][enc_key])

    n = cfg.get('truncate_syllables')
    texts = [truncate_syllables(t, n) for t in df['text']]
    if enc_key == 'phobert':
        texts = segment_all(texts)

    return TextDataset(texts, df['label'].tolist(), tokenizer, encoder_max_length(enc_key, cfg))

def make_loaders(dataset, split, batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(Subset(dataset, split['train']), batch_size=batch_size,
                       shuffle=True, generator=g)
    val = DataLoader(Subset(dataset, split['val']), batch_size=batch_size)
    test = DataLoader(Subset(dataset, split['test']), batch_size=batch_size)
    return train, val, test
