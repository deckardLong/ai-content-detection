import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .dataset import FeatureDataset
from ..models.encoders import load_encoder

@torch.no_grad()
def extract_features(encoder_name, dataset, device, cache_path, batch_size=32):
    """Run the frozen encoder once over the whole dataset and store last_hidden_state
    (fp16, memmap .npy) so the heads can be trained without recomputing the encoder."""
    path = Path(cache_path)
    n, L = len(dataset), dataset.max_length

    if not path.exists():
        encoder = load_encoder(encoder_name).to(device).eval()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + '.tmp.npy')
        feats = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16,
                                          shape=(n, L, encoder.config.hidden_size))
        i = 0
        for b in tqdm(DataLoader(dataset, batch_size=batch_size), desc=f'extract {encoder_name}'):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                h = encoder(input_ids=b['input_ids'].to(device),
                            attention_mask=b['attention_mask'].to(device)).last_hidden_state
            feats[i:i + len(h)] = h.half().cpu().numpy()
            i += len(h)
        feats.flush()
        del feats, encoder
        if device == 'cuda':
            torch.cuda.empty_cache()
        os.replace(tmp, path)

    feats = np.load(path, mmap_mode='r')
    assert feats.shape[:2] == (n, L), f'Stale feature cache {path}, delete it'
    return FeatureDataset(feats, dataset.enc['attention_mask'], dataset.labels)
