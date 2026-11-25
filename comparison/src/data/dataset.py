import numpy as np
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.max_length = max_length
        self.enc = tokenizer(
            texts, truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt',
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.enc['input_ids'][idx],
            'attention_mask': self.enc['attention_mask'][idx],
            'labels': self.labels[idx],
        }

class FeatureDataset(Dataset):
    """last_hidden_state pre-extracted by a frozen encoder (memmap fp16)."""
    def __init__(self, features, attention_mask, labels):
        self.features = features
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        # One sorted fancy-index read per batch is much faster on a memmap than one read per sample.
        # Kept as fp16 here (half the copy cost); cast to fp32 on the GPU.
        order = np.argsort(indices)
        block = np.asarray(self.features[np.asarray(indices)[order]])
        feats = torch.from_numpy(block)[torch.from_numpy(np.argsort(order))]
        return [{
            'hidden_states': feats[i],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        } for i, idx in enumerate(indices)]
