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
        return {
            'hidden_states': torch.from_numpy(np.array(self.features[idx], dtype=np.float32)),
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }
