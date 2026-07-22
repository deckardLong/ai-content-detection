import torch
import pandas as pd
from torch.utils.data import Dataset

class AIDetectionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        """
        Dataframe containing input_ids, attention_mask and label.
        """
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        """
        Return number of samples.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Return one sample.
        """
        row = self.df.iloc[index]

        return {
            'input_ids': torch.tensor(row['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(row['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }