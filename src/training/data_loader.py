from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True):
    """
    Create a Pytorch DataLoader. 
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )