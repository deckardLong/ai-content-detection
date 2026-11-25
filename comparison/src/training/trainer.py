import torch
from tqdm.auto import tqdm

def _forward(model, batch, device):
    mask = batch['attention_mask'].to(device)
    if 'hidden_states' in batch:
        return model(mask, hidden_states=batch['hidden_states'].to(device).float())
    return model(mask, input_ids=batch['input_ids'].to(device))

def train_epoch(model, loader, optimizer, criterion, device, scaler, scheduler=None):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc='train', leave=False):
        labels = batch['labels'].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = _forward(model, batch, device)
        loss = criterion(logits.float(), labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def predict(model, loader, device):
    """Return (labels, prob_ai) in loader order."""
    model.eval()
    ys, probs = [], []
    for batch in tqdm(loader, desc='eval', leave=False):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
            logits = _forward(model, batch, device)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        ys.append(batch['labels'])
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()
