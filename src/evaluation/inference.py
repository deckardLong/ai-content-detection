import torch
import numpy as np
from src.preprocessing import cleaner

def predict(model, data_loader, device):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )

def predict_single_text(model, tokenizer, text, device, max_length=512, clean=True):
    if clean:
        processed_text = cleaner.clean_text(text)
    else:
        processed_text = text
 
    model.eval()
    encoding = tokenizer(
        processed_text, max_length=max_length, truncation=True,
        padding='max_length', return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
 
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = torch.argmax(probs).item()
 
    return {
        'cleaned_text': processed_text,
        'predicted_label': pred_label,
        'predicted_class': 'AI' if pred_label == 1 else 'Human',
        'prob_human': probs[0].item(),
        'prob_ai': probs[1].item(),
    }