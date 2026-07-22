import torch 
import torch.nn as nn
from transformers import AutoModel

class AIContentModel(nn.Module):
    """
    BamiBert Encoder + Classification Head.
    """
    def __init__(self, model_name='Qualcomm-AI-Research/BamiBERT', num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size # 768 for base 
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] token represents for first token
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        logits = self.classifier(x)

        return logits