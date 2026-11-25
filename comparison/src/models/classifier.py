# Merge encoder with head

import torch.nn as nn
from .encoders import load_encoder
from .heads.cnn_head import CNNHead
from .heads.bilstm_head import BiLSTMHead
from .heads.cnn_bilstm_head import CNNBiLSTMHead

HEADS = {"cnn": CNNHead, "bilstm": BiLSTMHead, "cnn_bilstm": CNNBiLSTMHead}

class EncoderClassifier(nn.Module):
    """Encoder (PhoBERT/BamiBERT) + DL head.

    encoder_name=None -> head only, takes pre-extracted features (frozen mode).
    """
    def __init__(self, encoder_name, head_type, head_cfg, num_classes, dropout, hidden_size=768):
        super().__init__()
        self.encoder = load_encoder(encoder_name) if encoder_name else None
        if self.encoder is not None:
            hidden_size = self.encoder.config.hidden_size

        if head_type not in HEADS:
            raise ValueError(f"Unknown head: {head_type}")
        self.head = HEADS[head_type](hidden_size, num_classes, dropout=dropout, **head_cfg)

    def forward(self, attention_mask, input_ids=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(hidden_states, attention_mask)
