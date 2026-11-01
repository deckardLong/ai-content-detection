import torch
import torch.nn as nn

class BiLSTMHead(nn.Module):
    def __init__(self, hidden_size, num_classes, lstm_hidden, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_size, lstm_hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, hidden_states, attention_mask):
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden_states, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)   # last layer: forward + backward
        return self.fc(self.dropout(h))
