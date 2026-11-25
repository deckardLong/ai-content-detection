import torch
import torch.nn as nn

class CNNBiLSTMHead(nn.Module):
    """CNN and BiLSTM branches in parallel over the same encoder features.

    CNN branch   : 1D convs (kernel sizes k) + mask-aware max-over-time  -> num_filters * len(k)
    BiLSTM branch: packed BiLSTM, concat of last forward/backward states -> 2 * lstm_hidden
    The two vectors are concatenated, then Dropout -> Linear -> logits (one joint classifier).
    """
    def __init__(self, hidden_size, num_classes, num_filters, kernel_sizes, lstm_hidden,
                 num_layers, dropout):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, num_filters, k) for k in self.kernel_sizes
        ])
        self.lstm = nn.LSTM(
            hidden_size, lstm_hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes) + 2 * lstm_hidden, num_classes)

    def forward(self, hidden_states, attention_mask):
        # --- CNN branch (same as CNNHead)
        mask = attention_mask.unsqueeze(1).to(hidden_states.dtype)    # (B, 1, L)
        x = hidden_states.transpose(1, 2) * mask                      # zero out padding
        pooled = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            c = torch.relu(conv(x))
            pooled.append((c * mask[:, :, k - 1:]).max(dim=2).values)
        h_cnn = torch.cat(pooled, dim=1)

        # --- BiLSTM branch (same as BiLSTMHead)
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden_states, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_lstm = torch.cat([h_n[-2], h_n[-1]], dim=1)

        return self.fc(self.dropout(torch.cat([h_cnn, h_lstm], dim=1)))
