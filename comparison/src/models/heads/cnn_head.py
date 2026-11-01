import torch
import torch.nn as nn

# Define CNN head
class CNNHead(nn.Module):
    def __init__(self, hidden_size, num_classes, num_filters, kernel_sizes, dropout):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, num_filters, k) for k in self.kernel_sizes    # 1D conv over tokens
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), num_classes)

    def forward(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(1).to(hidden_states.dtype)    # (B, 1, L)
        x = hidden_states.transpose(1, 2) * mask                      # zero out padding
        pooled = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            c = torch.relu(conv(x))                                   # (B, F, L-k+1)
            # Window t is valid only if its last token (t+k-1) is not padding.
            # relu >= 0, so multiplying by 0 excludes padded windows from max-pool.
            pooled.append((c * mask[:, :, k - 1:]).max(dim=2).values)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
