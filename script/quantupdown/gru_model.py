import torch
import torch.nn as nn


class GRUQuantileModel(nn.Module):
    """
    输入: (B, T, F)
    输出: (B, 2) -> [p_up90, p_dn10]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, h = self.gru(x)          # h: (num_layers, B, hidden_dim)
        last_h = h[-1]                # (B, hidden_dim)
        logits = self.head(last_h)    # (B, 2)
        return logits