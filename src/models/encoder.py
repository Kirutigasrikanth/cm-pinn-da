import torch
import torch.nn as nn

class CycleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, K, input_dim]
        z = self.input_proj(x)         # [B,K,H]
        z = self.encoder(z)            # [B,K,H]
        z = z.mean(dim=1)              # [B,H] (simple average pooling)
        return z

