import torch.nn as nn


class UniversalFeatureAdapter(nn.Module):
    def __init__(self, in_dim: int, tree_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(int(in_dim), int(tree_dim))
        self.norm = nn.LayerNorm(int(tree_dim))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, h):
        return self.drop(self.norm(self.proj(h)))
