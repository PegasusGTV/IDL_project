import torch
import torch.nn as nn
import math

class Time2Vec(nn.Module):
    """
    Time2Vec (Kazemi & Mehdad, 2020)
    t  : [..., 1]  scalar notion of time
    out: [..., k]  [periodic_1 … periodic_k , linear]
    """
    def __init__(self, k: int, periodic_activation=torch.sin):
        super().__init__()
        self.k = k
        self.f = periodic_activation
        # weights/bias for k - 1 periodic components
        self.w  = nn.Parameter(torch.randn(1, 1, k - 1))   # shape broadcastable to [..., k - 1]
        self.b  = nn.Parameter(torch.randn(1, 1, k - 1))
        # weights/bias for linear component
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t expected as [..., 1]
        periodic = self.f(t * self.w + self.b)          # [..., k - 1]
        linear   = t * self.w0 + self.b0                # [..., 1]
        return torch.cat([periodic, linear], dim=-1)    # [..., k]
