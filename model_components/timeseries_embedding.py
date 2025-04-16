import torch
import numpy as np

class Time2VecTorch(torch.nn.Module):
    """PyTorch implementation of Time2Vec embedding"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear = torch.nn.Linear(1, 1)
        self.periodic = torch.nn.Linear(1, d_model-1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x_mean = x[:, :, :4].mean(dim=-1, keepdim=True)  # Average OHLC features
        time_linear = self.linear(x_mean)
        time_periodic = torch.sin(self.periodic(x_mean))
        return torch.cat([time_linear, time_periodic], dim=-1)