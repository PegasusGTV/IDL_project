import torch
import torch.nn as nn
import math

class TimeSeriesPositionalEncoding(nn.Module):
    """
    Positional Encoding for Time Series Data using sin/cos functions.

    Injects time-dependent features to the input embeddings to help 
    the Transformer understand the order of time steps.

    Args:
        d_model (int): Dimension of the input embeddings.
        max_len (int): Maximum sequence length (time steps).
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._create_pe_table(d_model, max_len)

    def _create_pe_table(self, d_model, max_len):
        # Initialize positional encoding table
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input of shape (B, T, d_model)
        Returns:
            torch.Tensor: Input with positional encodings added
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Input sequence length {seq_len} exceeds max_len {self.pe.size(1)}")
        
        return x + self.pe[:, :seq_len, :]
