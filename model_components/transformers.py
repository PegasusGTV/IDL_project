import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import TimeSeriesPositionalEncoding
from .masks import PadMask, CausalMask


class TimeSeriesTransformer(nn.Module):
    """
    Transformer for time series forecasting with raw time feature (no Time2Vec).

    Args:
        input_features: Number of input features per time step
        output_features: Number of output features to predict
        num_layers: Number of transformer decoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        forecast_horizon: Number of future time steps to predict
    """
    def __init__(
        self,
        input_features: int,
        output_features: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        d_freq: int,  # unused but kept for API compatibility
        dropout: float,
        forecast_horizon: int
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_features = input_features
        self.d_model = d_model

        # Combined input dim: (F-1) features + 1 time + 1 dummy pad
        self.combined_dim = d_model * (input_features - 1) + 1 + 1
        self.tgt_combined_dim = d_model + 1 + 1

        self.ips = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
            for _ in range(input_features - 1)
        ])
        self.final_ip = nn.Linear(self.combined_dim, self.tgt_combined_dim)
        self.tgt_embedding = nn.Linear(1, d_model)

        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = TimeSeriesPositionalEncoding(self.combined_dim, max_len=2000)
        self.tgt_pe = TimeSeriesPositionalEncoding(self.tgt_combined_dim, max_len=500)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.tgt_combined_dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.tgt_combined_dim)

        self.scale_factor = nn.Parameter(torch.tensor(0.3))
        self.output_proj = nn.Sequential(
            nn.Linear(self.tgt_combined_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            # nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, memory: torch.Tensor, tgt_shifted: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory: Input context, shape [B, T, F]
            tgt_shifted: Shifted target, shape [B, T, 1]

        Returns:
            output: Predicted target values, shape [B, T, 1]
        """
        B, T, _ = memory.shape
        memory_feat = memory[:, :, :-1]   # [B, T, F-1]
        memory_time = memory[:, :, -1:]   # [B, T, 1]

        memory_proj = torch.zeros(B, T, (self.input_features - 1) * self.d_model,
                                  device=memory.device, dtype=memory.dtype)
        for i in range(self.input_features - 1):
            memory_proj[:, :, i * self.d_model: (i + 1) * self.d_model] = \
                self.ips[i](memory_feat[:, :, i].unsqueeze(-1))

        memory_combined = torch.cat([memory_proj, memory_time], dim=-1)  # [B, T, ...]
        memory_combined = F.pad(memory_combined, (0, 1))  # Pad with 1 dummy feature
        # memory_combined = self.positional_encoding(memory_combined)
        memory_combined = self.final_ip(memory_combined)
        memory_combined = self.dropout(memory_combined)

        tgt_embed = self.tgt_embedding(tgt_shifted)  # [B, T, d_model]
        tgt_combined = torch.cat([tgt_embed, memory_time], dim=-1)
        tgt_combined = F.pad(tgt_combined, (0, 1))
        # tgt_combined = self.tgt_pe(tgt_combined)
        tgt_combined = self.dropout(tgt_combined)

        causal_mask = CausalMask(tgt_combined).to(dtype=torch.float32, device=memory.device)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).masked_fill(causal_mask == 0, float(0.0))

        decoded = self.decoder(tgt=tgt_combined, memory=memory_combined, tgt_mask=causal_mask)
        decoded = self.norm(decoded)
        output = self.output_proj(decoded)  # [B, T, 1]
        return output
