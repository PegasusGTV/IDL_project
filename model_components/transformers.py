import torch
import torch.nn as nn
from .positional_encoding import TimeSeriesPositionalEncoding
from .timeseries_embedding import Time2VecTorch
from .masks import PadMask, CausalMask

class TimeSeriesTransformer(nn.Module):
    """
    Transformer for time series forecasting with Time2Vec embeddings and positional encoding.
    
    Args:
        input_features: Number of input features per time step
        output_features: Number of output features to predict
        num_layers: Number of transformer decoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        max_len: Maximum sequence length
    """
    def __init__(
        self,
        input_features: int,
        output_features: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        d_freq: int,
        dropout: float,
        forecast_horizon: int
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        # for projecting all the features to a higher dimension
        self.input_proj = nn.Linear(input_features-1, d_model)
        
        # Time2Vec module: outputs [batch, seq_len, d_freq + 1]
        self.time2vec = Time2VecTorch(num_frequency=d_freq)

        # Final input dim = projected features + time2vec
        self.combined_dim = d_model + d_freq + 1

        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.combined_dim, 
            nhead=num_heads, 
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # Important: ensures input is (batch, seq, feature)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)


        self.norm = nn.LayerNorm(self.combined_dim)
        self.output_proj = nn.Linear(self.combined_dim, output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        CURRENTLY THIS VERSION SUPPORTS UNIVARIATE PREDICTION OF CLOSE PRICES ONLY.
        WE CAN LATER EXPAND THIS TO BE AUTOREGRESSIVE OR HAVE MULTIVARIATE PREDICTIONS
        Args:
            x: Input tensor of shape (batch_size, context_len, input_features)
            forecast_horizon: Number of future steps to forecast

        Returns:
            output: Predictions of shape (batch_size, forecast_horizon, output_features)
        """
        batch_size, context_len, input_features = x.shape

        # Split input into features and time
        x_feat = x[:, :, :-1]  # [B, T, input_features - 1]
        x_time = x[:, :, -1:]  # [B, T, 1]

        # Project input features
        x_proj = self.input_proj(x_feat)  # [B, T, d_model]
        x_time2vec = self.time2vec(x)     # [B, T, d_freq + 1]
        x_combined = torch.cat([x_proj, x_time2vec], dim=-1)  # [B, T, combined_dim]
        x_combined = self.dropout(x_combined)

        # Prepare tgt inputs (zeros, but add time info and Time2Vec)
        tgt_time = x_time[:, -1:] + torch.arange(
            1, self.forecast_horizon + 1, device=x.device
        ).reshape(1, -1, 1)  # [B, forecast_horizon, 1]

        # Repeat last feature vector or use zeros for target initialization
        tgt_init = torch.zeros(batch_size, self.forecast_horizon, x_feat.shape[-1], device=x.device)

        tgt_full = torch.cat([tgt_init, tgt_time], dim=-1)  # [B, forecast_horizon, input_features]
        tgt_proj = self.input_proj(tgt_full[:, :, :-1])     # [B, H, d_model]
        tgt_time2vec = self.time2vec(tgt_full)              # [B, H, d_freq + 1]
        tgt_combined = torch.cat([tgt_proj, tgt_time2vec], dim=-1)  # [B, H, combined_dim]
        tgt_combined = self.dropout(tgt_combined)

        # Causal mask for decoder
        causal_mask = CausalMask(tgt_combined).to(dtype=torch.float32)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).masked_fill(causal_mask == 0, float(0.0))

        # Decode
        decoded = self.decoder(tgt=tgt_combined, memory=x_combined, tgt_mask=causal_mask)

        # Project to output
        decoded = self.norm(decoded)
        output = self.output_proj(decoded)  # [B, forecast_horizon, output_features]

        # We only care about predicting the 'Close' value (1D for each future step)
        return output  # This selects the 'Close' column for each time step in the forecast

