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
        self.input_act = nn.GELU()
        self.input_proj_target = nn.Linear(1, d_model)  # for tgt_init: only close

        
        # Time2Vec module: outputs [batch, seq_len, d_freq + 1]
        self.time2vec = Time2VecTorch(num_frequency=d_freq)

        # Final input dim = projected features + time2vec
        self.combined_dim = d_model + d_freq + 1

        self.dropout = nn.Dropout(dropout)

        # positional_encoding
        self.positional_encoding = TimeSeriesPositionalEncoding(self.combined_dim, max_len=500)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.combined_dim, 
            nhead=num_heads, 
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # Important: ensures input is (batch, seq, feature)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(self.combined_dim)
        # self.output_proj = nn.Linear(self.combined_dim, 1)
        self.scale_factor = nn.Parameter(torch.tensor(100.0))  # Start at 1.0, but it will be learned
        self.output_proj = nn.Sequential(
            nn.Linear(self.combined_dim, 2*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

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
        # print(x.shape)

        # Split input into features and time
        x_feat = x[:, :, :-1]  # [B, T, input_features - 1]
        x_time = x[:, :, -1:]  # [B, T, 1]

        # Project input features
        x_proj = self.input_proj(x_feat)  # [B, T, d_model]
        x_proj = self.input_act(x_proj)
        x_time2vec = self.time2vec(x)     # [B, T, d_freq + 1]
        x_combined = torch.cat([x_proj, x_time2vec], dim=-1)  # [B, T, combined_dim]
        x_combined = self.positional_encoding(x_combined)
        x_combined = self.dropout(x_combined)
       

        # DEBUGGING THINGS
        assert torch.isfinite(x).all(), "Non-finite (NaN or Inf) in input"
        assert torch.isfinite(x_proj).all(), "Non-finite (NaN or Inf) after input_proj"
        assert torch.isfinite(x_time2vec).all(), "Non-finite (NaN or Inf) in Time2Vec"

        # print(x_combined.shape)
        # HARDCODED FEATURE INDICES, NEED TO FIX!!!
        # last_close = x_feat[:, -1:, 3:4]  # Shape: [B, 1, 1]
        # print(f" last close is {last_close}")
        last_feat = x_feat[:, -1:, :]  # Get all features from last time step [B, 1, input_features-1]
        tgt_init = last_feat.repeat(1, self.forecast_horizon, 1)  # [B, H, input_features-1]

        # Prepare tgt inputs (zeros, but add time info and Time2Vec)
        tgt_time = x_time[:, -1:] + torch.arange(
            1, self.forecast_horizon + 1, device=x.device
        ).reshape(1, -1, 1)  # [B, forecast_horizon, 1]
        # tgt_init = last_close.repeat(1, self.forecast_horizon, 1)  # [B, H, 1]
        # Option 1: Repeat last close
        # tgt_init = last_close.repeat(1, self.forecast_horizon, 1)

        # Option 2: Start with zeros
        # tgt_init = torch.zeros_like(last_close).repeat(1, self.forecast_horizon, 1)

        tgt_full = torch.cat([tgt_init, tgt_time], dim=-1)  # [B, H, input_features]
        tgt_full = tgt_full.to(x.device)  # Move tgt to the same device as x


        # Repeat last feature vector or use zeros for target initialization
        # tgt_full = torch.cat([tgt_features, tgt_close, tgt_time], dim=-1)  # [B, H, input_features]
        # print(f"target full shape is {tgt_full.shape}")

        tgt_proj = self.input_proj(tgt_init)  # [B, H, d_model]
        tgt_time2vec = self.time2vec(tgt_full)              # [B, H, d_freq + 1]
        tgt_combined = torch.cat([tgt_proj, tgt_time2vec], dim=-1)  # [B, H, combined_dim]
        tgt_combined = self.positional_encoding(tgt_combined)
        tgt_combined = self.dropout(tgt_combined)
       
        # print(f"tgt combined shape is {tgt_combined.shape}")

        # Causal mask for decoder
        causal_mask = CausalMask(tgt_combined).to(dtype=torch.float32, device=x.device)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).masked_fill(causal_mask == 0, float(0.0))
        causal_mask = causal_mask.to(tgt_combined.device)

        # Decode
        decoded = self.decoder(tgt=tgt_combined, memory=x_combined, tgt_mask=causal_mask)
        # print(decoded.shape)

        # Project to output
        decoded = self.norm(decoded)
        output = self.output_proj(decoded)  # [B, forecast_horizon, output_features]
        output = output * self.scale_factor
        # We only care about predicting the 'Close' value (1D for each future step)
        return output  # This selects the 'Close' column for each time step in the forecast

