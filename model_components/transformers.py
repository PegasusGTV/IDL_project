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
        self.input_features = input_features
        self.d_model = d_model
        # Final input dim = projected features + time2vec
        self.combined_dim = d_model*(input_features-1) + d_freq + 1
        self.tgt_combined_dim = d_model + d_freq + 1

        # for projecting all the features to a higher dimension
        # self.input_proj = nn.Sequential(
        #     nn.Linear(input_features - 1, d_model),  # First layer
        #     nn.GELU(),                              # Activation
        #     nn.Dropout(dropout),                    # Dropout
        #     nn.Linear(d_model, d_model),            # Second layer
        #     nn.GELU(),                              # Activation
        #     nn.Dropout(dropout)                     # Dropout
        # )
        # self.input_proj_norm = nn.LayerNorm(d_model)
        # self.input_act = nn.GELU()
        self.ips = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_model),  # For each feature
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
            for _ in range(input_features - 1)
        ])
        self.final_ip = nn.Linear(self.combined_dim, self.tgt_combined_dim)


        self.tgt_embedding = nn.Linear(1, d_model)  # For univariate shifted targets

        # Time2Vec module: outputs [batch, seq_len, d_freq + 1]
        self.time2vec = Time2VecTorch(num_frequency=d_freq)

        self.dropout = nn.Dropout(dropout)
        

        # positional_encoding
        self.positional_encoding = TimeSeriesPositionalEncoding(self.combined_dim, max_len=2000)
        self.tgt_pe = TimeSeriesPositionalEncoding(self.tgt_combined_dim, max_len=500)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.tgt_combined_dim, 
            nhead=num_heads, 
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # Important: ensures input is (batch, seq, feature)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(self.tgt_combined_dim)
        # self.output_proj = nn.Linear(self.combined_dim, 1)
        self.scale_factor = nn.Parameter(torch.tensor(0.3))  # Start at 1.0, but it will be learned
        self.output_proj = nn.Sequential(
            nn.Linear(self.tgt_combined_dim, 2*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, memory: torch.Tensor, tgt_shifted: torch.Tensor) -> torch.Tensor:
        """
        Decoder-only Transformer for forecasting univariate time series using past multivariate data.

        Args:
            memory: Input context, shape [B, T, F]
            tgt_shifted: Shifted target, shape [B, T, 1]

        Returns:
            output: Predicted target values, shape [B, T, 1]
        """
        B, T, _ = memory.shape

        # Embed memory (multivariate input)
        memory_feat = memory[:, :, :-1]
        memory_time = memory[:, :, -1:]

        # memory_proj = self.input_proj(memory_feat)         # [B, T, d_model]
        # memory_proj = self.input_proj_norm(memory_proj)  # Normalize projected features
        memory_proj = torch.zeros(B, T, (self.input_features-1)*self.d_model, device=memory.device, dtype=memory.dtype)  # [B, T, input_features-1]

        for i in range(self.input_features-1):
            memory_proj[:, :, i*self.d_model : (i+1)*self.d_model] = self.ips[i](memory_feat[:, :, i].unsqueeze(-1))
        
        memory_time2vec = self.time2vec(memory)            # [B, T, d_freq + 1]
        memory_combined = torch.cat([memory_proj, memory_time2vec], dim=-1)  # [B, T, combined_dim]
        memory_combined = self.positional_encoding(memory_combined)
        memory_combined = self.final_ip(memory_combined)  # [B, T, d_model]        
        memory_combined = self.dropout(memory_combined)

        # Embed target (univariate shifted target)
        tgt_embed = self.tgt_embedding(tgt_shifted)        # [B, T, d_model]
        tgt_time2vec = self.time2vec(torch.cat([tgt_shifted, memory_time], dim=-1))  # Add time info to tgt
        tgt_combined = torch.cat([tgt_embed, tgt_time2vec], dim=-1)  # [B, T, combined_dim]
        tgt_combined = self.tgt_pe(tgt_combined)
        tgt_combined = self.dropout(tgt_combined)

        # Causal mask for autoregressive decoding
        causal_mask = CausalMask(tgt_combined).to(dtype=torch.float32, device=memory.device)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf')).masked_fill(causal_mask == 0, float(0.0))

        # Decode
        print(f"shape of tgt_combined: {tgt_combined.shape}, shape of memory_combined: {memory_combined.shape}")
        decoded = self.decoder(tgt=tgt_combined, memory=memory_combined, tgt_mask=causal_mask)

        # Output projection
        decoded = self.norm(decoded)
        output = self.output_proj(decoded)  # [B, T, 1]
        
        # * self.scale_factor
        return output 