import torch
import torch.nn as nn
from positional_encoding import TimeSeriesPositionalEncoding
from timeseries_embedding import Time2VecTorch
from .masks import CausalMask

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
        max_len: int = 5000,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_features, d_model)
        # this will lift the time dimension to d_freq +1  dimensions. 
        # so, the total number of dims for the embedding will be d_model + d_freq +1
        self.time2vec = Time2VecTorch(num_frequency=d_freq)   
        self.pos_enc = TimeSeriesPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.decoder_layers = nn.ModuleList([
            SelfAttentionDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_features)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_features)
        
        Returns:
            output: Predictions of shape (batch_size, seq_len, output_features)
        """
        # Input projection - everything except time
        features = x[:, :, :-1]     
        x_embed = self.input_proj(features)  # (B, T, d_model)
        
        # Add Time2Vec temporal embeddings
        print('Input tensor shape:', x.shape)
        t2v = self.time2vec(x)        # (B, T, d_model)
        print('Output tensor shape:', x_out.shape)
        # x_embed += t2v
        torch.cat([features, t_vec], dim=-1)
        
        # Add positional encoding
        x_embed = self.pos_enc(x_embed)
        x_embed = self.dropout(x_embed)

        # Create causal mask
        causal_mask = CausalMask(x_embed)

        # Process through decoder layers
        for layer in self.decoder_layers:
            x_embed, _ = layer(x_embed, attn_mask=causal_mask)

        # Final projection
        x_embed = self.norm(x_embed)
        output = self.output_proj(x_embed)
        
        return output

    def predict(self, x: torch.Tensor, forecast_steps: int) -> torch.Tensor:
        """
        Autoregressive prediction for time series forecasting
        
        Args:
            x: Initial sequence (B, T, input_features)
            forecast_steps: Number of steps to predict ahead
            
        Returns:
            predictions: Forecasted values (B, forecast_steps, output_features)
        """
        predictions = []
        current_seq = x
        
        for _ in range(forecast_steps):
            # Get next prediction
            output = self(current_seq)[:, -1:, :]  # (B, 1, output_features)
            predictions.append(output)
            
            # Update sequence with new prediction
            current_seq = torch.cat([current_seq, output], dim=1)
            
        return torch.cat(predictions, dim=1)
