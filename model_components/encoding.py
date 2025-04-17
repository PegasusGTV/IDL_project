import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_sma(prices, window=14):
    """Compute Simple Moving Average (SMA)"""
    return F.avg_pool1d(prices.unsqueeze(1), window, stride=1, padding=window-1).squeeze(1)

def compute_ema(prices, window=14):
    """Compute Exponential Moving Average (EMA)"""
    alpha = 2 / (window + 1)
    ema = prices.clone()
    for t in range(1, prices.size(0)):
        ema[t] = alpha * prices[t] + (1 - alpha) * ema[t-1]
    return ema

def compute_rsi(prices, window=14):
    """Compute Relative Strength Index (RSI)"""
    delta = prices[1:] - prices[:-1]
    gain = F.relu(delta)
    loss = F.relu(-delta)
    
    avg_gain = compute_sma(gain, window)
    avg_loss = compute_sma(loss, window)
    
    rs = avg_gain / (avg_loss + 1e-7)
    rsi = 100 - (100 / (1 + rs))
    return F.pad(rsi, (1, 0), value=0)

def compute_velocity(prices):
    return torch.diff(prices, dim=-1)

def compute_acceleration(velocity):
    return torch.diff(velocity, dim=-1)



class Time2VecTorch(nn.Module):
    def __init__(self, num_frequency: int):
        """
        Time2Vec layer as described in Kazemi et al. (2019).
        
        Args:
            num_frequency (int): Number of periodic (sinusoidal) components.
                                 Final output will be (1 + num_frequency) features.
        """
        super(Time2VecTorch, self).__init__()
        self.num_frequency = num_frequency

        # Linear (trend) component
        self.w0 = nn.Parameter(torch.randn(1))  # Scalar weight for linear trend
        self.b0 = nn.Parameter(torch.randn(1))  # Scalar bias for trend

        # Periodic (sinusoidal) components
        self.w = nn.Parameter(torch.randn(num_frequency))  # Frequencies
        self.b = nn.Parameter(torch.randn(num_frequency))  # Phases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input of shape [batch_size, time_steps, feature_dim],
                        where the last feature is normalized time in [0, 1].

        Returns:
            Tensor: Time2Vec embedding of shape [batch_size, time_steps, 1 + num_frequency]
        """
        t = x[:, :, -1:]  # Extract time feature: [B, T, 1]

        # Linear trend: w0 * t + b0 → [B, T, 1]
        trend = self.w0 * t + self.b0

        # Periodic terms: sin(w * t + b) using broadcasting → [B, T, num_frequency]
        periodic = torch.sin(t * self.w + self.b)  # [B, T, num_frequency]

        # Concatenate trend and periodic → [B, T, 1 + num_frequency]
        t_encoded = torch.cat([trend, periodic], dim=-1)

        return t_encoded
    


class StockEmbedding(nn.Module):
    def __init__(self, num_frequency, num_technical_indicators):
        super(StockEmbedding, self).__init__()
        self.time2vec = Time2VecTorch(num_frequency=num_frequency)
        
        # Positional Encoding (Optional, but can be useful if you are using a Transformer)
        self.positional_encoding = nn.Embedding(1000, 64)  # assuming you have 1000 time steps at most
        
        # Example: a simple way to include velocity, acceleration, and technical indicators
        self.velocity_layer = nn.Linear(1, 32)  # Assuming velocity is a scalar (rate of change)
        self.acceleration_layer = nn.Linear(1, 32)  # Acceleration as a scalar
        self.technical_indicators_layer = nn.Linear(num_technical_indicators, 64)
        
        self.final_layer = nn.Linear(64 + 1 + num_frequency + 64 + 64, 128)  # For example
    
    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, technical_indicators: torch.Tensor) -> torch.Tensor:
        # Time2Vec encoding for time-related features
        time_embedding = self.time2vec(x)
        
        # Optional: positional encoding to capture the order of the sequence (important in transformers)
        position_embedding = self.positional_encoding(time_steps)
        
        # Velocity and Acceleration features (assuming they are precomputed)
        velocity = compute_velocity(x)
        acceleration = compute_acceleration(velocity)
        velocity_emb = F.relu(self.velocity_layer(velocity.unsqueeze(-1)))  # Replace with actual velocity computation
        acceleration_emb = F.relu(self.acceleration_layer(acceleration.unsqueeze(-1)))  # Replace with actual acceleration computation

        sma = compute_sma(x)
        ema = compute_ema(x)
        rsi = compute_rsi(x)

        technical_indicators = torch.stack([sma, ema, rsi], dim=-1)
        
        technical_indicators_embedding = self.technical_indicators_layer(technical_indicators)
        
        # Concatenate time, positional, velocity, acceleration, and technical indicators
        x_embedding = torch.cat([time_embedding, position_embedding, velocity_emb, acceleration_emb, technical_indicators_embedding], dim=-1)
        
        # Final embedding layer
        final_embedding = self.final_layer(x_embedding)
        
        return final_embedding



