import torch
import torch.nn as nn


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
