import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class SyntheticTimeSeriesDataset(Dataset):
    def __init__(
        self,
        num_sequences: int = 1000,
        window_size: int = 30,
        forecast_horizon: int = 1,
        features: List[str] = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'time'],
        target: str = 'feature1',
        normalize: Optional[str] = 'minmax',
        split: Optional[str] = None,
        val_ratio: float = 0.2,
        shift: int = 2,
        scaler: Optional[object] = None,
        fit_scaler: bool = False,
        noise_std: float = 0.02,
        num_frequencies: int = 3
    ):
        """
        Synthetic dataset with 5 engineered features + time dimension.
        Feature relationships:
        - feature1: Primary sine wave (target)
        - feature2: Lagged version of feature1
        - feature3: Sine wave with phase shift
        - feature4: Amplitude-modulated version
        - feature5: Frequency-modulated version
        - time: Normalized time dimension
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.features = features
        self.target = target
        self.normalize = normalize
        self.split = split
        self.shift = shift
        self.scaler = scaler
        self.fit_scaler = fit_scaler
        self.noise_std = noise_std
        self.num_frequencies = num_frequencies

        # Generate time points
        total_length = num_sequences * shift + window_size + forecast_horizon
        raw_time = np.linspace(0, 10, total_length)
        time = (raw_time - raw_time.min()) / (raw_time.max() - raw_time.min())  # Normalized 0-1

        # Base signal with multiple frequencies
        base_signal = np.zeros(total_length)
        for freq in np.linspace(0.5, 2.0, num_frequencies):
            base_signal += np.sin(2 * np.pi * freq * raw_time)
        
        # Create engineered features
        feature1 = base_signal  # Target feature
        feature2 = np.roll(feature1, 3)  # Lagged version
        feature3 = np.sin(2 * np.pi * 1.0 * raw_time + np.pi/4)  # Phase shift
        feature4 = feature1 * (1 + 0.5*np.sin(2*np.pi*0.1*raw_time))  # Amplitude modulation
        feature5 = np.sin(2 * np.pi * (0.5 + 0.1*raw_time) * raw_time)  # Frequency modulation

        # Combine features with noise
        full_features = np.column_stack([
            feature1 + np.random.normal(0, noise_std, total_length),
            feature2 + np.random.normal(0, noise_std, total_length),
            feature3 + np.random.normal(0, noise_std, total_length),
            feature4 + np.random.normal(0, noise_std, total_length),
            feature5 + np.random.normal(0, noise_std, total_length),
            time  # Normalized time dimension
        ])

        # Split data
        train_end_idx = int(len(full_features) * (1 - val_ratio))
        if split == 'train':
            full_features = full_features[:train_end_idx]
        elif split == 'val':
            full_features = full_features[train_end_idx:]

        # Normalization (excluding time column)
        if normalize:
            scaler_cls = MinMaxScaler if normalize == 'minmax' else StandardScaler
            if fit_scaler:
                self.scaler = scaler_cls()
                self.scaler.fit(full_features[:, :-1])  # Don't scale time column
            
            if self.scaler is not None:
                full_features[:, :-1] = self.scaler.transform(full_features[:, :-1])  # Keep time 0-1

        # Create sequences
        sequences = []
        labels = []
        target_idx = features.index(target)
        
        for i in range(0, len(full_features) - window_size - forecast_horizon + 1, shift):
            window = full_features[i:i + window_size]
            label = full_features[i + window_size:i + window_size + forecast_horizon, target_idx]
            
            sequences.append(torch.tensor(window, dtype=torch.float32))
            labels.append(torch.tensor(label, dtype=torch.float32).unsqueeze(-1))

        self.data = torch.stack(sequences)
        self.targets = torch.stack(labels)
        self.input_features = len(features)
        self.output_features = 1

        print(f"Data shape: {self.data.shape}")  # [num_samples, window_size, 6]
        print(f"Targets shape: {self.targets.shape}")  # [num_samples, forecast_horizon, 1]
        self._visualize_sample()

    def _visualize_sample(self):
        """Plot feature relationships for first sample"""
        plt.figure(figsize=(15, 10))
        sample_idx = 0
        
        # Plot all features
        for i in range(5):  # Exclude time dimension
            plt.subplot(3, 2, i+1)
            plt.plot(self.data[sample_idx][:, i], label=f'Feature {i+1}')
            if i == 0:  # Highlight targets
                plt.scatter(
                    range(self.window_size, self.window_size + self.forecast_horizon),
                    self.targets[sample_idx][:, 0],
                    color='red',
                    marker='x',
                    label='Targets'
                )
            plt.legend()
        
        # Plot time feature
        plt.subplot(3, 2, 6)
        plt.plot(self.data[sample_idx][:, -1], label='Time', color='black')
        plt.legend()
        
        plt.tight_layout()
        plt.suptitle("Synthetic Features Visualization (First Sample)", y=1.02)
        plt.show()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
