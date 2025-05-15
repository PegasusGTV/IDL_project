import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm


class SyntheticTimeSeriesDataset(Dataset):
    def __init__(
        self,
        num_sequences: int = 1000,
        window_size: int = 30,
        forecast_horizon: int = 1,
        features: List[str] = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'time'],
        target: str = 'feature5',
        normalize: Optional[str] = 'minmax',
        split: Optional[str] = None,
        val_ratio: float = 0.2,
        shift: int = 2,
        fit_scaler: bool = True,
        noise_std: float = 0.02,
        num_frequencies: int = 3,
        tgt_scaler: Optional[object] = None, 
        cls_scaler: Optional[object] = None, # <-- new
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
        self.num_feats = len(features)
        self.target = target
        self.normalize = normalize
        self.split = split
        self.shift = shift
        self.fit_scaler = fit_scaler
        self.noise_std = noise_std
        self.num_frequencies = num_frequencies

        self.tgt_scaler = tgt_scaler
        self.cls_scaler = cls_scaler

        sequences = []
        labels_shifted = []
        labels_shifted_targfeat = []
        labels_golden = []

        # Generate time points
        total_length = num_sequences * shift + window_size + forecast_horizon
        raw_time = np.linspace(0, 10, total_length)
        time = (raw_time - raw_time.min()) / (raw_time.max() - raw_time.min())  # Normalized 0-1

        total_length = num_sequences * shift + window_size + forecast_horizon
        t = np.linspace(0, 1, total_length)

        # Generate linear features: f_i(t) = a_i * t + b_i
        np.random.seed(42)  # Reproducible
        coefficients = [np.random.uniform(1.0, 5.0, size=2) for _ in range(self.num_feats - 1)]
        # completely linear data
        features_linear = [a * t + b for a, b in coefficients]

        # Time feature stays normalized [0,1]
        features_linear.append(t)

        full_features = np.stack(features_linear, axis=1)  # shape: [T, 6]

        # Split data
        train_end_idx = int(len(full_features) * (1 - val_ratio))
        if split == 'train':
            full_features = full_features[:train_end_idx]
        elif split == 'val':
            full_features = full_features[train_end_idx:]

        # Normalization (excluding time column)
        if normalize:
            scaler_cls = StandardScaler if normalize == 'zscore' else MinMaxScaler
            scaler_tgt = StandardScaler if normalize == 'zscore' else MinMaxScaler

            if fit_scaler:
                self.tgt_scaler = scaler_tgt()
                self.cls_scaler = scaler_cls()
                self.cls_scaler.fit(full_features[:, :-2])  # Don't scale time column and the second last column (target)
                self.tgt_scaler.fit(full_features[:, -2:-1])  # Scale only the target feature

            if self.tgt_scaler is not None:
                full_features[:, :-2] = self.cls_scaler.transform(full_features[:, :-2])  # Keep time 0-1
                full_features[:, -2:-1] = self.tgt_scaler.transform(full_features[:, -2:-1])  # Scale only the target feature

        # Create sequences
        values = full_features
        target_idx = features.index(target)
        
        # Calculate valid window positions
        total_windows = (len(full_features) - window_size - forecast_horizon) // shift + 1
        start_indices = np.arange(0, total_windows * shift, shift)
        
        for i in tqdm(start_indices):
            x = values[i:i + window_size]  # Input sequence (past window)
            # print(f"x is {x.shape}")

            # # Create shifted and golden targets
            if i == 0:
                # For the first window, there is no previous input, so replicate the first row
                targets_shifted = np.vstack((x[0], x[:-1]))  # [window_size, num_features]
            else:
                # Use the last row of the previous window as the first row of the shifted targets
                targets_shifted = np.vstack((values[i - shift + window_size - 1], x[:-1]))  # [window_size, num_features]

            # print(f"targets_shifted is {targets_shifted.shape}")
            targets_shifted_targfeat = targets_shifted[:, target_idx:target_idx + 1]  # [window_size, 1]
            targets_golden = x[:, target_idx]  # [window_size]
            

            # Convert to PyTorch tensors (no flattening anymore!)
            targets_shifted = torch.tensor(targets_shifted, dtype=torch.float32)                # [window_size, num_features]
            targets_shifted_targfeat = torch.tensor(targets_shifted_targfeat, dtype=torch.float32)  # [window_size, 1]
            targets_golden = torch.tensor(targets_golden, dtype=torch.float32).unsqueeze(-1)       # [window_size, 1]
            # print(f"targets_golden is {targets_golden.shape}")
            
            sequences.append(torch.tensor(x, dtype=torch.float32))  # [window_size, num_features]
            labels_shifted.append(targets_shifted)                  # [window_size, num_features]
            labels_shifted_targfeat.append(targets_shifted_targfeat)  # [window_size, 1]
            labels_golden.append(targets_golden)                       # [window_size, 1]



        self.data = torch.stack(sequences)                        # [N, window_size, input_features]
        self.targets_shifted = torch.stack(labels_shifted)        # [N, window_size, input_features]
        self.targets_shifted_targfeat = torch.stack(labels_shifted_targfeat)  # [N, window_size, 1]
        self.targets = torch.stack(labels_golden)                 # [N, window_size, 1]
        self.input_features = self.data.shape[-1]
        self.output_features = 1  # still predicting only one feature TODO un-hard code

        print(f"Data shape: {self.data.shape}")  # [num_samples, window_size, 6]
        print(f"Targets shape: {self.targets.shape}")  # [num_samples, forecast_horizon, 1]
        print(f"Shifted targets shape: {self.targets_shifted.shape}")
        print(f"Shifted targets (target feature) shape: {self.targets_shifted_targfeat.shape}")
        self._visualize_sample()

    def _visualize_sample(self):
        """Plot feature relationships for first sample"""
        plt.figure(figsize=(15, 10))
        sample_idx = 0
        
        # Plot all features
        for i in range(self.num_feats):  # include time dimension
            plt.subplot(3, 2, i+1)
            plt.plot(self.data[sample_idx][:, i], label=f'Feature {i+1}')
            plt.legend()
        
        plt.tight_layout()
        plt.suptitle("Synthetic Features Visualization (First Sample)", y=1.02)
        plt.show()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets_shifted[idx], self.targets_shifted_targfeat[idx], self.targets[idx]
