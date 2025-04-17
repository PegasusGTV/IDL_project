import yfinance as yf
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm


class FinancialTimeSeriesDataset(Dataset):
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        features: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume'],
        window_size: int = 30,
        forecast_horizon: int = 1,
        target: str = 'Close',
        normalize: Optional[str] = 'zscore',
        split: Optional[str] = None,  # 'train' or 'val' or None (use full set)
        val_ratio: float = 0.2
    ):
        """
        Dataset for time series forecasting from Yahoo Finance.

        Args:
            tickers (List[str]): Stock tickers.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            features (List[str]): Input features.
            window_size (int): History window size (past Y days).
            forecast_horizon (int): Number of days to predict (future X days).
            target (str): Feature to forecast.
            normalize (Optional[str]): 'zscore', 'minmax', or None.
            split (Optional[str]): 'train' or 'val' or None (for full dataset).
            val_ratio (float): Fraction for validation if split is used.
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.features = features
        self.target = target
        self.normalize = normalize

        self.data = []
        self.targets = []

        self.split = split

        all_sequences = []
        all_labels = []

        for ticker in tickers:
            df = yf.download(ticker, start=start_date, end=end_date)[features]
            df = df.dropna()

            # Add normalized time feature
            df['Time'] = (df.index - df.index.min()).days
            df['Time'] = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())
            df = df[features + ['Time']]

            # Normalize
            if normalize == 'zscore':
                df = (df - df.mean()) / (df.std() + 1e-8)
            elif normalize == 'minmax':
                df = (df - df.min()) / (df.max() - df.min() + 1e-8)

            values = df.values
            target_idx = features.index(target)
            # CHANGE THIS SO THAT WE ARE PREDICTING THE ENTIRE FEATURE SEQ, NOT JUST THE CLOSING PRICE

            for i in tqdm(range(len(values) - window_size - forecast_horizon + 1), desc=f"Processing {ticker}"):
                window = values[i:i + window_size]
                label_seq = values[i + window_size:i + window_size + forecast_horizon, target_idx]
                all_sequences.append(torch.tensor(window, dtype=torch.float32))
                all_labels.append(torch.tensor(label_seq, dtype=torch.float32))  # shape: [forecast_horizon]

        self.input_features = df.shape[1]
        self.output_features = 1       #hardcoding for now, TODO fix later

        # Split into train/val if requested
        total = len(all_sequences)
        split_idx = int(total * (1 - val_ratio))

        if split == 'train':
            self.data = all_sequences[:split_idx]
            self.targets = torch.stack(all_labels[:split_idx])  # Convert list to tensor
            self.targets = self.targets.unsqueeze(-1)  # Reshape to (B, forecast_horizon, 1)
        elif split == 'val':
            self.data = all_sequences[split_idx:]
            self.targets = torch.stack(all_labels[split_idx:])  # Convert list to tensor
            self.targets = self.targets.unsqueeze(-1)  # Reshape to (B, forecast_horizon, 1)
        else:
            self.data = all_sequences
            self.targets = torch.stack(all_labels)  # Convert list to tensor
            self.targets = self.targets.unsqueeze(-1)  # Reshape to (B, forecast_horizon, 1)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[Tensor, Tensor]: (past_window, future_targets)
                - past_window: shape [window_size, num_features]
                - future_targets: shape [forecast_horizon]
        """
        return self.data[idx], self.targets[idx]

def verify_financial_dataloader(dataloader):
    '''
    Verifies and displays key information about a FinancialTimeSeriesDataset dataloader.

    1. Shows partition (train/val or full)
    2. Displays number of batches and batch size
    3. Displays input and output shapes from the first batch
    4. Reports window size, forecast horizon, and number of input features
    '''
    def print_shapes(past_windows, future_targets):
        print(f"{'Past Window Shape':<25}: {list(past_windows.shape)}")
        print(f"{'Future Target Shape':<25}: {list(future_targets.shape)}")

    print("=" * 50)
    print(f"{'Financial Dataloader Verification':^50}")
    print("=" * 50)

    # Determine partition name if available
    partition = getattr(dataloader.dataset, 'split', 'unspecified')
    print(f"{'Dataloader Partition':<25}: {partition}")
    print("-" * 50)

    print(f"{'Number of Batches':<25}: {len(dataloader)}")
    print(f"{'Batch Size':<25}: {dataloader.batch_size}")
    print("-" * 50)

    print(f"{'Checking shapes of the first batch...':<50}\n")
    for i, batch in enumerate(dataloader):
        if i > 0:
            break
        past_windows, future_targets = batch
        print_shapes(past_windows, future_targets)

    print("-" * 50)
    print(f"{'Window Size':<25}: {dataloader.dataset.window_size}")
    print(f"{'Forecast Horizon':<25}: {dataloader.dataset.forecast_horizon}")
    if len(dataloader.dataset.features) > 0:
        print(f"{'Num Input Features':<25}: {len(dataloader.dataset.features) + 1} (includes Time)")
    else:
        print(f"{'Num Input Features':<25}: Unknown")
    print(f"{'Target Feature':<25}: {dataloader.dataset.target}")
    print("=" * 50)
