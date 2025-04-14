import yfinance as yf
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

"""
financial_dataset.py

A PyTorch Dataset class for loading and processing financial time series data from Yahoo Finance.

This module defines the `FinancialTimeSeriesDataset` class, which provides a structured way to
prepare stock market data for machine learning models. It supports multi-ticker loading, customizable
features (such as 'Open', 'Close', 'Volume'), windowed input sequences, and prediction targets (e.g., next-day close).
The dataset also includes optional normalization methods (z-score and min-max scaling) and works with
standard PyTorch DataLoader objects for batching and training.

Typical usage example:

    dataset = FinancialTimeSeriesDataset(
        tickers=["AAPL", "GOOG"],
        start_date="2015-01-01",
        end_date="2024-12-31",
        features=["Open", "High", "Low", "Close", "Volume"],
        window_size=30,
        target="Close",
        normalize="zscore"
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for inputs, targets in loader:
        # inputs: Tensor of shape [batch_size, window_size, num_features]
        # targets: Tensor of shape [batch_size, 1]
        ...
"""


class FinancialTimeSeriesDataset(Dataset):
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        features: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume'],
        window_size: int = 30,
        target: str = 'Close',
        normalize: Optional[str] = 'zscore'
    ):
        """
        Initialize the financial time series dataset.

        Args:
            tickers (List[str]): List of stock tickers.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            features (List[str]): List of features to include.
            window_size (int): Number of days in each input window.
            target (str): Feature to predict at day t+1.
            normalize (Optional[str]): Normalization method ('zscore', 'minmax', or None).
        """
        self.window_size = window_size
        self.features = features
        self.target = target
        self.normalize = normalize

        self.data = []
        self.targets = []

        for ticker in tickers:
            df = yf.download(ticker, start=start_date, end=end_date)[features]
            df = df.dropna()
            print(df.head)

            if normalize == 'zscore':
                df = (df - df.mean()) / (df.std() + 1e-8)
            elif normalize == 'minmax':
                df = (df - df.min()) / (df.max() - df.min() + 1e-8)

            df_values = df.values
            target_idx = features.index(target)

            for i in tqdm(range(len(df_values) - window_size)):
                window = df_values[i:i + window_size]
                label = df_values[i + window_size][target_idx]
                self.data.append(torch.tensor(window, dtype=torch.float32))
                self.targets.append(torch.tensor(label, dtype=torch.float32))

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the input-output pair for a given index.

        Returns:
            Tuple: (features: Tensor of shape [window_size, num_features], target: Tensor [1])
        """
        return self.data[idx], self.targets[idx].unsqueeze(0)
