import yfinance as yf
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
        val_ratio: float = 0.2,
        shift: int = 2,  # NEW
        scaler: Optional[object] = None,  # <-- new
        fit_scaler: bool = False          # <-- new
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

        self.split = split
        self.shift = shift
        self.scaler = scaler
        self.fit_scaler = fit_scaler

        sequences = []
        labels = []

       

        for ticker in tickers:
            df = yf.download(ticker, start=start_date, end=end_date)[features]
            df = df.dropna()
            # print(f"initial dataset is {df.head()}")
            # print(f"df columns are {df.columns}")
            # df['Time'] = (df.index - df.index.min()).days
            df['Time', ticker] = np.arange(len(df))   #for continuous time

            # print(f"after time dataset is {df.head()}")
            # print(f"df columns are {df.columns}")

            # Split into train/val if requested
            train_end_idx = int(len(df) * (1 - val_ratio))

            if split == 'train':
                df = df.iloc[:train_end_idx]
            elif split == 'val':
                df = df.iloc[train_end_idx:]
           
            # Add normalized time feature
            # df['Time'] = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())
            # print(f"after resorting dataset is {df["Close"].head()}")
            df.columns = ['{}'.format(col[0]) for col in df.columns]

            if normalize:
                scaler_cls = StandardScaler if normalize == 'zscore' else MinMaxScaler
                if fit_scaler:
                    self.scaler = scaler_cls()
                    self.scaler.fit(df[features])

                if self.scaler is not None:
                    df[features] = self.scaler.transform(df[features])

            # df = df.reset_index(drop=True)
            # print(f"after resorting dataset is {df[features].head()}")
            values = df.values
            target_idx = features.index(target)
            # print(f"values are {values}")
            # print(f"target is {df[features[target_idx]]}")
            # CHANGE THIS SO THAT WE ARE PREDICTING THE ENTIRE FEATURE SEQ, NOT JUST THE CLOSING PRICE

            for i in tqdm(
                range(0, len(values) - window_size - forecast_horizon + 1, shift),
                desc=f"Processing {ticker} ({split or 'full'})"
            ):
                window = values[i:i + window_size]
                label_seq = values[i + window_size:i + window_size + forecast_horizon, target_idx]
                
                # DEBUG:
                # print(window)
                # print(label_seq)

                label_seq = torch.tensor(label_seq, dtype=torch.float32).unsqueeze(-1)  # [forecast_horizon, 1]

                sequences.append(torch.tensor(window, dtype=torch.float32))  # [window_size, input_features]
                labels.append(label_seq)                                     # [forecast_horizon, 1]

        self.data = torch.stack(sequences)      # [num_samples, window_size, input_features]
        self.targets = torch.stack(labels)      # [num_samples, forecast_horizon, 1]
        self.input_features = self.data.shape[-1]
        self.output_features = 1  # still predicting only one feature TODO un-hard code

        print(self.data, self.targets)


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