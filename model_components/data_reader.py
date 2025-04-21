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
        tgt_scaler: Optional[object] = None, 
        cls_scaler: Optional[object] = None, # <-- new
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
        self.tgt_scaler = tgt_scaler
        self.cls_scaler = cls_scaler
        self.fit_scaler = fit_scaler

        sequences = []
        labels_shifted = []
        labels_shifted_targfeat = []
        labels_golden = []

       

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
            cls_features = [f for f in features if f != target]
            print(cls_features)
            

            if normalize:
                scaler_cls = StandardScaler if normalize == 'zscore' else MinMaxScaler
                scaler_tgt = StandardScaler if normalize == 'zscore' else MinMaxScaler
                if fit_scaler:
                    self.tgt_scaler = scaler_tgt()
                    self.cls_scaler = scaler_cls()
                    self.cls_scaler.fit(df[cls_features])
                    self.tgt_scaler.fit(df[[target]])
                if self.tgt_scaler is not None:
                    df[cls_features] = self.cls_scaler.transform(df[cls_features])
                    df[[target]] = self.tgt_scaler.transform(df[[target]])

                    print(f"after normalization dataset is {df[cls_features].head()}")
                    print(f"after normalization dataset is {df[target].head()}")

            # df = df.reset_index(drop=True)
            # print(f"after resorting dataset is {df[features].head()}")
            values = df.values
            target_idx = features.index(target)
            print(df.columns)

            # Inside your for-loop where you process each time window
            for i in tqdm(
                range(0, len(values) - window_size - forecast_horizon + 1, shift),
                desc=f"Processing {ticker} ({split or 'full'})"
            ):
                x = values[i:i + window_size]  # Input sequence (past window)
                print(f"x is {x.shape}")

                # # Create shifted and golden targets
                if i == 0:
                    # For the first window, there is no previous input, so replicate the first row
                    targets_shifted = np.vstack((x[0], x[:-1]))  # [window_size, num_features]
                else:
                    # Use the last row of the previous window as the first row of the shifted targets
                    targets_shifted = np.vstack((values[i - 1 + window_size - 1], x[:-1]))  # [window_size, num_features]

                print(f"targets_shifted is {targets_shifted.shape}")
                targets_shifted_targfeat = targets_shifted[:, target_idx:target_idx + 1]  # [window_size, 1]
                targets_golden = x[:, target_idx]  # [window_size]
                

                # Convert to PyTorch tensors (no flattening anymore!)
                targets_shifted = torch.tensor(targets_shifted, dtype=torch.float32)                # [window_size, num_features]
                targets_shifted_targfeat = torch.tensor(targets_shifted_targfeat, dtype=torch.float32)  # [window_size, 1]
                targets_golden = torch.tensor(targets_golden, dtype=torch.float32).unsqueeze(-1)       # [window_size, 1]
                print(f"targets_golden is {targets_golden.shape}")
                
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


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: (past_window, (shifted_targets, golden_targets))
                - past_window: shape [window_size, num_features]
                - shifted_targets: shape [window_size, num_features]
                - golden_targets: shape [window_size + forecast_horizon, num_features]
        """
        return self.data[idx], self.targets_shifted[idx], self.targets_shifted_targfeat[idx], self.targets[idx]