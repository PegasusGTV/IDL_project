import yfinance as yf
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Helper functions for feature calculations
import pandas as pd
from io import StringIO


def compute_sma(df, column, window):
    """
    Compute Simple Moving Average (SMA).
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to calculate SMA on.
        window (int): Window size for moving average.
    
    Returns:
        pd.Series: SMA values.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    return df[column].rolling(window=window, min_periods=1).mean()


def compute_rsi(df, column, window=14):
    """
    Compute Relative Strength Index (RSI).
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to calculate RSI on.
        window (int): Period over which to compute RSI.
    
    Returns:
        pd.Series: RSI values.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    # Make sure it's a 1D Series, not DataFrame
    series = df[column]

    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_halftrend(df, high_col='High', low_col='Low', close_col='Close', window=20):
    """
    Compute Half-Trend indicator (basic but closer to true version).
    
    Args:
        df (pd.DataFrame): DataFrame with High, Low, Close columns.
        high_col (str): Column name for high prices.
        low_col (str): Column name for low prices.
        close_col (str): Column name for close prices.
        window (int): Window for trend smoothing and logic.
    
    Returns:
        pd.Series: Half-Trend signal (+1 for uptrend, -1 for downtrend).
    """
    if not all(col in df.columns for col in [high_col, low_col, close_col]):
        raise ValueError("Missing required columns.")

    hl2 = (df[high_col] + df[low_col]) / 2

    # Smoothed version of hl2
    avg_price = hl2.rolling(window=window, min_periods=1).mean()

    direction = np.zeros(len(df))
    trend = np.zeros(len(df))

    # Initial direction
    for i in range(1, len(df)):
        # Compare the close price to the smoothed average
        if df[close_col].iloc[i].item() > avg_price.iloc[i - 1].item():
            direction[i] = 1
        elif df[close_col].iloc[i].item() < avg_price.iloc[i - 1].item():
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]  # maintain previous direction

        # Store trend for visualization or later use
        trend[i] = direction[i]

    return pd.Series(trend, index=df.index, name='HalfTrend')



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
            
            # Add additional features
            # df['SMA_10'] = compute_sma(df, 'Close', window=10)
            # df['RSI_14'] = compute_rsi(df, 'Close', window=14)
            # df['HalfTrend_20'] = compute_halftrend(df)

            

            # print(f"initial dataset is {df.head()}")
            # print(f"df columns are {df.columns}")
            # df['Time'] = (df.index - df.index.min()).days
            

            # Split into train/val if requested
            train_end_idx = int(len(df) * (1 - val_ratio))
           
            # Add normalized time feature
            # df['Time'] = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())
            # print(f"after resorting dataset is {df["Close"].head()}")
            df.columns = ['{}'.format(col[0]) for col in df.columns]
            df.reset_index().rename(columns={'Date': 'date'})

            # print(f"after time dataset is {df.head()}")
            # print(f"df columns are {df.head}")
            print(f"df columns are {df.columns}")

            # ADDING SENTIMENT DATA

            # sentiment_df = pd.read_csv('daily_finbert_sentiment.csv')
            # print(f"sentiment_df is {sentiment_df.columns}")
            # sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

            # # Generate full date range (replace with actual Yahoo Finance dates)
            # merged_df = pd.merge_asof(
            #     left=df.reset_index().rename(columns={'Date': 'date'}),
            #     right=sentiment_df[['date', 'sentiment_score']],
            #     on='date',
            #     direction='nearest'
            # ).set_index('date')


            # Linear interpolation for missing values
            # merged_df['sentiment_score'] = merged_df['sentiment_score'].interpolate(method='linear')
            # merged_df['Time'] = np.arange(len(merged_df))   #for continuous time

            # df = merged_df
            cls_features = [f for f in df.columns if f != target]


            # ADDING SENTIMENT DATA
            

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

            if split == 'train':
                df = df.iloc[:train_end_idx]
            elif split == 'val':
                df = df.iloc[train_end_idx:]

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
                # print(f"x is {x.shape}")

                # # Create shifted and golden targets
                if i == 0:
                    # For the first window, there is no previous input, so replicate the first row
                    targets_shifted = np.vstack((x[0], x[:-1]))  # [window_size, num_features]
                else:
                    # Use the last row of the previous window as the first row of the shifted targets
                    targets_shifted = np.vstack((values[i - 1 + window_size - 1], x[:-1]))  # [window_size, num_features]

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