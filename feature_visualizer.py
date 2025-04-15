import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_reader import FinancialTimeSeriesDataset
from positional_encoding import TimeSeriesPositionalEncoding
from timeseries_embedding import Time2VecTorch

def process_and_visualize(tickers, start_date, end_date, features):
    # 1. Load Data
    dataset = FinancialTimeSeriesDataset(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        features=features,
        window_size=30,
        normalize='zscore'
    )
    
    # 2. Create Model Components
    d_model = 16  # Embedding dimension
    time_embedder = Time2VecTorch(d_model)
    pos_encoder = TimeSeriesPositionalEncoding(d_model, max_len=5000)
    
    # 3. Process Batch
    sample_idx = 0  # Visualize first window
    raw_data, target = dataset[sample_idx]  # (seq_len, features), (1,)
    
    # Add batch dimension
    raw_data = raw_data.unsqueeze(0)  # (1, seq_len, features)
    
    # 4. Apply Embedding and Encoding
    time_emb = time_embedder(raw_data)  # (1, seq_len, d_model)
    encoded_data = pos_encoder(time_emb)  # (1, seq_len, d_model)
    
    # 5. Visualization Preparation
    seq_len = encoded_data.shape[1]
    time_steps = np.arange(seq_len)
    feature_labels = [f'Emb_{i}' for i in range(d_model)]
    
    # Convert to numpy for visualization
    encoded_np = encoded_data.squeeze(0).detach().numpy()
    
    # 6. Create Visualizations
    plt.figure(figsize=(20, 12))
    
    # 6.1 Original Features Heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(
        raw_data.squeeze(0).numpy().T,
        annot=False,
        cmap='viridis',
        yticklabels=features,
        xticklabels=time_steps
    )
    plt.title('Original Features (Normalized)')
    plt.xlabel('Time Steps')
    plt.ylabel('Financial Features')
    
    # 6.2 Time2Vec Components
    plt.subplot(2, 2, 2)
    time_emb_np = time_emb.squeeze(0).detach().numpy()
    plt.plot(time_emb_np[:, 0], label='Linear Component')
    plt.plot(time_emb_np[:, 1], label='Periodic Component 1')
    plt.plot(time_emb_np[:, 2], label='Periodic Component 2')
    plt.title('Time2Vec Embedding Components')
    plt.xlabel('Time Steps')
    plt.ylabel('Embedding Value')
    plt.legend()
    
    # 6.3 Positional Encoding Heatmap
    plt.subplot(2, 2, 3)
    pos_encoding = pos_encoder.pe.squeeze(0)[:seq_len].numpy().T
    sns.heatmap(
        pos_encoding,
        cmap='coolwarm',
        yticklabels=[f'PE_{i}' for i in range(d_model)],
        xticklabels=time_steps
    )
    plt.title('Positional Encoding Patterns')
    plt.xlabel('Time Steps')
    plt.ylabel('Encoding Dimensions')
    
    # 6.4 Combined Embedding Correlation
    plt.subplot(2, 2, 4)
    combined_features = encoded_np.T
    sns.heatmap(
        np.corrcoef(combined_features),
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        annot=True,
        fmt='.2f',
        xticklabels=feature_labels,
        yticklabels=feature_labels
    )
    plt.title('Embedding Feature Correlations')
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('Embedding Dimensions')
    
    plt.tight_layout()
    plt.show()
    plt.savefig("fin_embed.png")

# Usage
if __name__ == "__main__":
    process_and_visualize(
        tickers=['AAPL'],
        start_date='2022-01-01',
        end_date='2023-01-01',
        features=['Open', 'High', 'Low', 'Close', 'Volume']
    )
