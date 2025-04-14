from torch.utils.data import DataLoader
from data_reader import FinancialTimeSeriesDataset
import matplotlib.pyplot as plt
import torch

# Define tickers and time range
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
start_date = "2014-01-01"
end_date = "2024-12-31"

# Instantiate dataset
dataset = FinancialTimeSeriesDataset(
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    features=["Open", "High", "Low", "Close", "Volume"],
    window_size=30,
    target="Close",
    normalize=None  # You can also try "minmax" or None
)

# Check dataset length and example sample
print(f"Total samples: {len(dataset)}")
sample_input, sample_target = dataset[0]
# print(f"{sample_input}")

# Plot the input window
plt.plot(sample_input.numpy())  # Index 3 is 'Close' after feature list
plt.title("Normalized Close Price (30-day window)")
plt.xlabel("Days")
plt.ylabel("Z-score Normalized Value")
plt.show()

plt.savefig("fin.png")

# Optional: Test in a DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
batch_inputs, batch_targets = next(iter(loader))
print(f"Batch input shape: {batch_inputs.shape}")
print(f"Batch target shape: {batch_targets.shape}")
