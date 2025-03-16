import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax
import pandas as pd
import matplotlib.pyplot as plt

# Load FinBERT model and tokenizer
MODEL_NAME = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Function to get sentiment scores
def get_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs.logits.numpy()[0])  # Convert logits to probabilities
    labels = ['negative', 'neutral', 'positive']
    return dict(zip(labels, scores))

# Load extracted financial news data (example CSV file with 'date' and 'headline')
df = pd.read_csv("financial_news.csv")  # Ensure it has 'date' and 'headline' columns
df['date'] = pd.to_datetime(df['date'])

# Compute sentiment scores
df['sentiment'] = df['headline'].apply(lambda x: get_sentiment(x)['positive'] - get_sentiment(x)['negative'])

# Aggregate sentiment over time
df_sentiment = df.groupby('date')['sentiment'].mean().reset_index()

# Plot sentiment over time
plt.figure(figsize=(12, 6))
plt.plot(df_sentiment['date'], df_sentiment['sentiment'], label='Sentiment', color='blue')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Financial News Sentiment Over Time')
plt.legend()
plt.show()
