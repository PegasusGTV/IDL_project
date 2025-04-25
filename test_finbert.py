import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import csv




def main():
    # Load FinBERT
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Stream FNSPID dataset
    ds = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)

    # Dictionary to store one entry per date
    daily_articles = {}
    max_entries = 3000  # Limit to 500 unique dates

    print("📥 Streaming and filtering dataset...")
    for entry in tqdm(ds, desc="Loading entries"):
        # Parse date
        date_str = entry.get("Date", "")[:10]  # e.g., '2020-03-15'
        # print(date_str)
        sentence = entry.get("Article_title", None)
        if not sentence or not date_str:
            continue
        if date_str not in daily_articles:
            # print(f"new entry!")
            daily_articles[date_str] = sentence
        if len(daily_articles) >= max_entries:
            break

    # Sort by date
    sorted_dates = sorted(daily_articles.keys())
    print(f"\n📅 Date range: {sorted_dates[0]} → {sorted_dates[-1]}\n")

    # Sentiment mapping: Negative=-1, Neutral=0, Positive=+1
    label_to_score = {0: -1, 1: 0, 2: 1}
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    print("🔍 Running sentiment analysis...")
    sentiment_scores = []
    sentiment_data = []

    for date in tqdm(sorted_dates, desc="Predicting sentiments"):
        text = daily_articles[date]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            label = label_map[pred]
            score = label_to_score[pred]
            sentiment_scores.append(score)
            sentiment_data.append({
                "date": date,
                "title": text[:100].replace('\n', ' '),  # truncate and sanitize
                "sentiment_label": label,
                "sentiment_score": score
            })

    # Save to CSVsaafe-+//
    output_csv = "daily_finbert_sentiment.csv"
    with open(output_csv, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "title", "sentiment_label", "sentiment_score"])
        writer.writeheader()
        writer.writerows(sentiment_data)

    print(f"\n✅ Sentiment data saved to: {output_csv}")

    # Plotting
    print("📊 Plotting sentiment over time...")
    dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in sentiment_data]
    scores = [row["sentiment_score"] for row in sentiment_data]

    plt.figure(figsize=(14, 6))
    plt.plot(dates, scores, label="Daily Sentiment", color="tab:blue", linewidth=0.8)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Daily FinBERT Sentiment Score (2015–2024)")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score (-1 = Negative, 0 = Neutral, 1 = Positive)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("finbert_sentiment_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()