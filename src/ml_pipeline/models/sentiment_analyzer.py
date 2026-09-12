import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def analyze_news_sentiment(news_file: str, ticker: str = None) -> str:
    """Analyzes sentiment of headlines and saves output dynamically per ticker."""
    # Infer ticker from filename if not explicitly passed
    if ticker is None:
        ticker = os.path.basename(news_file).replace("_news.csv", "")

    os.makedirs("data/processed", exist_ok=True)
    output_file = f"data/processed/{ticker}_news_sentiment.csv"

    if not os.path.exists(news_file):
        print(f"Warning: {news_file} missing. Creating fallback sentiment matrix.")
        df_empty = pd.DataFrame({"headline": ["No news"], "sentiment_score": [0.0]})
        df_empty.to_csv(output_file, index=False)
        return output_file

    df = pd.read_csv(news_file)
    if df.empty or "headline" not in df.columns:
        print(f"No headlines found in {news_file}. Saving neutral baseline.")
        df_empty = pd.DataFrame({"headline": ["No news"], "sentiment_score": [0.0]})
        df_empty.to_csv(output_file, index=False)
        return output_file

    print("Loading FinBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    print(f"Analyzing sentiment for {len(df)} headlines...")
    results = nlp(df["headline"].tolist())

    scores = []
    for res in results:
        label = res["label"]
        score = res["score"]
        if label == "positive":
            scores.append(score)
        elif label == "negative":
            scores.append(-score)
        else:
            scores.append(0.0)

    df["sentiment_score"] = scores

    df.to_csv(output_file, index=False)
    print(f"Successfully saved sentiment analysis results to {output_file}")

    return output_file