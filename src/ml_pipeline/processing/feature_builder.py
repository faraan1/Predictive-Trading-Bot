import os
import pandas as pd
from typing import Union
from pathlib import Path
from config.config import DATA_DIR


def build_unified_dataset(
    price_file: Union[str, Path], 
    sentiment_file: Union[str, Path], 
    output_file: Union[str, Path]
) -> None:
    """Merge historical price indicators with aggregated daily sentiment scores."""
    price_file = Path(price_file)
    sentiment_file = Path(sentiment_file)
    output_file = Path(output_file)

    if not price_file.exists():
        raise FileNotFoundError(f"Price data file missing: {price_file}")

    # 1. Load Datasets
    price_df = pd.read_csv(price_file)
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date

    # Load sentiment safely if file exists and has content
    if sentiment_file.exists() and sentiment_file.stat().st_size > 0:
        sentiment_df = pd.read_csv(sentiment_file)
    else:
        sentiment_df = pd.DataFrame()

    # 2. Check for valid sentiment columns and published_at timestamp
    if not sentiment_df.empty and 'published_at' in sentiment_df.columns:
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['published_at'], errors='coerce').dt.date
        sentiment_df = sentiment_df.dropna(subset=['Date'])

        possible_sentiment_cols = ['positive', 'negative', 'neutral', 'sentiment_score']
        available_sentiment_cols = [col for col in possible_sentiment_cols if col in sentiment_df.columns]

        if available_sentiment_cols:
            agg_dict = {col: 'mean' for col in available_sentiment_cols}
            daily_sentiment = sentiment_df.groupby('Date').agg(agg_dict).reset_index()
            merged_df = pd.merge(price_df, daily_sentiment, on='Date', how='left')
            merged_df[available_sentiment_cols] = merged_df[available_sentiment_cols].fillna(0.0)
        else:
            merged_df = price_df.copy()
            merged_df['sentiment_score'] = 0.0
    else:
        # Fallback when zero news articles are scraped for a ticker
        merged_df = price_df.copy()
        merged_df['sentiment_score'] = 0.0

    # 3. Save Unified Dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully generated merged dataset ({len(merged_df)} rows) at {output_file}")


if __name__ == "__main__":
    price_path = DATA_DIR / "raw" / "AAPL_historical.csv"
    sentiment_path = DATA_DIR / "processed" / "AAPL_news_sentiment.csv"
    output_path = DATA_DIR / "processed" / "AAPL_feature_matrix.csv"

    build_unified_dataset(price_path, sentiment_path, output_path)