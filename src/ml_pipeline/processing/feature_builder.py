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

    if not price_file.exists() or not sentiment_file.exists():
        raise FileNotFoundError("Input files missing. Run market_data and sentiment_analyzer first.")

    # 1. Load Datasets
    price_df = pd.read_csv(price_file)
    sentiment_df = pd.read_csv(sentiment_file)

    # 2. Format Dates
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    
    # Extract date from news published timestamp
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['published_at'], errors='coerce').dt.date
    sentiment_df = sentiment_df.dropna(subset=['Date'])

    # 3. Identify Available Sentiment Columns
    possible_sentiment_cols = ['positive', 'negative', 'neutral', 'sentiment_score']
    available_sentiment_cols = [col for col in possible_sentiment_cols if col in sentiment_df.columns]

    if not available_sentiment_cols:
        raise KeyError("No recognized sentiment columns found in sentiment dataset.")

    # 4. Aggregate Daily Sentiment Scores
    agg_dict = {col: 'mean' for col in available_sentiment_cols}
    daily_sentiment = sentiment_df.groupby('Date').agg(agg_dict).reset_index()

    # 5. Merge Price Data with Sentiment Data
    merged_df = pd.merge(price_df, daily_sentiment, on='Date', how='left')

    # 6. Handle Days Without News (Fill NaN sentiment with default 0.0)
    merged_df[available_sentiment_cols] = merged_df[available_sentiment_cols].fillna(0.0)

    # 7. Save Unified Dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully generated merged dataset ({len(merged_df)} rows) at {output_file}")


if __name__ == "__main__":
    price_path = DATA_DIR / "raw" / "AAPL_historical.csv"
    sentiment_path = DATA_DIR / "processed" / "AAPL_news_sentiment.csv"
    output_path = DATA_DIR / "processed" / "AAPL_feature_matrix.csv"

    build_unified_dataset(price_path, sentiment_path, output_path)