from src.ml_pipeline.processing.market_data import fetch_market_data
from src.ml_pipeline.scrapers.news_scraper import fetch_latest_news
from src.ml_pipeline.models.sentiment_analyzer import analyze_news_sentiment
from src.ml_pipeline.processing.feature_builder import build_unified_dataset
from src.execution_engine.trade_executor import PaperTradingEngine


def run_pipeline(ticker="AAPL"):
    print("=" * 60)
    print(f"🚀 RUNNING END-TO-END TRADING PIPELINE FOR: {ticker}")
    print("=" * 60)

    # Step 1: Fetch Raw Market & News Data
    print("\n[1/5] Fetching Historical Market Data & Latest News...")
    price_file = fetch_market_data(ticker=ticker)
    raw_news_file = fetch_latest_news(ticker=ticker)

    # Step 2: FinBERT Sentiment Analysis
    print("\n[2/5] Analyzing News Sentiment using FinBERT...")
    sentiment_file = analyze_news_sentiment(raw_news_file)

    # Step 3: Build Merged Feature Matrix
    print("\n[3/5] Merging Price Technical Indicators with Sentiment...")
    feature_file = f"data/processed/{ticker}_feature_matrix.csv"
    build_unified_dataset(price_file, sentiment_file, feature_file)

    # Step 4: Model Signal Output
    print("\n[4/5] Generating Model Trading Signal...")
    prediction_probability = 0.62  # Simulated signal confidence
    current_price = 225.50
    print(f"-> Directional Upward Probability: {prediction_probability:.2f}")

    # Step 5: Execute Trade
    print("\n[5/5] Executing Paper Trade via Execution Engine...")
    engine = PaperTradingEngine(initial_capital=10004.50)
    engine.execute_signal(
        ticker=ticker,
        current_price=current_price,
        prediction_probability=prediction_probability
    )
    engine.save_trade_logs()

    print("\n" + "=" * 60)
    print("✅ PIPELINE EXECUTION COMPLETE! Refresh dashboard to view updates.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()

