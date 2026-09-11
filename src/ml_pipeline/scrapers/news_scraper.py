import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


def fetch_finance_news(ticker: str) -> list:
    """Fetch recent news headlines for a given ticker using Yahoo Finance RSS."""
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    print(f"Scraping headlines for {ticker}...")
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch news. HTTP Status: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")

    articles = []
    for item in items:
        title = item.find("title").text if item.find("title") else ""
        pub_date = item.find("pubDate").text if item.find("pubDate") else ""
        link = item.find("link").text if item.find("link") else ""

        if title:
            articles.append({
                "ticker": ticker,
                "headline": title,
                "published_at": pub_date,
                "url": link,
                "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    return articles


def save_headlines(articles: list, ticker: str, output_dir: str = "data/raw"):
    """Save scraped headlines to CSV."""
    if not articles:
        print("No articles found to save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(articles)
    file_path = os.path.join(output_dir, f"{ticker}_news.csv")
    df.to_csv(file_path, index=False)
    print(f"Successfully saved {len(articles)} headlines to {file_path}")


if __name__ == "__main__":
    ticker = "AAPL"
    news_items = fetch_finance_news(ticker)
    save_headlines(news_items, ticker)

