import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        print("Loading FinBERT model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ["positive", "negative", "neutral"]

    def analyze_headline(self, headline: str) -> dict:
        """Process a single headline and return probabilities for positive, negative, and neutral."""
        inputs = self.tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()

        result = dict(zip(self.labels, scores))
        # Compute a single aggregate sentiment score (-1.0 to +1.0)
        result["sentiment_score"] = result["positive"] - result["negative"]
        return result

    def process_csv(self, input_file: str, output_file: str):
        """Load headlines from CSV, calculate sentiment scores, and save processed results."""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")

        df = pd.read_csv(input_file)
        if "headline" not in df.columns:
            raise ValueError("CSV must contain a 'headline' column.")

        print(f"Analyzing sentiment for {len(df)} headlines...")
        
        scores_list = []
        for headline in df["headline"]:
            scores = self.analyze_headline(str(headline))
            scores_list.append(scores)

        scores_df = pd.DataFrame(scores_list)
        final_df = pd.concat([df, scores_df], axis=1)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_csv(output_file, index=False)
        print(f"Successfully saved sentiment analysis results to {output_file}")


def analyze_news_sentiment(input_file: str = "data/raw/AAPL_news.csv") -> str:
    """Wrapper function for main pipeline orchestrator."""
    analyzer = SentimentAnalyzer()
    output_path = "data/processed/AAPL_news_sentiment.csv"
    analyzer.process_csv(input_file, output_path)
    return output_path


if __name__ == "__main__":
    analyze_news_sentiment()