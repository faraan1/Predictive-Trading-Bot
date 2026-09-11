import os
import json
from datetime import datetime


class PaperTradingEngine:
    def __init__(self, initial_capital: float = 10000.0, max_risk_per_trade: float = 0.02):
        self.cash = initial_capital
        self.portfolio = {}  # {ticker: shares}
        self.trade_history = []
        self.max_risk_per_trade = max_risk_per_trade  # 2% max risk per trade

    def calculate_position_size(self, current_price: float) -> int:
        """Risk Management: Calculate maximum shares to buy based on cash risk limits."""
        risk_amount = self.cash * self.max_risk_per_trade
        shares = int(risk_amount // current_price)
        return max(shares, 1)  # Ensure at least 1 share if affordable

    def execute_signal(self, ticker: str, current_price: float, prediction_probability: float, threshold: float = 0.55):
        """Execute Buy/Sell paper trades based on model signal confidence."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # BUY SIGNAL: High model confidence of upward movement
        if prediction_probability >= threshold:
            shares_to_buy = self.calculate_position_size(current_price)
            cost = shares_to_buy * current_price

            if self.cash >= cost:
                self.cash -= cost
                self.portfolio[ticker] = self.portfolio.get(ticker, 0) + shares_to_buy
                
                log_entry = {
                    "timestamp": timestamp, "ticker": ticker, "action": "BUY",
                    "price": current_price, "shares": shares_to_buy,
                    "confidence": round(prediction_probability, 4), "remaining_cash": round(self.cash, 2)
                }
                self.trade_history.append(log_entry)
                print(f"🟢 [BUY ORDER] Bought {shares_to_buy} shares of {ticker} @ ${current_price:.2f}")

        # SELL SIGNAL: Low confidence / bearish prediction
        elif prediction_probability < (1 - threshold):
            if ticker in self.portfolio and self.portfolio[ticker] > 0:
                shares_to_sell = self.portfolio[ticker]
                revenue = shares_to_sell * current_price
                self.cash += revenue
                self.portfolio[ticker] = 0

                log_entry = {
                    "timestamp": timestamp, "ticker": ticker, "action": "SELL",
                    "price": current_price, "shares": shares_to_sell,
                    "confidence": round(prediction_probability, 4), "remaining_cash": round(self.cash, 2)
                }
                self.trade_history.append(log_entry)
                print(f"🔴 [SELL ORDER] Sold {shares_to_sell} shares of {ticker} @ ${current_price:.2f}")
        else:
            print(f"⚪ [HOLD] Model confidence ({prediction_probability:.2f}) within neutral band. No trade executed.")

    def save_trade_logs(self, output_path: str = "data/processed/trade_logs.json"):
        """Persist trade logs to file for dashboard reporting."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "account_balance": round(self.cash, 2),
                "portfolio": self.portfolio,
                "history": self.trade_history
            }, f, indent=4)
        print(f"Trade logs successfully saved to {output_path}")


if __name__ == "__main__":
    # Test trading engine with mock signals
    engine = PaperTradingEngine(initial_capital=10000.0)
    
    # Test Buy signal (0.68 confidence)
    engine.execute_signal(ticker="AAPL", current_price=180.50, prediction_probability=0.68)
    
    # Test Sell signal (0.35 confidence)
    engine.execute_signal(ticker="AAPL", current_price=185.00, prediction_probability=0.35)
    
    engine.save_trade_logs()

