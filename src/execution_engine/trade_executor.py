python -c "
code = '''import os
import json
from datetime import datetime


class PaperTradingEngine:
    def __init__(self, initial_capital: float = 10000.0, max_risk_per_trade: float = 0.10, log_path: str = \"data/processed/trade_logs.json\"):
        self.log_path = log_path
        self.cash = initial_capital
        self.portfolio = {}  # {ticker: shares}
        self.trade_history = []
        self.max_risk_per_trade = max_risk_per_trade  # 10% max capital allocation per trade

        # Load existing state if present
        self.load_trade_logs()

    def calculate_position_size(self, current_price: float) -> int:
        \"\"\"Calculate maximum shares to buy based on risk allocation limits.\"\"\"
        allocation_amount = self.cash * self.max_risk_per_trade
        shares = int(allocation_amount // current_price)
        if shares == 0 and self.cash >= current_price:
            shares = 1  # Buy at least 1 share if cash permits
        return shares

    def execute_signal(self, ticker: str, current_price: float, prediction_probability: float, threshold: float = 0.51):
        \"\"\"Execute Buy/Sell paper trades based on model signal confidence.\"\"\"
        timestamp = datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")

        # BUY SIGNAL
        if prediction_probability >= threshold:
            shares_to_buy = self.calculate_position_size(current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                if self.cash >= cost:
                    self.cash -= cost
                    self.portfolio[ticker] = self.portfolio.get(ticker, 0) + shares_to_buy

                    log_entry = {
                        \"timestamp\": timestamp, \"ticker\": ticker, \"action\": \"BUY\",
                        \"price\": round(current_price, 2), \"shares\": shares_to_buy,
                        \"confidence\": round(prediction_probability, 4), \"remaining_cash\": round(self.cash, 2)
                    }
                    self.trade_history.append(log_entry)
                    print(f\"🟢 [BUY ORDER] Bought {shares_to_buy} shares of {ticker} @ ${current_price:.2f}\")

        # SELL SIGNAL
        elif prediction_probability <= (1.0 - threshold):
            if ticker in self.portfolio and self.portfolio[ticker] > 0:
                shares_to_sell = self.portfolio[ticker]
                revenue = shares_to_sell * current_price
                self.cash += revenue
                self.portfolio[ticker] = 0

                log_entry = {
                    \"timestamp\": timestamp, \"ticker\": ticker, \"action\": \"SELL\",
                    \"price\": round(current_price, 2), \"shares\": shares_to_sell,
                    \"confidence\": round(prediction_probability, 4), \"remaining_cash\": round(self.cash, 2)
                }
                self.trade_history.append(log_entry)
                print(f\"🔴 [SELL ORDER] Sold {shares_to_sell} shares of {ticker} @ ${current_price:.2f}\")
        else:
            print(f\"⚪ [HOLD] Model confidence ({prediction_probability:.2f}) within neutral band. No trade executed.\")

    def load_trade_logs(self):
        \"\"\"Load existing portfolio balance and history if available.\"\"\"
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, \"r\") as f:
                    data = json.load(f)
                    self.cash = data.get(\"account_balance\", self.cash)
                    self.portfolio = data.get(\"portfolio\", self.portfolio)
                    self.trade_history = data.get(\"history\", self.trade_history)
            except Exception as e:
                print(f\"Could not load existing logs: {e}\")

    def save_trade_logs(self):
        \"\"\"Persist updated trade logs to file.\"\"\"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, \"w\") as f:
            json.dump({
                \"account_balance\": round(self.cash, 2),
                \"portfolio\": self.portfolio,
                \"history\": self.trade_history
            }, f, indent=4)
        print(f\"Trade logs successfully saved to {self.log_path}\")
'''
with open('src/execution_engine/trade_executor.py', 'w') as f:
    f.write(code)
"