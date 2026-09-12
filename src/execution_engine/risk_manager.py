import numpy as np


class RiskManager:
    """Calculates position sizes, stop-loss, and take-profit targets based on account balance and risk limits."""

    def __init__(
        self,
        account_balance: float = 10000.0,
        risk_per_trade: float = 0.02,  # Risk 2% of account balance per trade
        stop_loss_pct: float = 0.015,   # 1.5% stop loss
        take_profit_pct: float = 0.03,  # 3.0% take profit (1:2 Risk-to-Reward)
    ):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def calculate_position_size(self, current_price: float) -> dict:
        """Calculate optimal trade units, stop loss price, and take profit price."""
        if current_price <= 0:
            raise ValueError("Current price must be greater than zero.")

        # Total capital allocated to risk per trade
        risk_amount = self.account_balance * self.risk_per_trade

        # Price thresholds
        stop_loss_price = current_price * (1 - self.stop_loss_pct)
        take_profit_price = current_price * (1 + self.take_profit_pct)

        # Risk amount per unit traded
        risk_per_unit = current_price - stop_loss_price

        # Max units allowed based on strict risk limit
        position_units = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

        return {
            "entry_price": round(current_price, 2),
            "units": round(position_units, 4),
            "total_position_value": round(position_units * current_price, 2),
            "stop_loss_price": round(stop_loss_price, 2),
            "take_profit_price": round(take_profit_price, 2),
            "risk_amount": round(risk_amount, 2),
        }