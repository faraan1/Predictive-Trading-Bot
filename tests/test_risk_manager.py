import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest
from src.execution_engine.risk_manager import RiskManager


def test_position_sizing_calculation():
    """Verify position sizing calculations and risk boundary outputs."""
    manager = RiskManager(account_balance=10000.0, risk_per_trade=0.02, stop_loss_pct=0.02, take_profit_pct=0.04)
    result = manager.calculate_position_size(current_price=100.0)

    # 2% of $10,000 = $200 risk. Stop loss is $2 per unit. Position size = 100 units ($10,000 total position value).
    assert result["entry_price"] == 100.0
    assert result["stop_loss_price"] == 98.0
    assert result["take_profit_price"] == 104.0
    assert result["risk_amount"] == 200.0
    assert result["units"] == 100.0