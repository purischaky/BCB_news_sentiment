from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.fx_utils import load_fx_series

OUTPUTS_REPORTS_DIR = Path("outputs/reports")
FX_PATH = Path("data/raw/official_exchange_rate.csv")

# Illustrative volatility stress levels. They are NOT estimated from the NLP signal.
STRESS_SCENARIOS = [
    ("Baseline (observed flexible-regime volatility)", 1.0),
    ("Stress x1.5 (illustrative)", 1.5),
    ("Stress x2.0 (illustrative)", 2.0),
]

@dataclass
class SMEContractParameters:
    contract_amount_usd: float
    horizon_days: int
    spot_rate: float
    base_daily_volatility: float

def estimate_flexible_regime_inputs(fx_path: Path | str = FX_PATH) -> tuple[float, float]:
    """Spot = last observed value. Daily volatility = std of log returns since the flexible regime began."""
    fx = load_fx_series(fx_path)
    returns = fx.loc[fx["policy_regime"] == "flexible", "log_return"].dropna()
    if len(returns) < 10:
        raise ValueError("Not enough flexible-regime observations to estimate volatility.")
    return float(fx["value"].iloc[-1]), float(returns.std())

class MonteCarloRiskEngine:
    """
    Zero-drift geometric Brownian motion for the exchange rate.
    Each call re-seeds its generator, so identical inputs always give identical draws.
    This makes the CSV report and the plots agree exactly.
    """
    def __init__(self, random_seed: int = 42):
        self.seed = random_seed

    def simulate_terminal_rates(
        self, params: SMEContractParameters, volatility_multiplier: float = 1.0, num_simulations: int = 10_000
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        daily_vol = params.base_daily_volatility * volatility_multiplier
        daily_returns = rng.normal(0.0, daily_vol, size=(params.horizon_days, num_simulations))
        return params.spot_rate * np.exp(daily_returns.sum(axis=0))

    def simulate_contract_hedge(
        self, params: SMEContractParameters, volatility_multiplier: float = 1.0, num_simulations: int = 10_000
    ) -> Dict[str, Any]:
        terminal = self.simulate_terminal_rates(params, volatility_multiplier, num_simulations)
        var_95 = float(np.percentile(terminal, 95))
        var_99 = float(np.percentile(terminal, 99))

        nominal_cost = params.contract_amount_usd * params.spot_rate
        hedged_cost = params.contract_amount_usd * var_95
        return {
            "horizon_days": params.horizon_days,
            "contract_amount_usd": params.contract_amount_usd,
            "spot_rate": round(params.spot_rate, 4),
            "daily_volatility": round(params.base_daily_volatility * volatility_multiplier, 5),
            "volatility_multiplier": volatility_multiplier,
            "var_95_rate": round(var_95, 4),
            "var_99_rate": round(var_99, 4),
            "hedged_cost_var95_bs": round(hedged_cost, 2),
            "required_cash_buffer_bs": round(hedged_cost - nominal_cost, 2),
        }

def generate_sme_risk_matrix_report(
    output_filename: str = "sme_contract_risk_matrix.csv",
    fx_path: Path | str = FX_PATH,
    contract_usd: float = 100_000.0,
    horizons: tuple = (30, 60, 90),
) -> Path:
    spot, daily_vol = estimate_flexible_regime_inputs(fx_path)
    print(f"Calibration: spot = {spot:.2f} Bs/USD, flexible-regime daily volatility = {daily_vol:.4f}")

    engine = MonteCarloRiskEngine()
    rows = []
    for horizon in horizons:
        params = SMEContractParameters(contract_usd, horizon, spot, daily_vol)
        for name, multiplier in STRESS_SCENARIOS:
            result = engine.simulate_contract_hedge(params, multiplier)
            result["scenario_name"] = name
            rows.append(result)

    columns = [
        "scenario_name", "horizon_days", "contract_amount_usd", "spot_rate", "daily_volatility",
        "volatility_multiplier", "var_95_rate", "var_99_rate", "hedged_cost_var95_bs", "required_cash_buffer_bs",
    ]
    OUTPUTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_REPORTS_DIR / output_filename
    pd.DataFrame(rows)[columns].to_csv(output_path, index=False)
    print(f"Risk matrix saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_sme_risk_matrix_report()