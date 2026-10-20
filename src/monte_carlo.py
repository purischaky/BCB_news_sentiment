import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

# --- Paths & Configuration ---
OUTPUTS_REPORTS_DIR = Path("outputs/reports")
OUTPUTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class SMEContractParameters:
    """Dataclass holding SME import contract parameters under RM 245/2026 & RD 142/2026."""
    contract_amount_usd: float
    horizon_days: int
    spot_rate: float
    base_daily_volatility: float

class MonteCarloRiskEngine:
    """
    Simulates exchange rate paths for SME import contracts under Bolivia's flexible exchange 
    rate regime (Ministerial Resolution No. 245/2026 and BCB Board Resolution No. 142/2026).
    Applies volatility shocks driven by central bank NLP policy stance scores.
    """

    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)

    def calculate_hawkish_volatility_multiplier(self, intensity_score: float, is_hawkish: bool) -> float:
        """
        Calculates a volatility multiplier based on the NLP hawkish signal.
        Under the flexible regime, high hawkish intensity (e.g., currency controls or 
        fx access restrictions) induces hoarding behavior and spikes market volatility.
        """
        if not is_hawkish:
            return 1.0  # Baseline market volatility
        
        # Scaling factor: High-intensity hawkish signals boost daily volatility by up to 50%
        return 1.0 + (intensity_score * 0.50)

    def simulate_contract_hedge(
        self, 
        params: SMEContractParameters, 
        nlp_intensity_score: float = 0.0,
        is_hawkish: bool = False,
        num_simulations: int = 10_000
    ) -> Dict[str, Any]:
        """
        Executes a Geometric Brownian Motion (GBM) Monte Carlo simulation to project future 
        exchange rate distributions and compute VaR-based hedging buffers for SMEs.
        """
        vol_multiplier = self.calculate_hawkish_volatility_multiplier(nlp_intensity_score, is_hawkish)
        adjusted_daily_vol = params.base_daily_volatility * vol_multiplier
        
        # Convert daily volatility to the simulation horizon scale
        # Assuming minor drift (mu = 0) under neutral market expectation
        dt = 1  # 1 day steps
        daily_returns = np.random.normal(
            loc=0.0, 
            scale=adjusted_daily_vol * np.sqrt(dt), 
            size=(params.horizon_days, num_simulations)
        )

        # Calculate exchange rate trajectories: S_t = S_0 * exp(sum(r_t))
        price_trajectories = params.spot_rate * np.exp(np.cumsum(daily_returns, axis=0))
        terminal_rates = price_trajectories[-1, :]

        # Risk metrics computation
        mean_terminal_rate = float(np.mean(terminal_rates))
        var_95_rate = float(np.percentile(terminal_rates, 95))   # 95% Value at Risk (Upper bound)
        var_99_rate = float(np.percentile(terminal_rates, 99))   # 99% Value at Risk (Extreme shock)
        
        # Cost of contract under baseline vs 95% VaR hedge
        nominal_cost_bs = params.contract_amount_usd * params.spot_rate
        hedged_cost_var95_bs = params.contract_amount_usd * var_95_rate
        required_hedge_buffer_bs = hedged_cost_var95_bs - nominal_cost_bs

        return {
            "regulatory_framework": "RM No. 245/2026 & BCB RD No. 142/2026",
            "horizon_days": params.horizon_days,
            "contract_amount_usd": params.contract_amount_usd,
            "spot_rate": params.spot_rate,
            "nlp_hawkish_signal": is_hawkish,
            "nlp_intensity_score": nlp_intensity_score,
            "volatility_multiplier": round(vol_multiplier, 3),
            "adjusted_daily_volatility": round(adjusted_daily_vol, 5),
            "expected_mean_rate": round(mean_terminal_rate, 4),
            "var_95_hedging_rate": round(var_95_rate, 4),
            "var_99_hedging_rate": round(var_99_rate, 4),
            "nominal_contract_cost_bs": round(nominal_cost_bs, 2),
            "recommended_hedged_cost_bs": round(hedged_cost_var95_bs, 2),
            "required_cash_reserve_buffer_bs": round(required_hedge_buffer_bs, 2)
        }

def generate_sme_risk_matrix_report(output_filename: str = "sme_contract_risk_matrix.csv") -> Path:
    """
    Generates a multi-horizon (30, 60, 90 days) risk matrix report comparing 
    baseline vs. post-Hawkish signal scenarios for a standard $100,000 USD SME import contract.
    """
    engine = MonteCarloRiskEngine()
    contract_usd = 100_000.0
    spot_rate = 12.50  # Parallel / reference market rate
    base_daily_vol = 0.008  # ~0.8% daily volatility under floating regime

    results = []
    horizons = [30, 60, 90]
    scenarios = [
        {"name": "Baseline (Neutral/Routine)", "score": 0.10, "hawkish": False},
        {"name": "Post-Policy Statement (High Hawkish Signal)", "score": 0.88, "hawkish": True}
    ]

    for horizon in horizons:
        for sc in scenarios:
            params = SMEContractParameters(
                contract_amount_usd=contract_usd,
                horizon_days=horizon,
                spot_rate=spot_rate,
                base_daily_volatility=base_daily_vol
            )
            sim = engine.simulate_contract_hedge(
                params=params, 
                nlp_intensity_score=sc["score"], 
                is_hawkish=sc["hawkish"]
            )
            sim["scenario_name"] = sc["name"]
            results.append(sim)

    df_report = pd.DataFrame(results)
    
    # Reorder columns for executive presentation
    cols_order = [
        "scenario_name", "horizon_days", "contract_amount_usd", "spot_rate",
        "nlp_hawkish_signal", "volatility_multiplier", "var_95_hedging_rate",
        "recommended_hedged_cost_bs", "required_cash_reserve_buffer_bs"
    ]
    df_report = df_report[cols_order]

    output_path = OUTPUTS_REPORTS_DIR / output_filename
    df_report.to_csv(output_path, index=False)
    print(f"✓ Executive SME Risk Matrix saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_sme_risk_matrix_report()