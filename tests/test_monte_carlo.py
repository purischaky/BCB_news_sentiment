import pytest
import pandas as pd
from src.monte_carlo import MonteCarloRiskEngine, SMEContractParameters, generate_sme_risk_matrix_report

def test_volatility_multiplier_calculation():
    """Verify that hawkish signals correctly scale market volatility."""
    engine = MonteCarloRiskEngine()
    
    # Neutral/Dovish statement should not boost volatility
    assert engine.calculate_hawkish_volatility_multiplier(0.80, is_hawkish=False) == 1.0
    
    # High Hawkish intensity (0.90) should boost volatility by 45% (1 + 0.90 * 0.50)
    assert engine.calculate_hawkish_volatility_multiplier(0.90, is_hawkish=True) == 1.45

def test_monte_carlo_simulation_outputs():
    """Verify Monte Carlo simulation results stick to risk & mathematical bounds."""
    engine = MonteCarloRiskEngine(random_seed=123)
    params = SMEContractParameters(
        contract_amount_usd=50_000.0,
        horizon_days=30,
        spot_rate=10.00,
        base_daily_volatility=0.01
    )
    
    res = engine.simulate_contract_hedge(params, nlp_intensity_score=0.75, is_hawkish=True)

    # 95% VaR hedging rate must strictly exceed the initial spot rate for long FX risk
    assert res["var_95_hedging_rate"] > res["spot_rate"]
    assert res["var_99_hedging_rate"] >= res["var_95_hedging_rate"]
    assert res["required_cash_reserve_buffer_bs"] > 0
    assert res["regulatory_framework"] == "RM No. 245/2026 & BCB RD No. 142/2026"

def test_generate_sme_risk_matrix_report(tmp_path, monkeypatch):
    """Test that the executive CSV report builds correctly."""
    monkeypatch.setattr("src.monte_carlo.OUTPUTS_REPORTS_DIR", tmp_path)
    
    report_path = generate_sme_risk_matrix_report("test_report.csv")
    assert report_path.exists()
    
    df = pd.read_csv(report_path)
    assert len(df) == 6  # 3 horizons (30, 60, 90) x 2 scenarios
    assert "required_cash_reserve_buffer_bs" in df.columns