import numpy as np
import pandas as pd
import pytest

from src.fx_utils import FLEXIBLE_REGIME_START
from src.monte_carlo import (
    MonteCarloRiskEngine,
    SMEContractParameters,
    estimate_flexible_regime_inputs,
    generate_sme_risk_matrix_report,
)

def make_fx_csv(tmp_path, pre_vol=0.001, post_vol=0.02, start="2026-05-01", periods=120):
    """Synthetic FX file: calm before the flexible regime, volatile after."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(start, periods=periods)
    vols = np.where(dates < FLEXIBLE_REGIME_START, pre_vol, post_vol)
    values = 11 * np.exp(np.cumsum(rng.normal(0, 1, periods) * vols))
    path = tmp_path / "fx.csv"
    pd.DataFrame({"date": dates, "value": values}).to_csv(path, index=False)
    return path

def make_params(daily_vol=0.01, horizon=30):
    return SMEContractParameters(
        contract_amount_usd=50_000.0, horizon_days=horizon, spot_rate=10.0, base_daily_volatility=daily_vol
    )

def test_simulation_outputs_respect_risk_ordering():
    res = MonteCarloRiskEngine(random_seed=123).simulate_contract_hedge(make_params())
    assert res["var_95_rate"] > res["spot_rate"]
    assert res["var_99_rate"] >= res["var_95_rate"]
    assert res["required_cash_buffer_bs"] > 0

def test_same_inputs_give_identical_results():
    engine = MonteCarloRiskEngine(random_seed=7)
    assert engine.simulate_contract_hedge(make_params()) == engine.simulate_contract_hedge(make_params())

def test_higher_volatility_multiplier_raises_var():
    engine = MonteCarloRiskEngine()
    base = engine.simulate_contract_hedge(make_params(), volatility_multiplier=1.0)
    stress = engine.simulate_contract_hedge(make_params(), volatility_multiplier=2.0)
    assert stress["var_95_rate"] > base["var_95_rate"]

def test_inputs_are_estimated_from_flexible_regime_only(tmp_path):
    path = make_fx_csv(tmp_path, pre_vol=0.001, post_vol=0.02)
    spot, daily_vol = estimate_flexible_regime_inputs(path)
    assert 0.015 < daily_vol < 0.025  # close to 0.02, not diluted by the calm pre-regime period
    assert spot == pytest.approx(pd.read_csv(path)["value"].iloc[-1])

def test_estimation_fails_without_enough_flexible_data(tmp_path):
    path = make_fx_csv(tmp_path, start="2026-01-05", periods=40)  # ends before 29 June
    with pytest.raises(ValueError):
        estimate_flexible_regime_inputs(path)

def test_generate_sme_risk_matrix_report(tmp_path, monkeypatch):
    monkeypatch.setattr("src.monte_carlo.OUTPUTS_REPORTS_DIR", tmp_path)
    fx_path = make_fx_csv(tmp_path)

    report_path = generate_sme_risk_matrix_report("test_report.csv", fx_path=fx_path)

    df = pd.read_csv(report_path)
    assert len(df) == 9  # 3 horizons x 3 scenarios
    for _, group in df.groupby("horizon_days"):
        buffers = group.sort_values("volatility_multiplier")["required_cash_buffer_bs"].tolist()
        assert buffers == sorted(buffers)  # more volatility, bigger buffer