import numpy as np
import pandas as pd
import pytest
import json

from src.backtest_signals import compute_event_volatility, minimum_detectable_correlation, run_signal_backtest

def make_fx(returns, start="2026-08-03"):  # 3 Aug 2026 is a Monday
    index = pd.bdate_range(start, periods=len(returns))
    return pd.DataFrame({"log_return": returns}, index=index)

def random_returns(n=40, seed=0):
    return np.random.default_rng(seed).normal(0, 0.01, n)

def test_windows_use_exact_return_positions():
    r = random_returns()
    fx = make_fx(r)
    event = fx.index[24]  # a Friday

    out = compute_event_volatility(fx, event)

    assert out["pre_vol"] == pytest.approx(np.std(r[5:25], ddof=1))   # 20 returns ending at event
    assert out["post_vol"] == pytest.approx(np.std(r[25:30], ddof=1))  # the 5 returns after

def test_pre_event_shock_does_not_leak_into_post_window():
    r = np.zeros(40)
    r[20] = 0.5  # big move BEFORE the event
    fx = make_fx(r)

    out = compute_event_volatility(fx, fx.index[24])

    assert out["post_vol"] == 0.0
    assert out["pre_vol"] > 0

def test_returns_after_the_window_are_ignored():
    r = random_returns()
    r_changed = r.copy()
    r_changed[35] = 0.9  # a shock AFTER the 5-day window
    event = make_fx(r).index[24]

    a = compute_event_volatility(make_fx(r), event)
    b = compute_event_volatility(make_fx(r_changed), event)

    assert a["post_vol"] == pytest.approx(b["post_vol"])

def test_weekend_event_uses_the_following_business_days():
    fx = make_fx(random_returns())
    friday = fx.index[24]
    saturday = friday + pd.Timedelta(days=1)

    assert compute_event_volatility(fx, saturday)["post_vol"] == pytest.approx(
        compute_event_volatility(fx, friday)["post_vol"]
    )

def test_returns_none_when_post_window_is_incomplete():
    fx = make_fx(random_returns(n=28))  # only 3 returns after position 24
    assert compute_event_volatility(fx, fx.index[24]) is None

def test_minimum_detectable_correlation_n41():
    # ~0.43 for 80% power at 5% significance
    assert minimum_detectable_correlation(41) == pytest.approx(0.43, abs=0.01)
    
def test_run_signal_backtest_end_to_end(tmp_path):
    """Smoke test: the whole function runs on small synthetic data and writes its report."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2026-01-05", periods=80)
    values = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))

    fx_path = tmp_path / "fx.csv"
    pd.DataFrame({"date": dates, "value": values}).to_csv(fx_path, index=False)

    event_dates = dates[25:65:3]
    signals = [
        {
            "title": f"doc {i}",
            "date": d.strftime("%Y-%m-%d"),
            "intensity_score": float(rng.uniform(0.3, 1.0)),
            "is_hawkish": bool(i % 2),
        }
        for i, d in enumerate(event_dates)
    ]
    signals_path = tmp_path / "signals.json"
    signals_path.write_text(json.dumps(signals), encoding="utf-8")

    output_path = tmp_path / "out" / "summary.csv"
    run_signal_backtest(signals_path, fx_path, output_path)

    result = pd.read_csv(output_path)
    assert len(result) > 5
    assert {"log_vol_ratio", "policy_regime"} <= set(result.columns)