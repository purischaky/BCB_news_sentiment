import numpy as np
import pandas as pd
import pytest

from src.fx_utils import load_fx_series

def make_csv(tmp_path, rows):
    path = tmp_path / "fx.csv"
    pd.DataFrame(rows, columns=["date", "value"]).to_csv(path, index=False)
    return path

def test_weekends_are_removed(tmp_path):
    path = make_csv(tmp_path, [
        ("2026-10-09", 11.85),  # Friday
        ("2026-10-10", 11.73),  # Saturday
        ("2026-10-11", 11.73),  # Sunday
        ("2026-10-12", 11.73),  # Monday
    ])
    df = load_fx_series(path)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2026-10-09", "2026-10-12"]

def test_log_return_between_business_days(tmp_path):
    path = make_csv(tmp_path, [("2026-10-09", 10.0), ("2026-10-12", 11.0)])
    df = load_fx_series(path)
    assert df["log_return"].iloc[1] == pytest.approx(np.log(1.1))

def test_policy_regime_split(tmp_path):
    path = make_csv(tmp_path, [("2026-06-26", 10.0), ("2026-06-29", 10.2)])
    df = load_fx_series(path)
    assert df.loc["2026-06-26", "policy_regime"] == "before_flexible"
    assert df.loc["2026-06-29", "policy_regime"] == "flexible"