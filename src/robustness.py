from pathlib import Path

import numpy as np
import pandas as pd

from src.stats_utils import fisher_ci, permutation_pvalue

BACKTEST_PATH = Path("outputs/reports/nlp_fx_backtest_summary.csv")
REPORT_PATH = Path("outputs/reports/robustness_summary.csv")

# Chosen AFTER inspecting the FX plot, so this is a sensitivity check, not a primary test.
NOISY_START, NOISY_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2026-01-31")

def leave_one_out_range(df: pd.DataFrame) -> tuple[float, float]:
    """Min and max Pearson r when each event is dropped in turn."""
    rs = [
        df.drop(i)["intensity_score"].corr(df.drop(i)["log_vol_ratio"])
        for i in df.index
    ]
    return float(min(rs)), float(max(rs))

def summarize(label: str, df: pd.DataFrame) -> dict:
    n = len(df)
    if n < 6:
        return {"sample": label, "n": n}
    x, y = df["intensity_score"], df["log_vol_ratio"]
    r = x.corr(y)
    low, high = fisher_ci(r, n)
    loo_min, loo_max = leave_one_out_range(df)
    return {
        "sample": label,
        "n": n,
        "pearson": round(r, 3),
        "ci95_low": round(low, 3),
        "ci95_high": round(high, 3),
        "spearman": round(x.corr(y, method="spearman"), 3),
        "perm_p": round(permutation_pvalue(x, y), 3),
        "loo_min": round(loo_min, 3),
        "loo_max": round(loo_max, 3),
    }

def run_robustness(path: Path = BACKTEST_PATH, output_path: Path = REPORT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).dropna(subset=["log_vol_ratio"])
    in_noisy = df["date"].between(NOISY_START, NOISY_END)

    samples = {
        "full sample": df,
        "before flexible regime": df[df["policy_regime"] == "before_flexible"],
        "flexible regime": df[df["policy_regime"] == "flexible"],
        "excluding Nov25-Jan26": df[~in_noisy],
    }
    table = pd.DataFrame([summarize(k, v) for k, v in samples.items()])
    print(table.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table

if __name__ == "__main__":
    run_robustness()