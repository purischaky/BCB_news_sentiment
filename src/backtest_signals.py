import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.fx_utils import FLEXIBLE_REGIME_START, load_fx_series

SIGNALS_PATH = Path("data/processed/processed_signals.json")
FX_PATH = Path("data/raw/official_exchange_rate.csv")
OUTPUT_PATH = Path("outputs/reports/nlp_fx_backtest_summary.csv")

PRE_DAYS = 20   # business-day returns ending on the event date
POST_DAYS = 5   # business-day returns strictly after the event date

def compute_event_volatility(fx: pd.DataFrame, event_date, pre_days: int = PRE_DAYS, post_days: int = POST_DAYS):
    """
    Volatility of returns before vs. after an event.
    - pre window: the last `pre_days` returns up to and including the event date
    - post window: the next `post_days` returns strictly after the event date
    Returns None unless both windows are complete.
    """
    returns = fx["log_return"].dropna()
    event_date = pd.Timestamp(event_date)

    pre = returns[returns.index <= event_date].tail(pre_days)
    post = returns[returns.index > event_date].head(post_days)
    if len(pre) < pre_days or len(post) < post_days:
        return None

    pre_vol, post_vol = pre.std(), post.std()
    ratio = np.log(post_vol / pre_vol) if pre_vol > 0 and post_vol > 0 else np.nan
    return {"pre_vol": pre_vol, "post_vol": post_vol, "log_vol_ratio": ratio}

def minimum_detectable_correlation(n: int) -> float:
    """Smallest |r| detectable with 80% power at the 5% level (Fisher z approximation)."""
    return float(np.tanh((1.96 + 0.84) / np.sqrt(n - 3)))

def run_signal_backtest(
    signals_path: Path = SIGNALS_PATH, fx_path: Path = FX_PATH, output_path: Path = OUTPUT_PATH
):
    print("Running event-study backtest: NLP signal vs. change in FX volatility...")
    if not Path(signals_path).exists() or not Path(fx_path).exists():
        print("Required data files missing. Run data collection and classification first.")
        return

    fx = load_fx_series(fx_path)
    with open(signals_path, "r", encoding="utf-8") as f:
        signals = pd.DataFrame(json.load(f))
    signals["date"] = pd.to_datetime(signals["date"])

    # One observation per date. Several documents on the same date count once;
    # the date's score is the maximum score among its documents.
    events = signals.groupby("date").agg(
        intensity_score=("intensity_score", "max"),
        is_hawkish=("is_hawkish", "max"),
        n_documents=("title", "count"),
    ).reset_index()

    rows = []
    for _, ev in events.iterrows():
        vol = compute_event_volatility(fx, ev["date"])
        if vol is None:
            continue
        regime = "flexible" if ev["date"] >= FLEXIBLE_REGIME_START else "before_flexible"
        rows.append({**ev.to_dict(), **vol, "policy_regime": regime})

    df = pd.DataFrame(rows).dropna(subset=["log_vol_ratio"])
    if df.empty:
        print("Not enough overlapping data.")
        return

    n = len(df)
    pearson = df["intensity_score"].corr(df["log_vol_ratio"])
    spearman = df["intensity_score"].corr(df["log_vol_ratio"], method="spearman")

    print(f"\n  Event dates used (n): {n}   (of {len(events)} distinct dates)")
    print(f"  Pearson r : {pearson:.4f}")
    print(f"  Spearman  : {spearman:.4f}")
    print(f"  Smallest |r| this sample could reliably detect: {minimum_detectable_correlation(n):.2f}")
    print(df["policy_regime"].value_counts().to_string())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

if __name__ == "__main__":
    run_signal_backtest()