from pathlib import Path

import numpy as np
import pandas as pd

# Verify this date against the text of RM 245/2026 before publishing.
FLEXIBLE_REGIME_START = pd.Timestamp("2026-06-29")

def load_fx_series(path: Path | str = "data/raw/official_exchange_rate.csv") -> pd.DataFrame:
    """
    Load the BCB exchange-rate series with consistent rules for the whole project:
    - one row per business day (weekends in the source are copies of Friday),
    - log returns computed between consecutive business days,
    - a 'policy_regime' label so analyses can be split by regime.
    """
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.drop_duplicates(subset="date", keep="last")
    df = df[df["date"].dt.dayofweek < 5].set_index("date")

    df["log_return"] = np.log(df["value"]).diff()
    df["policy_regime"] = np.where(
        df.index >= FLEXIBLE_REGIME_START, "flexible", "before_flexible"
    )
    return df