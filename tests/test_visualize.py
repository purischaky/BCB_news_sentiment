import matplotlib

matplotlib.use("Agg")  # draw without opening a window

import numpy as np
import pandas as pd

from src.visualize_results import plot_nlp_vs_fx_volatility

def test_correlation_plot_runs_on_backtest_output_columns(tmp_path, monkeypatch):
    monkeypatch.setattr("src.visualize_results.OUTPUTS_FIGURES_DIR", tmp_path)

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "intensity_score": rng.uniform(0.3, 1.0, 20),
        "log_vol_ratio": rng.normal(0, 0.5, 20),
    })
    backtest_path = tmp_path / "backtest.csv"
    df.to_csv(backtest_path, index=False)

    output = plot_nlp_vs_fx_volatility(backtest_path, "test_plot.png")

    assert output.exists()