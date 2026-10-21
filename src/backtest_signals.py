import json
from pathlib import Path
import pandas as pd
import numpy as np

def run_signal_backtest():
    print("Running Econometric Backtest: NLP Stance vs. FX Volatility...")
    
    # Load processed signals and exchange rate data
    signals_path = Path("data/processed/processed_signals.json")
    fx_path = Path("data/raw/official_exchange_rate.csv")
    
    if not signals_path.exists() or not fx_path.exists():
        print("⚠ Required data files missing. Run data collection and classification first.")
        return
        
    with open(signals_path, "r", encoding="utf-8") as f:
        signals = json.load(f)
        
    df_fx = pd.read_csv(fx_path)
    df_fx["date"] = pd.to_datetime(df_fx["date"])
    df_fx = df_fx.sort_values("date").set_index("date")
    
    # Calculate daily returns and rolling 5-day realized volatility
    df_fx["daily_return"] = df_fx["value"].pct_change()
    df_fx["realized_vol_5d"] = df_fx["daily_return"].rolling(window=5).std()
    
    # Merge signals with realized volatility over subsequent windows
    df_signals = pd.DataFrame(signals)
    df_signals["date"] = pd.to_datetime(df_signals["date"])
    
    merged_data = []
    for _, sig in df_signals.iterrows():
        sig_date = sig["date"]
        
        # Strict post-publication window: T+1 to T+5 trading days
        future_window = df_fx.loc[df_fx.index > sig_date].head(5)
        
        if len(future_window) >= 3:
            avg_future_vol = future_window["realized_vol_5d"].mean()
            merged_data.append({
                "date": sig_date,
                "title": sig["title"],
                "intensity_score": sig["intensity_score"],
                "is_hawkish": sig["is_hawkish"],
                "subsequent_fx_vol": avg_future_vol
            })
                
    df_backtest = pd.DataFrame(merged_data).dropna()
    
    if df_backtest.empty:
        print("⚠ Not enough overlapping date ranges found.")
        return

    # Handle duplicate releases on the same date by aggregating (e.g., mean intensity per date)
    df_distinct_events = df_backtest.groupby("date").agg({
        "intensity_score": "max",
        "is_hawkish": "max",
        "subsequent_fx_vol": "mean"
    }).reset_index()
    
    # Compute Pearson correlation to measure indicator strength
    correlation = df_distinct_events["intensity_score"].corr(df_distinct_events["subsequent_fx_vol"])
    print(f"\n✓ Revised Econometric Link Established:")
    print(f"  - Total Raw Records: {len(df_backtest)}")
    print(f"  - Distinct Event Dates: {len(df_distinct_events)}")
    print(f"  - Pearson Correlation (Distinct Events): {correlation:.4f}")
    
    # Save backtest report
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "nlp_fx_backtest_summary.csv"
    df_distinct_events.to_csv(report_path, index=False)
    print(f"  - Saved cleaned backtest summary to: {report_path}")

if __name__ == "__main__":
    run_signal_backtest()