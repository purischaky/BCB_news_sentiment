import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication-quality style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

# Configuration & Paths 
DATA_PROCESSED_PATH = Path("data/processed/processed_signals.json")
REPORT_PATH = Path("outputs/reports/sme_contract_risk_matrix.csv")
OUTPUTS_FIGURES_DIR = Path("outputs/figures")
OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_monte_carlo_density_distributions(
    spot_rate: float = 12.50,
    horizon_days: int = 60,
    base_daily_vol: float = 0.008,
    num_simulations: int = 10_000,
    output_filename: str = "monte_carlo_fx_density.png"
) -> Path:
    """
    Generates a KDE density plot comparing FX rate distributions under 
    Baseline (Neutral) vs. Hawkish Policy Shock scenarios.
    """
    np.random.seed(42)

    # Scenarios setup
    vol_neutral = base_daily_vol * 1.0
    vol_hawkish = base_daily_vol * 1.44  # ~0.88 hawkish score multiplier

    # Simulate terminal exchange rate distributions (GBM)
    returns_neutral = np.random.normal(0, vol_neutral * np.sqrt(horizon_days), num_simulations)
    returns_hawkish = np.random.normal(0, vol_hawkish * np.sqrt(horizon_days), num_simulations)

    rates_neutral = spot_rate * np.exp(returns_neutral)
    rates_hawkish = spot_rate * np.exp(returns_hawkish)

    var95_neutral = np.percentile(rates_neutral, 95)
    var95_hawkish = np.percentile(rates_hawkish, 95)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(rates_neutral, ax=ax, label="Baseline (Neutral / Routine)", color="#2b5c8f", fill=True, alpha=0.3)
    sns.kdeplot(rates_hawkish, ax=ax, label="Hawkish Policy Shock (High FX Control Risk)", color="#d95f02", fill=True, alpha=0.3)

    # Value at Risk Vertical Lines
    ax.axvline(var95_neutral, color="#2b5c8f", linestyle="--", linewidth=1.8, label=f"Baseline 95% VaR: {var95_neutral:.2f} Bs/USD")
    ax.axvline(var95_hawkish, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Hawkish 95% VaR: {var95_hawkish:.2f} Bs/USD")

    ax.set_title(f"Monte Carlo Exchange Rate Density Projection ({horizon_days}-Day Horizon)\nRegulatory Context: RM 245/2026 & BCB RD 142/2026", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Projected Parallel Exchange Rate (Bs / USD)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(loc="upper right", frameon=True)

    output_path = OUTPUTS_FIGURES_DIR / output_filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  ✓ Saved density plot to: {output_path}")
    return output_path

def plot_nlp_signal_distribution(
    input_path: Path = DATA_PROCESSED_PATH,
    output_filename: str = "nlp_signal_distribution.png"
) -> Path:
    """
    Generates a breakdown plot of BCB communications classified by policy stance.
    """
    if not input_path.exists():
        print(f"  ⚠ Skipping NLP plot: {input_path} not found.")
        return input_path

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    stance_counts = df["stance_label"].value_counts().reset_index()
    stance_counts.columns = ["stance_label", "count"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = sns.barplot(
        data=stance_counts,
        y="stance_label",
        x="count",
        palette="Blues_r",
        ax=ax
    )

    ax.set_title("Distribution of BCB Official Statements by Stance Class", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Statements Processed", fontsize=11)
    ax.set_ylabel("Policy Stance Category", fontsize=11)

    for bar in bars.patches:
        width = bar.get_width()
        ax.annotate(
            f"{int(width)}",
            (width, bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=10
        )

    output_path = OUTPUTS_FIGURES_DIR / output_filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  ✓ Saved NLP signal breakdown to: {output_path}")
    return output_path

def run_visualizations():
    """Main execution function to generate all executive visual figures."""
    print("=" * 50)
    print("Generating Executive Visualizations & Charts")
    print("=" * 50)

    print("\n1. Generating Monte Carlo FX Density Plot...")
    plot_monte_carlo_density_distributions()

    print("\n2. Generating NLP Stance Distribution Chart...")
    plot_nlp_signal_distribution()

    print("\n✓ Visual Analytics Generation Complete!")
    print("=" * 50)

if __name__ == "__main__":
    run_visualizations()