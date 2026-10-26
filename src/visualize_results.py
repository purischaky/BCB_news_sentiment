import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.monte_carlo import MonteCarloRiskEngine, SMEContractParameters, estimate_flexible_regime_inputs

# Set publication-quality style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

# Configuration & Paths 
DATA_PROCESSED_PATH = Path("data/processed/processed_signals.json")
REPORT_PATH = Path("outputs/reports/sme_contract_risk_matrix.csv")
BACKTEST_PATH = Path("outputs/reports/nlp_fx_backtest_summary.csv")
OUTPUTS_FIGURES_DIR = Path("outputs/figures")
OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_monte_carlo_density_distributions(
    horizon_days: int = 60,
    stress_multiplier: float = 1.5,
    num_simulations: int = 10_000,
    output_filename: str = "monte_carlo_fx_density.png"
) -> Path:
    """Density of simulated exchange rates: baseline vs. an illustrative volatility stress."""
    spot, daily_vol = estimate_flexible_regime_inputs()
    params = SMEContractParameters(100_000.0, horizon_days, spot, daily_vol)
    engine = MonteCarloRiskEngine()

    rates_base = engine.simulate_terminal_rates(params, 1.0, num_simulations)
    rates_stress = engine.simulate_terminal_rates(params, stress_multiplier, num_simulations)
    var95_base = np.percentile(rates_base, 95)
    var95_stress = np.percentile(rates_stress, 95)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(rates_base, ax=ax, label="Baseline volatility", color="#2b5c8f", fill=True, alpha=0.3)
    sns.kdeplot(rates_stress, ax=ax, label=f"Stress x{stress_multiplier} volatility (illustrative)",
                color="#d95f02", fill=True, alpha=0.3)
    ax.axvline(var95_base, color="#2b5c8f", linestyle="--", linewidth=1.8,
               label=f"Baseline 95% VaR: {var95_base:.2f} Bs/USD")
    ax.axvline(var95_stress, color="#d95f02", linestyle="--", linewidth=1.8,
               label=f"Stress 95% VaR: {var95_stress:.2f} Bs/USD")

    ax.set_title(f"Simulated Exchange Rate Distribution ({horizon_days}-Day Horizon)\n"
                 f"Spot {spot:.2f}, daily volatility {daily_vol:.2%} (flexible-regime calibration)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Simulated exchange rate (Bs / USD)", fontsize=11)
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

def plot_nlp_vs_fx_volatility(
    input_path: Path = BACKTEST_PATH,
    output_filename: str = "nlp_vs_fx_correlation.png"
) -> Path:
    """
    Scatter of NLP classifier confidence vs. the change in FX volatility around each
    announcement, log(post-event vol / pre-event vol). Exploratory, not a predictive test.
    """
    if not input_path.exists():
        print(f"  ⚠ Skipping NLP vs FX correlation plot: {input_path} not found.")
        return input_path

    df = pd.read_csv(input_path)
    r = df["intensity_score"].corr(df["log_vol_ratio"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        data=df,
        x="intensity_score",
        y="log_vol_ratio",
        ax=ax,
        color="#2b5c8f",
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "#d95f02"}
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    ax.set_title(
        "NLP Classifier Confidence vs. Change in FX Volatility Around Announcements\n"
        f"Exploratory: n = {len(df)}, Pearson r = {r:.2f} (CI includes zero)",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.set_xlabel("NLP classifier confidence (max zero-shot score)", fontsize=10)
    ax.set_ylabel("log(post-event vol / pre-event vol)", fontsize=10)

    output_path = OUTPUTS_FIGURES_DIR / output_filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  ✓ Saved NLP vs FX correlation plot to: {output_path}")
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

    print("\n3. Generating NLP vs. FX Volatility Regression Plot...")
    plot_nlp_vs_fx_volatility()

    print("\n✓ Visual Analytics Generation Complete!")
    print("=" * 50)

if __name__ == "__main__":
    run_visualizations()