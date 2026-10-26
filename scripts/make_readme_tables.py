from pathlib import Path

import pandas as pd

REPORTS = Path("outputs/reports")
OUTPUT = REPORTS / "readme_tables.md"

def to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, separator, *rows])

def risk_table() -> str:
    df = pd.read_csv(REPORTS / "sme_contract_risk_matrix.csv")
    out = pd.DataFrame({
        "Scenario": df["scenario_name"],
        "Horizon (business days)": df["horizon_days"],
        "Daily volatility": (df["daily_volatility"] * 100).round(2).astype(str) + "%",
        "95% VaR rate (Bs/USD)": df["var_95_rate"].round(2),
        "Required buffer (Bs)": df["required_cash_buffer_bs"].map("{:,.0f}".format),
    })
    return to_markdown(out)

def robustness_table() -> str:
    df = pd.read_csv(REPORTS / "robustness_summary.csv")
    out = pd.DataFrame({
        "Sample": df["sample"],
        "n": df["n"],
        "Pearson r": df["pearson"],
        "95% CI": "[" + df["ci95_low"].astype(str) + ", " + df["ci95_high"].astype(str) + "]",
        "Permutation p": df["perm_p"],
    })
    return to_markdown(out)

if __name__ == "__main__":
    text = (
        "## SME risk matrix (USD 100,000 contract)\n\n" + risk_table()
        + "\n\n## Robustness of the NLP-volatility correlation\n\n" + robustness_table() + "\n"
    )
    OUTPUT.write_text(text, encoding="utf-8")
    print(text)