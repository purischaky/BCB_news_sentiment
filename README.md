## Central Bank Policy NLP & Exchange Rate Risk Engine

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Executive Overview & Business Problem
Under Bolivia's flexible exchange rate framework (Ministerial Resolution No. 245/2026 & BCB Board Resolution No. 142/2026), businesses importing goods under U.S. dollar contracts face foreign exchange risk. This repository provides an exploratory quantitative research pipeline designed to evaluate two core business questions:

1. Signaling Power: How strong of an indicator is an official central bank policy communication regarding subsequent shifts in exchange rate volatility?

2. Contract Risk Translation: How can small and medium enterprises (SMEs) translate central bank policy signals and market uncertainty into Value-at-Risk (VaR) cash reserve buffers for USD-denominated contracts?

## Architecture & Data Pipeline
```text
├── data/
│   ├── raw/                 # Raw BCB communications (JSON/PDF) & official exchange rate CSV
│   └── processed/           # Zero-shot NLP classified signals dataset
├── outputs/
│   ├── figures/             # Distribution plots, backtest regressions, and density charts
│   └── reports/             # Risk matrix scenarios and backtest summary CSVs
├── src/
│   ├── add_manual_pdf.py    # Batch processing utility for official announcement PDFs
│   ├── data_collection.py   # Resilient BCB web scraper & exchange rate historical downloader
│   ├── signal_classifier.py # Zero-Shot NLP classifier using mDeBERTa-v3
│   ├── backtest_signals.py  # Econometric event backtest (NLP score vs. trading-day volatility)
│   ├── monte_carlo.py       # Illustrative Geometric Brownian Motion (GBM) risk simulation
│   └── visualize_results.py # Publication-style figures using Matplotlib and Seaborn
└── tests/                   # Pytest suite verifying pipeline logic and numerical bounds
```

## Methodology & Model Mechanics

### 1. Zero-Shot NLP Classification (src/signal_classifier.py)
Official communications are evaluated using Hugging Face's MoritzLaurer/mDeBERTa-v3-base-mnli-xnli model across candidate policy labels:
    - restricción de divisas o control de cambios
    - ajuste de política monetaria o régimen cambiario
    - estabilidad de reservas y liquidez
    - comunicado administrativo o rutinario

Classifier Mechanics: Documents are processed in chunked segments to fit model context windows. The classifier identifies the top-ranked label and its associated confidence score (intensity_score). A document is classified as hawkish / restrictive (is_hawkish = True) if the top score is $\ge 0.50$ and the top label explicitly contains restriction or adjustment keywords (restricción or ajuste).

### 2. Econometric Backtesting (src/backtest_signals.py)
To test the empirical relationship between policy statements and exchange rate dynamics, the backtesting module measures realized volatility over strict post-publication trading days ($T+1$ to $T+5$).
    - Event Aggregation: When multiple press releases occur on the same calendar date, events are aggregated at the distinct date level to prevent duplicate observation bias.
    - Empirical Findings: Across distinct communication event dates, the Pearson correlation between NLP hawkish intensity scores and subsequent 5-day trading volatility was -0.0740.
    - Economic Interpretation: Official central bank statements operate primarily as administrative guidance rather than immediate short-term market volatility shocks.

### 3. Scenario-Based Risk Simulation (src/monte_carlo.py)
Because empirical backtesting shows near-zero short-term volatility correlation, the Monte Carlo simulation engine is treated strictly as an illustrative scenario analysis rather than a fitted predictive model.
    - Simulation Framework: Uses Geometric Brownian Motion (GBM) over 30, 60, and 90-day horizons for a standard $100,000 USD SME import contract.
    - Scenario Uplift: Compares a baseline neutral volatility scenario against a hypothetical policy shock scenario (applying an illustrative +44% volatility multiplier) to compute 95% and 99% Value-at-Risk (VaR) cash reserve buffers.

## Empirical Results & Backtesting

1. Signaling Power (NLP Stance vs. FX Volatility)
- Pearson Correlation: $r = -0.0740$
- Interpretation: The empirical correlation between central bank hawkish intensity scores and subsequent 5-day exchange rate volatility is near zero. This indicates that official central bank statements in managed flexible regimes act primarily as administrative guidance and anchoring signals rather than drivers of immediate short-term market shocks.

2. SME Executive Risk Matrix ($100,000 USD Import Contract)Below is an excerpt from outputs/reports/sme_contract_risk_matrix.csv:

| Scenario Name | Horizon (Days) | Spot Rate (Bs/USD) | NLP Hawkish Signal | Volatility Multiplier | 95% VaR Rate (Bs/USD) | Recommended Hedged Cost (Bs) | Required Reserve Buffer (Bs) | 
|-------------|-------|-------|------|------|---------|------|---------|
| Baseline (Neutral) | 30 | 12.50 | False | 1.000 | 12.82 | 1,282,000.00 | 32,000.00 |
| Post-Policy Shock | 30 | 12.50 | True | 1.440 | 12.96 | 1,296,000.00 | 46,000.00 |
| Baseline (Neutral) |60 | 12.50 | False | 1.000 | 12.95 | 1,295,000.00 | 45,000.00 |
| Post-Policy Shock | 60 | 12.50 | True | 1.440 | 13.16 | 1,316,000.00 | 66,000.00 |
| Baseline (Neutral) | 90 | 12.50 |False | 1.000 | 13.06 | 1,306,000.00 | 56,000.00 |
| Post-Policy Shock | 90 | 12.50 | True | 1.440 | 13.31 | 1,331,000.00 | 81,000.00 |

* Key Takeaway: Under a high hawkish policy shock scenario over a 90-day settlement horizon, an SME importing $100,000 USD worth of goods must hold an additional Bs 81,000 in reserve above nominal cost (compared to Bs 56,000 under baseline conditions) to achieve 95% settlement certainty.

## Installation & Execution Guide
### 1. Environment Setup
Clone the repository and install requirements.

### 2. Running the Analytical Pipeline
Run the modular pipeline sequentially:
1. Scrape raw BCB press notes and download exchange rates
python src/data_collection.py
python src/add_manual_pdf.py
2. Run Zero-Shot NLP stance classification
python src/signal_classifier.py
3. Execute the econometric backtest
python src/backtest_signals.py
4. Generate Monte Carlo scenario simulations
python src/monte_carlo.py
5. Output figures and visualizations
python src/visualize_results.py

### 3. Running Unit Tests
Validate code functionality and numerical constraints:
pytest -v

## Engineering Approach & AI Disclosure
- Domain Modeling & System Architecture: Human engineering defined the core business problem, regulatory context (RM 245/2026 & RD 142/2026), quantitative scenario design, and backtest specifications.
- AI Collaboration: AI tools were utilized to assist in drafting boilerplate web scraping handlers, configuring Hugging Face zero-shot pipelines, refining Seaborn styling parameters, and structuring test coverage.
- Validation & Character Encoding: All source files, CSV reports, and Markdown documentations are encoded in UTF-8 without BOM to preserve proper rendering of Spanish characters and mathematical notation.

