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
│   ├── raw/                 
│   └── processed/           
├── outputs/
│   ├── figures/             
│   └── reports/             
├── src/
└── tests/                   
```

## Methodology & Model Mechanics

### 1. Zero-Shot NLP Classification (src/signal_classifier.py)
Official communications are evaluated using Hugging Face's MoritzLaurer/mDeBERTa-v3-base-mnli-xnli model across candidate policy labels:
    - restricción de divisas o control de cambios
    - ajuste de política monetaria o régimen cambiario
    - estabilidad de reservas y liquidez
    - comunicado administrativo o rutinario

Classifier Mechanics: Documents are processed in chunked segments to fit model context windows. The classifier identifies the top-ranked label and its associated zero-shot confidence score. In the codebase, this metric is stored as the intensity_score field; however, it should be interpreted strictly as an exploratory policy-signal score (a relative confidence metric from the NLP model), not a calibrated macroeconomic measurement. A document is classified as hawkish / restrictive (is_hawkish = True) if this policy-signal score is $\ge 0.50$ and the top label explicitly contains restriction or adjustment keywords (restricción or ajuste).

### 2. Econometric Backtesting (src/backtest_signals.py)
To test the empirical relationship between policy statements and exchange rate dynamics, the backtesting module measures realized volatility over strict post-publication trading days ($T+1$ to $T+5$).
    - Event Aggregation: When multiple press releases occur on the same calendar date, events are aggregated at the distinct date level to prevent duplicate observation bias.
    - Empirical Findings: Across 41 event dates, the correlation between the classifier score and subsequent exchange-rate volatility was approximately −0.104. This is a weak association in this sample and should be treated as exploratory.
    - Economic Interpretation: Official central bank statements operate primarily as administrative guidance rather than immediate short-term market volatility shocks.

### 3. Scenario-Based Risk Simulation (src/monte_carlo.py)
Because empirical backtesting shows near-zero short-term volatility correlation, the Monte Carlo simulation engine is treated strictly as an illustrative scenario analysis rather than a fitted predictive model.
    - Simulation Framework: Uses Geometric Brownian Motion (GBM) over 30, 60, and 90-day horizons for a standard $100,000 USD SME import contract.
    - Scenario Uplift: Compares a baseline neutral volatility scenario against a hypothetical policy shock scenario (applying an illustrative +44% volatility multiplier) to compute 95% and 99% Value-at-Risk (VaR) cash reserve buffers.

## Empirical Results & Backtesting

1. Signaling Power (NLP Stance vs. FX Volatility)
- Pearson Correlation: $r \approx -0.104$ (across 41 distinct event dates)
- Interpretation: The empirical correlation between the NLP-derived policy-signal scores (intensity_score) and subsequent 5-day exchange rate volatility is near zero. This indicates that official central bank statements in managed flexible regimes act primarily as administrative guidance and anchoring signals rather than drivers of immediate short-term market shocks.

2. SME Executive Risk Matrix ($100,000 USD Import Contract)Below is an excerpt from outputs/reports/sme_contract_risk_matrix.csv:

| Scenario Name | Horizon (Days) | Contract Amount (USD) | Spot Rate | Hawkish Signal | Volatility Multiplier | 95% VaR Rate | Recommended Hedged Cost (Bs) | Required Cash Reserve Buffer (Bs) |
|-------|--------|-------|----|-----|----|----|-----|----| 
| Baseline (Neutral/Routine) | 30 | $100,000 | 12.50 | False | 1.00 | 13.4293 | 1,342,929.90 | 92,929.90 |
| Post-Policy Statement (High Hawkish Signal) | 30 | $100,000 | 12.50 | True | 1.44 | 13.8649 | 1,386,485.66 | 136,485.66 |
| Baseline (Neutral/Routine) | 60 | $100,000 | 12.50 | False | 1.00 | 13.8334 | 1,383,339.73 | 133,339.73 |
| Post-Policy Statement (High Hawkish Signal) | 60 | $100,000 | 12.50 | True | 1.44 | 14.4722 | 1,447,222.70 | 197,222.70 |
| Baseline (Neutral/Routine) | 90 | $100,000 | 12.50 | False | 1.00 | 14.1721 | 1,417,206.98 | 167,206.98 |
| Post-Policy Statement (High Hawkish Signal) | 90 | $100,000 | 12.50 | True | 1.44 | 14.9298 | 1,492,979.87 | 242,979.87 |

* Key Takeaway: Under a high hawkish policy shock scenario over a 90-day settlement horizon, an SME importing $100,000 USD worth of goods must hold an additional Bs 81,000 in reserve above nominal cost (compared to Bs 56,000 under baseline conditions) to achieve 95% settlement certainty.

## Installation & Execution Guide
### 1. Environment Setup
Clone the repository and install requirements.

### 2. Running the Analytical Pipeline
Run the modular pipeline sequentially:
1. Scrape raw BCB press notes and download exchange rates
python src/data_collection.py
python src/add_manual_pdf.py
python -m src.clean_communications
python src/signal_classifier.py
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

## Data Snapshot & Temporal Disclaimer
The datasets included in this repository (`data/raw/` and `data/processed/`) represent a **static snapshot of the Bolivian macroeconomic environment and central bank communications as of October 21, 2026**. 

Because this project relies on live web scraping from the Central Bank of Bolivia (BCB), running the data collection pipeline (`src/data_collection.py`) at a later date will automatically ingest newer communications and exchange rate data. 

**Please note the following when reproducing this analysis:**
* **Dynamic Market Conditions:** Financial markets and central bank policies are highly dynamic. Running this pipeline with post-October 2026 data may yield different NLP signal distributions, backtesting correlations, and Monte Carlo volatility parameters.
* **Reproducibility:** The empirical results, scatter plots, and risk matrices documented in this README are strictly tied to the provided historical snapshot. This ensures the reproducibility of the Q3 2026 analysis, regardless of when the repository is cloned.
* **No Forward-Looking Guarantee:** The Monte Carlo simulations and NLP classifications are intended for exploratory, educational, and portfolio demonstration purposes only. They do not constitute financial advice or guarantee future market behavior.