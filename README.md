## Central Bank Communications & Exchange-Rate Volatility: Bolivia (BCB)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Executive Overview & Business Problem
An exploratory pipeline that (1) collects Banco Central de Bolivia (BCB) press releases, (2) tags them with a zero-shot NLP classifier, (3) tests whether those tags are associated with changes in exchange-rate volatility around each announcement, and (4) turns observed volatility into scenario-based cash-reserve buffers for an SME importing under a USD contract. This repository provides an exploratory quantitative research pipeline designed to evaluate two core business questions:

1. Signaling: Are BCB communications, as tagged by a zero-shot classifier, associated with a change in exchange-rate volatility in the days after they are published?

2. Risk translation: Given observed volatility under the flexible regime, how large a cash buffer does an SME need to hold for a USD-denominated contract at 95% confidence, and how sensitive is that to a volatility shock?

## Institutional context
| Date | Event |
|----|----|
| Until 2026-06-28 | Fixed official rate (Bs 6.96 per USD, in place since 2011) |
| 2026-06-29 | Flexible exchange-rate regime takes effect (Ministerial Resolution 245/2026 and BCB Board Resolution 88/2026) |
| 2026-09-24 (published 09-25) | BCB Board Resolution 142/2026 repeals RD 88/2026 and issues an updated Reglamento de Operaciones Cambiarias |

Under the new rules the official exchange rate (TCO) is the amount-weighted median of banks' USD purchase operations with clients and the BCB, excluding interbank trades, rounded to two decimals and published each business day at 23:30 for use the next day.

## Architecture & Data Pipeline
```text
├── data/
│   ├── raw/          # scraped communications and exchange-rate snapshot
│   └── processed/    # deduplicated documents and classifier output
├── docs/             # data_notes.md: series definition and open questions
├── outputs/
│   ├── figures/
│   └── reports/
├── scripts/          # one-off diagnostics and README table generator
├── src/              # pipeline modules
└── tests/                 
```

## Methodology & Model Mechanics

### 1. NLP tagging (src/signal_classifier.py)
The model MoritzLaurer/mDeBERTa-v3-base-mnli-xnli assigns each text chunk to one of four candidate topics (restriction of currency / exchange controls; monetary or exchange-regime adjustment; reserves and liquidity stability; administrative or routine). For each document the field intensity_score is the maximum zero-shot confidence over chunks.

### 2. Event study (src/backtest_signals.py). 
For each event date T:
- Pre window: the 20 business-day log returns ending on T.
- Post window: the 5 business-day log returns strictly after T (both windows must be complete).
- Outcome: log(post-window volatility / pre-window volatility).
- Several documents on the same date count as one event, using the maximum score.

Measuring volatility relative to its own recent level avoids comparing calm and turbulent periods directly. Windows were fixed before looking at results.

### 3. Robustness (src/robustness.py)
Fisher confidence intervals, a permutation p-value (10,000 shuffles), Spearman correlation, leave-one-out ranges, and splits by regime. The exclusion of Nov 2025 to Jan 2026 was chosen after inspecting the exchange-rate plot, so it is a sensitivity check, not a primary test.

### 4. Scenario simulation (src/monte_carlo.py)
Geometric Brownian motion with zero drift and 10,000 paths. Spot is the last observed value (Bs 11.84) and daily volatility (1.99%) is estimated from log returns since 2026-06-29. Horizons are in business days. Stress scenarios multiply volatility by 1.5 and 2.0; these multipliers are illustrative and are not estimated from the NLP signal.

## Results

### 1. Is the NLP score associated with volatility around announcements?

NLP confidence vs change in FX volatility
| Sample | n | Pearson r | 95% CI | Permutation p |
| --- | --- | --- | --- | --- |
| full sample | 41 | 0.306 | [-0.002, 0.561] | 0.05 |
| before flexible regime | 30 | 0.305 | [-0.062, 0.6] | 0.104 |
| flexible regime | 11 | 0.432 | [-0.226, 0.82] | 0.196 |
| excluding Nov25-Jan26 | 26 | 0.175 | [-0.228, 0.527] | 0.398 |

Spearman correlation on the full sample is 0.22.

Reading: a weak positive association that sits right at the edge of conventional significance and weakens when the period with the noisiest exchange-rate data is removed (15 of the 41 events fall in it). The flexible-regime result (r = 0.43) rests on 11 events and its interval spans −0.23 to 0.82, so it supports no conclusion. Even a real association would not be causal: the BCB may publish more exchange-rate material when markets are already turbulent.

Topic labels across the 42 documents: 18 reserves and liquidity, 12 regime adjustment, 6 routine, 6 currency restriction.

### 2. SME cash-reserve buffers (USD 100,000 contract, spot Bs 11.84)
Simulated exchange-rate distribution

| Scenario | Horizon (business days) | Daily volatility | 95% VaR rate (Bs/USD) | Required buffer (Bs) |
| --- | --- | --- | --- | --- |
| Baseline (observed flexible-regime volatility) | 30 | 1.99% | 14.16 | 232,350 |
| Stress x1.5 (illustrative) | 30 | 2.99% | 15.49 | 365,102 |
| Stress x2.0 (illustrative) | 30 | 3.98% | 16.94 | 510,297 |
| Baseline (observed flexible-regime volatility) | 60 | 1.99% | 15.23 | 339,050 |
| Stress x1.5 (illustrative) | 60 | 2.99% | 17.27 | 543,410 |
| Stress x2.0 (illustrative) | 60 | 3.98% | 19.59 | 775,190 |
| Baseline (observed flexible-regime volatility) | 90 | 1.99% | 16.11 | 426,512 |
| Stress x1.5 (illustrative) | 90 | 2.99% | 18.78 | 694,321 |
| Stress x2.0 (illustrative) | 90 | 3.98% | 21.91 | 1,006,665 |

* The buffer is the 95th-percentile terminal rate times the contract amount, minus the cost at spot. Read these numbers as a conservative upper bound from a simple model, not as a forecast: constant 2% daily volatility compounding without mean reversion or policy intervention produces wide distributions (the simulated lower tail even falls below the former fixed rate)..

## Limitations
* Small sample and low power. 41 events can only detect |r| ≥ 0.43 (80% power, 5% level). The flexible-regime subsample has 11 events.
* The NLP score is not a stance measure. It is a topic classifier's confidence, unvalidated against human labels. Validating it is the most valuable next step.
* Two different series are pooled. Before 2026-06-29 the exchange-rate series is not the official rate and its definition is unverified.
* Suspect data period. Between about November 2025 and January 2026 the series shows daily moves of 10% to 24% whose origin is unclear. Results are shown with and without it.
* A median is smoother than a market quote. The TCO is a weighted median of bank purchases, which understates short-term volatility, so "no reaction" findings may partly reflect the instrument.
* Timing conventions are unconfirmed. Whether Fecha is the date a rate applies or the date it is published could shift windows by one day. Holidays may repeat the previous value.
Possible confounding and reverse causality. Announcements cluster in turbulent periods.
* Three regimes, not two. Events after 2026-09-25 fall under RD 142/2026; there are too few to analyze separately.
* Monte Carlo is a scenario tool. Zero drift, normally distributed returns, volatility from about 85 business days that include an unusual spike in September 2026, and horizons counted in business days (a 30-calendar-day contract is about 21 business days).
* Multiple comparisons. Four samples are reported; a borderline result in one of them should not be read as a discovery.


## How to reproduce
```
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Optional: refresh the data. This scrapes the live BCB site and will change results.
# python -m src.data_collection
# python -m src.add_manual_pdf

python -m src.clean_communications     # deduplicate documents
python -m src.signal_classifier        # zero-shot tagging (downloads the model on first run)
python -m src.backtest_signals         # event study
python -m src.robustness               # CIs, permutation test, regime splits
python -m src.monte_carlo              # SME risk matrix
python -m src.visualize_results        # figures
python scripts/make_readme_tables.py   # tables used in this README

pytest -v
```

## Engineering Approach & AI Disclosure
- Human-defined: the research question, institutional context, scenario design, event-study specification, and interpretation of results.
- AI-assisted: drafting of scraping and classification boilerplate, plotting styles, test scaffolding, an audit of v0.1, and step-by-step guidance through the v0.2 fixes. All code was run and all numbers reproduced locally; the audit's claims were checked against the data before being acted on.
- Encoding: files are UTF-8 to preserve Spanish characters

## Disclaimer
This is an exploratory, educational and portfolio project. It is not financial advice and does not guarantee future market behavior. Results are tied to the 2026-10-23 data snapshot.

