## SME risk matrix (USD 100,000 contract)

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

## Robustness of the NLP-volatility correlation

| Sample | n | Pearson r | 95% CI | Permutation p |
| --- | --- | --- | --- | --- |
| full sample | 41 | 0.306 | [-0.002, 0.561] | 0.05 |
| before flexible regime | 30 | 0.305 | [-0.062, 0.6] | 0.104 |
| flexible regime | 11 | 0.432 | [-0.226, 0.82] | 0.196 |
| excluding Nov25-Jan26 | 26 | 0.175 | [-0.228, 0.527] | 0.398 |
