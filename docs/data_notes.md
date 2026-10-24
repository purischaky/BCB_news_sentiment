# Data notes: exchange-rate series

## Source
BCB "grafico_evolutivo" download, columns `Fecha` and `Valor`. Snapshot as of 2026-10-21.

## Definition of the series
- From 2026-06-29 (flexible regime, RM 245/2026 and RD 88/2026): the series is believed to be the
  TCO, the amount-weighted median of bank USD purchases, rounded to 2 decimals, published each
  business day at 23:30 for use the next day. Method restated in RD 142/2026 (2026-09-24), which
  repealed RD 88/2026.

## Open questions
- Is `Fecha` the date the rate applies, or the publication date? (Check against the BCB's TCO page.)
- Do holidays repeat the previous value?
- Does the series definition change on 2026-06-29 or 2026-09-25?

## Regimes in the sample
1. before 2026-06-29: fixed official rate / other series
2. 2026-06-29 to 2026-09-24: RD 88/2026
3. from 2026-09-25: RD 142/2026 (few events)