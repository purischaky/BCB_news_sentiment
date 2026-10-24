import pandas as pd
import matplotlib.pyplot as plt

fx = pd.read_csv("data/raw/official_exchange_rate.csv", parse_dates=["date"]).sort_values("date")
fx["ret"] = fx["value"].pct_change()

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(fx["date"], fx["value"], lw=0.8)
axes[0].set_ylabel("Bs per USD")
axes[1].plot(fx["date"], fx["ret"], lw=0.6)
axes[1].set_ylabel("Daily return")
plt.tight_layout()
plt.savefig("outputs/figures/fx_series_diagnostic.png", dpi=150)

print(fx[fx["ret"].abs() > 0.05][["date", "value", "ret"]].to_string())