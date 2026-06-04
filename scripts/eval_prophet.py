"""
Prophet backtesting: train on 2022-2023, predict 2024, compare against actuals.

Metrics:
  MAE   — mean absolute error (NDVI units, lower is better)
  RMSE  — root mean square error
  MAPE  — mean absolute percentage error
  Cov80 — % of actuals falling inside the 80% confidence interval
"""
import sys, json, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

HISTORY = Path(__file__).parent.parent / "data" / "historical_ndvi.json"
QUARTER_DATES = {"Q1": "-02-15", "Q2": "-05-15", "Q3": "-08-15", "Q4": "-11-15"}


def backtest(aoi: str, records: list[dict]) -> dict:
    train = [r for r in records if r["year"] <= 2023]
    test  = [r for r in records if r["year"] == 2024]
    if not train or not test:
        return {}

    df_train = pd.DataFrame([
        {"ds": pd.Timestamp(f"{r['year']}{QUARTER_DATES[r['quarter']]}"),
         "y": r["ndvi_mean"]} for r in train
    ]).sort_values("ds")

    model = Prophet(
        yearly_seasonality=4,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,
        seasonality_mode="additive",
        growth="flat",
    )
    model.fit(df_train)

    future_dates = [
        pd.Timestamp(f"{r['year']}{QUARTER_DATES[r['quarter']]}") for r in test
    ]
    forecast = model.predict(pd.DataFrame({"ds": future_dates}))

    actuals = np.array([r["ndvi_mean"] for r in test])
    yhat    = forecast["yhat"].values
    lo      = forecast["yhat_lower"].values
    hi      = forecast["yhat_upper"].values

    mae   = float(np.mean(np.abs(actuals - yhat)))
    rmse  = float(np.sqrt(np.mean((actuals - yhat) ** 2)))
    mape  = float(np.mean(np.abs((actuals - yhat) / (actuals + 1e-8))) * 100)
    cov80 = float(np.mean((actuals >= lo) & (actuals <= hi)) * 100)

    print(f"\n{aoi.upper()} — 2024 backtest")
    print(f"  {'Quarter':<8} {'Actual':>8} {'Predicted':>10} {'Error':>8}  {'In CI?':>7}")
    print(f"  {'-'*50}")
    for r, act, pred, l, h in zip(test, actuals, yhat, lo, hi):
        err = act - pred
        in_ci = "YES" if l <= act <= h else "NO"
        print(f"  {r['quarter']:<8} {act:>8.4f} {pred:>10.4f} {err:>+8.4f}  {in_ci:>7}")
    print(f"  {'MAE':>10}: {mae:.4f}")
    print(f"  {'RMSE':>10}: {rmse:.4f}")
    print(f"  {'MAPE':>10}: {mape:.1f}%")
    print(f"  {'CI cov80':>10}: {cov80:.0f}%  (target ≥ 80%)")

    return {"aoi": aoi, "mae": mae, "rmse": rmse, "mape": mape, "cov80": cov80}


def main():
    data = json.loads(HISTORY.read_text())
    results = []
    for aoi, records in data.items():
        r = backtest(aoi, records)
        if r:
            results.append(r)

    print("\n" + "=" * 55)
    print(f"{'AOI':<14} {'MAE':>7} {'RMSE':>7} {'MAPE':>8} {'CI cov80':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['aoi']:<14} {r['mae']:>7.4f} {r['rmse']:>7.4f} {r['mape']:>7.1f}% {r['cov80']:>9.0f}%")
    print("=" * 55)
    avg_mae = np.mean([r["mae"] for r in results])
    print(f"Avg MAE: {avg_mae:.4f} NDVI units")


if __name__ == "__main__":
    main()
