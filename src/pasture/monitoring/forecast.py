"""
Prophet-based NDVI forecasting per AOI.

Fits a yearly-seasonal trend on 2022-2024 quarterly NDVI means,
forecasts the next N quarters with 80% confidence intervals.

## Assumptions
- Input: historical_ndvi.json with 12 quarterly records per AOI (3 years)
- Prophet yearly_seasonality ON, weekly/daily OFF (quarterly data)
- Quarterly dates mapped to mid-quarter (Feb 15, May 15, Aug 15, Nov 15)
- Uncertainty intervals reflect both trend and seasonal uncertainty
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from prophet import Prophet

_DEFAULT_HISTORY = Path(__file__).resolve().parents[3] / "data" / "historical_ndvi.json"

# Mid-quarter dates for Prophet's datetime axis
_QUARTER_DATES = {"Q1": "-02-15", "Q2": "-05-15", "Q3": "-08-15", "Q4": "-11-15"}


def _to_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        date_str = f"{r['year']}{_QUARTER_DATES[r['quarter']]}"
        rows.append({"ds": pd.Timestamp(date_str), "y": r["ndvi_mean"]})
    return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)


def forecast_ndvi(
    aoi: str,
    quarters_ahead: int = 4,
    history_path: str | Path = _DEFAULT_HISTORY,
) -> dict:
    """Fit Prophet on historical NDVI and return forecast for next N quarters.

    Args:
        aoi: one of 'waikato', 'canterbury', 'hawkes_bay', 'marlborough'.
        quarters_ahead: how many future quarters to predict.
        history_path: path to historical_ndvi.json.

    Returns:
        dict with keys: aoi, history, forecast
    """
    path = Path(history_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Run scripts/fetch_historical.py first — {path} not found."
        )
    data = json.loads(path.read_text())
    if aoi not in data:
        raise ValueError(f"AOI '{aoi}' not in history. Available: {list(data)}")

    df = _to_dataframe(data[aoi])

    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    model = Prophet(
        yearly_seasonality=4,     # 4 Fourier terms — right for quarterly data
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,
        seasonality_mode="additive",
        growth="flat",            # no linear trend extrapolation; NDVI is bounded
    )
    model.fit(df)

    # Future dates in mid-quarter format matching training data
    records_list = data[aoi]
    last = records_list[-1]
    quarters_cycle = ["Q1", "Q2", "Q3", "Q4"]
    qi = quarters_cycle.index(last["quarter"])
    yr = last["year"]
    future_dates = []
    for _ in range(quarters_ahead):
        qi += 1
        if qi >= 4:
            qi = 0
            yr += 1
        date_str = f"{yr}{_QUARTER_DATES[quarters_cycle[qi]]}"
        future_dates.append(pd.Timestamp(date_str))
    future_df = pd.DataFrame({"ds": future_dates})
    forecast = model.predict(future_df)

    history_out = [
        {"date": str(row.ds.date()), "ndvi_mean": round(row.y, 4)}
        for _, row in df.iterrows()
    ]
    def _clip(v: float) -> float:
        return round(float(max(-1.0, min(1.0, v))), 4)

    forecast_out = [
        {
            "date"      : str(row.ds.date()),
            "yhat"      : _clip(row.yhat),
            "yhat_lower": _clip(row.yhat_lower),
            "yhat_upper": _clip(row.yhat_upper),
        }
        for _, row in forecast.iterrows()
    ]

    return {
        "aoi"     : aoi,
        "history" : history_out,
        "forecast": forecast_out,
    }
