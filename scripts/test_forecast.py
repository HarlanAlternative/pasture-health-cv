"""Quick smoke test for seasonal baseline + Prophet forecast."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pasture.monitoring.seasonal import SeasonalBaseline
from pasture.monitoring.forecast import forecast_ndvi

# Seasonal baseline
baseline = SeasonalBaseline()
print("=== Seasonal baseline (Hawkes Bay) ===")
for q in baseline.summary("hawkes_bay"):
    mean = q.get("mean", "?")
    std  = q.get("std", "?")
    print(f"  {q['quarter']}: mean={mean:.4f}  std={std:.4f}")

# Hawke's Bay 2024-Q2 was 0.1557 — should flag as anomaly
z = baseline.zscore("hawkes_bay", "Q2", 0.1557)
print(f"\n  2024-Q2 observed NDVI=0.1557  Z-score={z}  alert={abs(z) > 2}")

# Prophet forecast
print("\n=== Prophet forecast (Waikato, next 4 quarters) ===")
result = forecast_ndvi("waikato", quarters_ahead=4)
for r in result["forecast"]:
    print(f"  {r['date']}  yhat={r['yhat']:.4f}  CI=[{r['yhat_lower']:.4f}, {r['yhat_upper']:.4f}]")

print("\n=== Prophet forecast (Hawkes Bay, next 4 quarters) ===")
result2 = forecast_ndvi("hawkes_bay", quarters_ahead=4)
for r in result2["forecast"]:
    print(f"  {r['date']}  yhat={r['yhat']:.4f}  CI=[{r['yhat_lower']:.4f}, {r['yhat_upper']:.4f}]")
