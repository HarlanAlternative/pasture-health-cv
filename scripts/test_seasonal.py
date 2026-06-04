"""Smoke test for seasonal AOI/quarter detection and Z-score."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pasture.monitoring.seasonal import detect_aoi, detect_quarter, SeasonalBaseline
from pasture.monitoring.drift import DriftDetector

print("=== AOI detection ===")
cases = [
    ((175.2, -38.1, 175.7, -37.6), "waikato"),
    ((171.5, -43.8, 172.0, -43.3), "canterbury"),
    ((176.6, -39.8, 177.1, -39.3), "hawkes_bay"),
    ((173.7, -41.8, 174.2, -41.3), "marlborough"),
]
for bbox, expected in cases:
    got = detect_aoi(bbox)
    status = "OK" if got == expected else "FAIL"
    print(f"  {expected:<12} -> {got:<12} {status}")

print("\n=== Quarter detection ===")
for date, expected in [("2024-01-01","Q1"),("2024-05-15","Q2"),
                        ("2024-08-01","Q3"),("2024-11-30","Q4")]:
    got = detect_quarter(date)
    status = "OK" if got == expected else "FAIL"
    print(f"  {date} -> {got}  {status}")

print("\n=== Seasonal Z-scores (Waikato) ===")
b = SeasonalBaseline()
tests = [
    ("Q1", 0.766, "normal"),
    ("Q2", 0.840, "normal"),
    ("Q3", 0.200, "should alert - low winter NDVI"),
    ("Q2", 0.150, "should alert - drought"),
]
for q, ndvi, desc in tests:
    z = b.zscore("waikato", q, ndvi)
    alert = abs(z) > 2 if z is not None else False
    print(f"  {q}  ndvi={ndvi:.3f}  z={z:+.3f}  alert={alert}  ({desc})")

print("\n=== DriftDetector with seasonal baseline ===")
d = DriftDetector(window=10)
waikato_bbox = [175.2, -38.1, 175.7, -37.6]
for date, ndvi in [("2024-01-15", 0.766), ("2024-08-15", 0.20), ("2024-05-15", 0.84)]:
    z = d.update(ndvi, bbox=waikato_bbox, date_from=date)
    print(f"  date={date}  ndvi={ndvi:.3f}  z={z:+.3f}  alert={abs(z)>2}")
