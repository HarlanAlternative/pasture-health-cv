# Pasture Health CV

> End-to-end semantic segmentation of NZ pasture and cropland health from
> Sentinel-2 multispectral imagery, served as a production REST API with full MLOps observability.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2Bcu128-red?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-tracked-blue?logo=mlflow)

---

## Results

### Architecture Ablation — 30 epochs, 4 AOIs, class-weighted loss

| Model | Encoder | Params | Val mIoU |
|:--|:--|--:|--:|
| **U-Net** ← best | **ResNet-50** | **32.5 M** | **0.582** |
| SegFormer-B0 | MiT-B0 | 3.7 M | 0.576 |
| DeepLabV3+ | EfficientNet-B3 | 11.7 M | 0.513 |

Per-class IoU (best model): `healthy 0.789 · stressed 0.507 · bare 0.541 · water 0.490`

**Data:** Sentinel-2 L2A · 4 NZ regions (Waikato, Canterbury, Hawke's Bay, Marlborough) ·
4 seasons 2024 · 546 training patches · class-weighted CE + Dice loss
**Labels:** LRIS LCDB v5.0 weak supervision → 4 classes: `healthy` · `stressed` · `bare` · `water`

---

## System Architecture

```
Sentinel Hub API
      │  Sentinel-2 L2A tile (4-band, 512×512)
      ▼
┌─────────────────────────────────────────────────┐
│               FastAPI  /infer                   │
│                                                 │
│  pandera input validation                       │
│       │                                         │
│  U-Net / ResNet-50 (GPU inference ~200 ms)      │
│       │                                         │
│  pandera output validation                      │
│       │                                         │
│  mask PNG · health score · NDVI stats           │
└──────────────┬──────────────────────────────────┘
               │
       ┌───────┴────────────────┐
       ▼                        ▼
  PostgreSQL              Seasonal Baseline
  (history)               (3-yr climatology)
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
               Prometheus            Prophet
               seasonal Z-score      quarterly forecast
               + latency alerts      /forecast endpoint
                    │
               Grafana (6 panels)
```

---

## Quick Start

**Prerequisites:** Python 3.11 · [uv](https://github.com/astral-sh/uv) · Docker Desktop

```bash
# 1. Clone and install
git clone https://github.com/HarlanAlternative/pasture-health-cv
cd pasture-health-cv
pip install uv && uv sync
cp .env.example .env          # fill in credentials (see below)

# 2. Fetch satellite tiles + extract patches  (~5 min, cached locally)
uv run python scripts/prepare_data.py

# 3. Fetch 3-year NDVI history for seasonal baseline + Prophet
uv run python scripts/fetch_historical.py

# 4. Train all architectures (ablation)
uv run python scripts/ablation.py --epochs 30
uv run mlflow ui                # view runs at http://localhost:5000

# 5. Start full stack
docker compose up

# 6. Run inference
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"bbox": [175.2, -38.1, 175.7, -37.6],
       "date_from": "2024-01-01", "date_to": "2024-03-31"}'

# 7. Get seasonal NDVI forecast
curl "http://localhost:8000/forecast?aoi=waikato&quarters=4"
```

**Inference response:**
```json
{
  "health_score": 99.4,
  "ndvi_stats":   { "mean": 0.766, "std": 0.202, "p25": 0.741, "p75": 0.876 },
  "class_pct":    { "healthy_pasture": 95.1, "stressed": 0.0, "bare": 0.6, "water": 1.9 },
  "tile_meta":    { "bbox": [175.2, -38.1, 175.7, -37.6], "gsd_m": 85.8 },
  "inference_ms": 183
}
```

**Forecast response:**
```json
{
  "aoi": "waikato",
  "forecast": [
    { "date": "2025-02-15", "yhat": 0.637, "yhat_lower": 0.613, "yhat_upper": 0.657 },
    { "date": "2025-05-15", "yhat": 0.822, "yhat_lower": 0.800, "yhat_upper": 0.844 }
  ]
}
```

---

## Services

| Service | URL | Notes |
|:--|:--|:--|
| FastAPI | http://localhost:8000/docs | Swagger UI — `/infer`, `/forecast`, `/drift` |
| Prometheus | http://localhost:9090 | 4 alert rules (latency, health, drift, traffic) |
| Grafana | http://localhost:3000 | `admin` / `admin` · 6-panel dashboard |
| PostgreSQL | localhost:5432 | Inference history |

---

## Repo Structure

```
src/pasture/
├── data/        Sentinel-2 fetch · LRIS ingest · tiling · dataset
├── models/      U-Net · DeepLabV3+ · SegFormer factory
├── training/    train loop · class-weighted loss · MLflow logging
├── inference/   predictor · health score · NDVI stats
├── api/         FastAPI · pandera schemas · PostgreSQL · /forecast
└── monitoring/  Prometheus metrics · seasonal drift detector · Prophet

scripts/
├── prepare_data.py      batch fetch 16 seasonal tiles (4 AOIs × 4 seasons)
├── fetch_historical.py  pull 2022-2024 NDVI history for seasonal baseline
├── ablation.py          3-architecture training study
└── eval_metrics.py      per-class IoU evaluation

infra/
├── prometheus/   scrape config + alerting rules
├── grafana/      6-panel dashboard + auto-provisioning
└── postgres/     schema init SQL
```

---

## Credentials

| Variable | Where to get |
|:--|:--|
| `SH_CLIENT_ID` / `SH_CLIENT_SECRET` | [Planet Insights Platform](https://insights.planet.com/account/#/) → OAuth clients → + New |
| `LRIS_API_KEY` | [lris.scinfo.org.nz](https://lris.scinfo.org.nz) → sign in → avatar → API keys |
