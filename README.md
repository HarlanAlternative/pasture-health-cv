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

### Architecture Ablation — 30 epochs, identical hyperparams

| Model | Encoder | Params | Val mIoU |
|:--|:--|--:|--:|
| U-Net | ResNet-50 | 32.5 M | 0.568 |
| DeepLabV3+ | EfficientNet-B3 | 11.7 M | 0.512 |
| **SegFormer-B0** | **MiT-B0** | **3.7 M** | **0.600** |

SegFormer achieves the best mIoU with **9× fewer parameters** than the U-Net baseline.

**Data:** Sentinel-2 L2A · Waikato + Canterbury · 4 seasons 2024 · 392 training patches
**Labels:** LRIS LCDB v5.0 weak supervision → 4 classes: `healthy` · `stressed` · `bare` · `water`

---

## System Architecture

```
Sentinel Hub API
      │  Sentinel-2 L2A tile (4-band, 512×512)
      ▼
┌─────────────────────────────────────────────┐
│              FastAPI  /infer                │
│                                             │
│  pandera input validation                   │
│       │                                     │
│  SegFormer-B0 (GPU inference ~200 ms)       │
│       │                                     │
│  pandera output validation                  │
│       │                                     │
│  mask PNG · health score · NDVI stats       │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
  PostgreSQL         Prometheus
  (history)     ┌────────┴────────┐
                │    Grafana      │
                │  health score   │
                │  NDVI drift     │
                │  latency p95    │
                └─────────────────┘
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

# 2. Fetch data + extract patches  (~5 min, API calls cached locally)
uv run python scripts/prepare_data.py

# 3. Train all architectures
uv run python scripts/ablation.py --epochs 30
uv run mlflow ui                # view runs at http://localhost:5000

# 4. Start full stack
docker compose up

# 5. Run inference
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"bbox": [175.2, -38.1, 175.7, -37.6],
       "date_from": "2024-01-01", "date_to": "2024-03-31"}'
```

**Example response:**
```json
{
  "health_score": 99.4,
  "ndvi_stats":   { "mean": 0.766, "std": 0.202, "p25": 0.741, "p75": 0.876 },
  "class_pct":    { "healthy_pasture": 95.1, "stressed": 0.0, "bare": 0.6, "water": 1.9 },
  "tile_meta":    { "bbox": [175.2, -38.1, 175.7, -37.6], "gsd_m": 85.8 },
  "inference_ms": 183
}
```

---

## Services

| Service | URL | Notes |
|:--|:--|:--|
| FastAPI | http://localhost:8000/docs | Interactive Swagger UI |
| Prometheus | http://localhost:9090 | 4 alert rules configured |
| Grafana | http://localhost:3000 | `admin` / `admin` |
| PostgreSQL | localhost:5432 | Inference history + drift metrics |

---

## Repo Structure

```
src/pasture/
├── data/        Sentinel-2 fetch · LRIS ingest · tiling · dataset
├── models/      U-Net · DeepLabV3+ · SegFormer factory
├── training/    train loop · metrics · MLflow logging
├── inference/   predictor · health score · NDVI stats
├── api/         FastAPI · pandera schemas · PostgreSQL persistence
└── monitoring/  Prometheus metrics · NDVI drift detector

scripts/
├── prepare_data.py   batch fetch 8 seasonal tiles + patch extraction
└── ablation.py       3-architecture training study

infra/
├── prometheus/   scrape config + 4 alerting rules
├── grafana/      6-panel dashboard + auto-provisioning
└── postgres/     schema init SQL
```

---

## Credentials

| Variable | Where to get |
|:--|:--|
| `SH_CLIENT_ID` / `SH_CLIENT_SECRET` | [Planet Insights Platform](https://insights.planet.com/account/#/) → OAuth clients → + New |
| `LRIS_API_KEY` | [lris.scinfo.org.nz](https://lris.scinfo.org.nz) → sign in → avatar → API keys |
