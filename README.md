# Pasture Health CV

Sentinel-2 multispectral image segmentation for NZ pasture / crop health monitoring.
Production-grade MLOps pipeline targeting NZ agritech employers (Fonterra, LIC, Halter).

**Stack:** Sentinel-2 L2A · PyTorch · SegFormer · FastAPI · MLflow · Prometheus · Grafana · PostgreSQL · Docker Compose

---

## Resume Bullets

```
Pasture Health Segmentation — Production CV Pipeline
Sentinel-2 · PyTorch · SegFormer · FastAPI · MLflow · Prometheus · Grafana

· Trained SegFormer-B0 (MiT-B0 encoder, 3.7 M params) on Sentinel-2 L2A
  multispectral tiles + NZ LRIS LCDB v5.0 weak labels for 4-class pasture
  segmentation (healthy / stressed / bare / water); benchmarked 3 architectures
  (U-Net 32.5 M / DeepLabV3+ 11.7 M / SegFormer-B0 3.7 M) — SegFormer achieved
  best val mIoU 0.60 with 9× fewer parameters than the U-Net baseline.

· Production MLOps pipeline: FastAPI REST service (POST /infer) with pandera
  I/O schema validation, MLflow experiment tracking across all ablation runs,
  Docker Compose full-stack deployment (API + PostgreSQL + Prometheus + Grafana).

· Prometheus + Grafana observability: inference latency histogram (p50/p95),
  rolling NDVI drift Z-score alerting vs training baseline, per-region health-score
  time series — directly applicable to NZ precision dairy farming use cases.
```

---

## Architecture Ablation (30 epochs, identical hyperparams)

| Model | Encoder | Params | Val mIoU |
|---|---|---|---|
| U-Net | ResNet-50 | 32.5 M | 0.568 |
| DeepLabV3+ | EfficientNet-B3 | 11.7 M | 0.512 |
| **SegFormer-B0** ← best | **MiT-B0** | **3.7 M** | **0.600** |

Data: Sentinel-2 L2A (Waikato + Canterbury, 4 seasons 2024) · LRIS LCDB v5.0 weak labels
4 classes: healthy pasture · stressed · bare soil · water

---

## Quick Start

### Prerequisites

- Python 3.11, [uv](https://github.com/astral-sh/uv)
- Docker Desktop
- Free API credentials (see `.env.example`)

### 1 — Clone and install

```bash
git clone https://github.com/HarlanAlternative/pasture-health-cv
cd pasture-health-cv
pip install uv && uv sync
cp .env.example .env   # fill in SH_CLIENT_ID, SH_CLIENT_SECRET, LRIS_API_KEY
```

### 2 — Fetch data + extract patches

```bash
uv run python scripts/prepare_data.py
# Downloads 8 seasonal Sentinel-2 tiles (Waikato + Canterbury Q1–Q4 2024)
# + LRIS LCDB v5.0 land-cover polygons, extracts 392 training patches
```

### 3 — Train all architectures (ablation)

```bash
uv run python scripts/ablation.py --epochs 30
# Trains U-Net → DeepLabV3+ → SegFormer sequentially
# All runs logged to MLflow — view with: uv run mlflow ui
```

### 4 — Full stack (API + Prometheus + Grafana)

```bash
docker compose up
```

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

### 5 — One-command inference

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": [175.2, -38.1, 175.7, -37.6],
    "date_from": "2024-01-01",
    "date_to":   "2024-03-31"
  }'
```

**Example response:**
```json
{
  "health_score": 99.4,
  "ndvi_stats":  {"mean": 0.766, "std": 0.202, "p25": 0.741, "p75": 0.876},
  "class_pct":   {"healthy_pasture": 95.1, "stressed": 0.0, "bare": 0.6, "water": 1.9},
  "tile_meta":   {"bbox": [175.2, -38.1, 175.7, -37.6], "gsd_m": 85.8},
  "inference_ms": 183,
  "mask_b64":    "<base64 PNG>"
}
```

---

## Repo Structure

```
pasture-health-cv/
├── src/pasture/
│   ├── data/          # Sentinel-2 fetch, LRIS ingest, tiling, dataset
│   ├── models/        # U-Net, DeepLabV3+, SegFormer factory
│   ├── training/      # train loop, metrics, MLflow logging
│   ├── inference/     # predictor, health score, NDVI stats
│   ├── api/           # FastAPI app, pandera schemas, PostgreSQL
│   └── monitoring/    # Prometheus metrics, NDVI drift detector
├── scripts/
│   ├── prepare_data.py   # batch fetch + patch extraction
│   └── ablation.py       # multi-architecture training study
├── notebooks/
│   ├── 01_eda_sentinel2.ipynb   # S2 true-colour + NDVI + LCDB overlay
│   └── 02_train_baseline.ipynb  # training + prediction visualisation
├── infra/
│   ├── prometheus/    # scrape config + alerting rules
│   ├── grafana/       # dashboard JSON + provisioning
│   └── postgres/      # schema init SQL
├── docker-compose.yml
└── Dockerfile
```

---

## Credentials

| Key | Where to get |
|---|---|
| `SH_CLIENT_ID` + `SH_CLIENT_SECRET` | [Planet Insights Platform](https://insights.planet.com/account/#/) → OAuth clients → + New |
| `LRIS_API_KEY` | [lris.scinfo.org.nz](https://lris.scinfo.org.nz) → sign in → avatar → API keys |
