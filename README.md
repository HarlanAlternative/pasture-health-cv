# Pasture Health CV

Sentinel-2 pasture/crop health segmentation for NZ agritech (Fonterra, LIC, Halter).

**Stack:** Sentinel-2 L2A · PyTorch · DeepLabV3+ · FastAPI · MLflow · Prometheus · Grafana

## Sprint 1 Quick Start (data + NDVI plot only)

```bash
# 1. Clone and enter
git clone <repo-url> pasture-health-cv && cd pasture-health-cv

# 2. Install deps (Python 3.11 required)
pip install uv
uv sync

# 3. Set credentials (see .env.example for where to get them)
cp .env.example .env
# edit .env — fill in SH_CLIENT_ID and SH_CLIENT_SECRET

# 4. Launch notebook
uv run jupyter lab notebooks/01_eda_sentinel2.ipynb
```

Run all cells. The final cell saves `data/eda_ndvi_waikato.png`.

## Full stack (Sprint 4+)

```bash
docker compose up
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"bbox": [175.2, -38.1, 175.7, -37.6], "date": "2024-02-01"}'
```
