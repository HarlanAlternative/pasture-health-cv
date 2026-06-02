# Pasture Health CV

Sentinel-2 pasture/crop health segmentation for NZ agritech (Fonterra, LIC, Halter).

**Stack:** Sentinel-2 L2A · PyTorch · SegFormer · FastAPI · MLflow · Prometheus · Grafana

## Results — Architecture Ablation (30 epochs, same hyperparams)

| Model | Encoder | Params | Val mIoU |
|---|---|---|---|
| U-Net | ResNet-50 | 32.5M | 0.568 |
| DeepLabV3+ | EfficientNet-B3 | 11.7M | 0.512 |
| **SegFormer-B0** ← best | **MiT-B0** | **3.7M** | **0.600** |

4-class segmentation: healthy pasture / stressed / bare soil / water  
Data: Sentinel-2 L2A (NZ Waikato + Canterbury, 4 seasons 2024) · LCDB v5.0 weak labels

## Quick Start

```bash
git clone https://github.com/HarlanAlternative/pasture-health-cv
cd pasture-health-cv
pip install uv && uv sync

# Set credentials (see .env.example)
cp .env.example .env   # fill in SH_CLIENT_ID, SH_CLIENT_SECRET, LRIS_API_KEY

# Fetch data + extract patches
uv run python scripts/prepare_data.py

# Train all architectures (ablation)
uv run python scripts/ablation.py --epochs 30

# View experiment results
uv run mlflow ui
```

## EDA Notebook

```bash
uv run jupyter lab notebooks/01_eda_sentinel2.ipynb
```

Renders Sentinel-2 true-colour + NDVI + LCDB land-cover overlay for Waikato.

## Full Stack (Sprint 4+)

```bash
docker compose up
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"bbox": [175.2, -38.1, 175.7, -37.6], "date": "2024-02-01"}'
```
