"""Build notebooks/02_train_baseline.ipynb (UTF-8, no BOM)."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "02_train_baseline.ipynb"


def cell(source: str, cell_type: str = "code") -> dict:
    base = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def md(source: str) -> dict:
    return cell(source, "markdown")


cells = [
    md(
        "# Sprint 2 — U-Net Baseline Training\n"
        "\n"
        "**Flow:** prepare_data.py -> Dataset -> U-Net (ResNet50) -> MLflow\n"
        "\n"
        "Run `uv run python scripts/prepare_data.py` before executing this notebook."
    ),
    cell(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "project_root = Path().resolve().parent\n"
        "if str(project_root / 'src') not in sys.path:\n"
        "    sys.path.insert(0, str(project_root / 'src'))\n"
        "\n"
        "import torch\n"
        "print(f'PyTorch : {torch.__version__}')\n"
        "print(f'CUDA    : {torch.cuda.is_available()}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'GPU     : {torch.cuda.get_device_name(0)}')"
    ),
    md("## 1. Verify patch dataset"),
    cell(
        "from pasture.data.dataset import PastureDataset\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "\n"
        "patch_dir = project_root / 'data' / 'patches'\n"
        "train_ds = PastureDataset(patch_dir / 'train', augment=False)\n"
        "val_ds   = PastureDataset(patch_dir / 'val',   augment=False)\n"
        "print(f'Train patches: {len(train_ds)}')\n"
        "print(f'Val   patches: {len(val_ds)}')\n"
        "\n"
        "img, msk = train_ds[0]\n"
        "print(f'Image shape : {img.shape}  dtype: {img.dtype}')\n"
        "print(f'Mask  shape : {msk.shape}  dtype: {msk.dtype}')\n"
        "print(f'Mask classes: {msk.unique().tolist()}')"
    ),
    cell(
        "# Visualise a sample patch\n"
        "LABEL_COLORS = ['#aaaaaa', '#2d6a2d', '#d4e157', '#c4a35a', '#4fc3f7']\n"
        "LABEL_NAMES  = ['ignored', 'healthy', 'stressed', 'bare', 'water']\n"
        "import matplotlib.colors as mcolors\n"
        "from matplotlib.patches import Patch\n"
        "\n"
        "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n"
        "fig.suptitle('Sample training patches (true colour | mask)', fontsize=12)\n"
        "\n"
        "from pasture.data.dataset import BAND_MEAN, BAND_STD\n"
        "\n"
        "for col in range(4):\n"
        "    img, msk = train_ds[col * len(train_ds) // 4]\n"
        "    # Denormalise RGB for display\n"
        "    rgb = img[:3].numpy() * BAND_STD[:3, None, None] + BAND_MEAN[:3, None, None]\n"
        "    rgb = np.transpose(np.clip(rgb / 0.25, 0, 1), (1, 2, 0))\n"
        "\n"
        "    axes[0, col].imshow(rgb)\n"
        "    axes[0, col].set_title(f'Patch {col}')\n"
        "    axes[0, col].axis('off')\n"
        "\n"
        "    cmap = mcolors.ListedColormap(LABEL_COLORS)\n"
        "    norm = mcolors.BoundaryNorm([-0.5+i for i in range(6)], cmap.N)\n"
        "    axes[1, col].imshow(msk.numpy(), cmap=cmap, norm=norm)\n"
        "    axes[1, col].axis('off')\n"
        "\n"
        "legend = [Patch(facecolor=c, label=n) for c, n in zip(LABEL_COLORS[1:], LABEL_NAMES[1:])]\n"
        "fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=9)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## 2. Train U-Net baseline"),
    cell(
        "# Runs training and logs everything to mlruns/\n"
        "# Open MLflow UI after: uv run mlflow ui\n"
        "from pasture.training.train import train\n"
        "\n"
        "train(\n"
        "    encoder='resnet50',\n"
        "    epochs=30,\n"
        "    batch_size=16,\n"
        "    lr=1e-4,\n"
        "    patch_dir=str(project_root / 'data' / 'patches'),\n"
        "    ckpt_dir=str(project_root / 'data' / 'checkpoints'),\n"
        ")"
    ),
    md("## 3. Inspect predictions"),
    cell(
        "import torch\n"
        "from pasture.models.unet import build_unet\n"
        "\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "ckpt = torch.load(project_root / 'data' / 'checkpoints' / 'best.pt',\n"
        "                  map_location=device)\n"
        "model = build_unet(encoder=ckpt['encoder']).to(device)\n"
        "model.load_state_dict(ckpt['model'])\n"
        "model.eval()\n"
        "print(f'Loaded checkpoint  epoch={ckpt[\"epoch\"]}  mIoU={ckpt[\"miou\"]:.4f}')"
    ),
    cell(
        "fig, axes = plt.subplots(3, 4, figsize=(16, 12))\n"
        "fig.suptitle('Val set: True Colour | Ground Truth | Prediction', fontsize=12)\n"
        "\n"
        "cmap = mcolors.ListedColormap(LABEL_COLORS)\n"
        "norm = mcolors.BoundaryNorm([-0.5+i for i in range(6)], cmap.N)\n"
        "\n"
        "with torch.no_grad():\n"
        "    for col in range(4):\n"
        "        img, msk = val_ds[col * len(val_ds) // 4]\n"
        "        logits = model(img.unsqueeze(0).to(device))\n"
        "        pred   = logits.argmax(1).squeeze().cpu().numpy()\n"
        "\n"
        "        rgb = img[:3].numpy() * BAND_STD[:3, None, None] + BAND_MEAN[:3, None, None]\n"
        "        rgb = np.transpose(np.clip(rgb / 0.25, 0, 1), (1, 2, 0))\n"
        "\n"
        "        axes[0, col].imshow(rgb)\n"
        "        axes[0, col].set_title(f'Patch {col}')\n"
        "        axes[0, col].axis('off')\n"
        "\n"
        "        axes[1, col].imshow(msk.numpy(), cmap=cmap, norm=norm)\n"
        "        axes[1, col].set_title('Ground truth')\n"
        "        axes[1, col].axis('off')\n"
        "\n"
        "        axes[2, col].imshow(pred, cmap=cmap, norm=norm)\n"
        "        axes[2, col].set_title('Prediction')\n"
        "        axes[2, col].axis('off')\n"
        "\n"
        "legend = [Patch(facecolor=c, label=n) for c, n in zip(LABEL_COLORS[1:], LABEL_NAMES[1:])]\n"
        "fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=9)\n"
        "plt.tight_layout()\n"
        "out = project_root / 'data' / 'val_predictions.png'\n"
        "plt.savefig(out, dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'Saved to {out}')"
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "pasture-cv (Python 3.11)",
            "language": "python",
            "name": "pasture-cv",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written {len(cells)} cells to {NB_PATH}")
