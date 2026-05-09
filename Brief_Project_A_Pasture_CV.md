# Project Brief — Pasture / Crop Health Detection (Computer Vision)

> **使用说明:** 把这份简报直接发到一个新对话(或新模型)里,作为开题输入。简报是自包含的——包含候选人背景、战略动机、技术方案、数据源、里程碑、风险——不需要重述前情。

---

## 1. 候选人背景(Context)

- **姓名:** Harlan Li
- **当前 level:** Junior(MSc Data Science 在读,2025 Mar – 2026 末);intermediate 及以下岗位
- **地点:** Auckland, NZ;NZ post-study work visa 预计 2026 年底获得
- **现有项目组合(全部 NLP/LLM 方向,这是这个项目要解决的盲区):**
  1. Real-time Fact-check Stream — Mistral-7B QLoRA fine-tuning, Kafka → vLLM → PostgreSQL pipeline at 1.31s/claim, Prometheus + Grafana drift alerting
  2. Multiagent Document Assistant — agent-based RAG, multi-LLM routing (OpenAI + AWS Bedrock), Qdrant vector retrieval
  3. Misinformation Risk Assessment — .NET 8 + FastAPI + DistilBERT hybrid ensemble, Azure deployment via Bicep IaC + GitHub Actions CI/CD
  4. End-to-End Retail ML Platform (forecasting + recommendation, MAPE 2.00%)
  5. Customer Lifetime Value Prediction (R + XGBoost, 2,000× revenue spread)
  6. NZ Retail Trend Analysis, US Honey Production Analytics, TGN Hate Speech, Honey ETL Pipeline

- **MLOps 标准(很重要,新项目要保持同等水准):**
  - 一律 FastAPI + REST API contract + Docker Compose 部署
  - Prometheus + Grafana 观测 + drift / quality alerting
  - Azure 偏好(App Service / Container Apps / Bicep IaC),也熟 AWS
  - GitHub Actions CI/CD
  - 结构化 JSON 输出 + schema validation (pandera)

## 2. 为什么是这个项目(战略动机)

NZ AI 雇主分析显示三大缺口:Computer Vision 是最大的一个。CV 项目能一次性打开三大 NZ 雇主池,这些都是 LLM/NLP 项目无法触及的:

| 雇主池 | 代表公司 | CV 用例 |
|---|---|---|
| **Agritech**(NZ 经济支柱) | Fonterra, LIC, Halter, Gallagher | 牧草健康监测、奶牛识别、作物胁迫检测 |
| **政府 / 保育** | DOC, MPI, MfE, Predator Free 2050 | 野生动物 ID、土地覆盖、入侵物种监测 |
| **遥感 / GIS** | LINZ, Beca, Jacobs, Tonkin+Taylor | 灾害响应、土地利用、森林监测 |

候选人的 MLOps 工程能力(已经超过多数 junior)+ NZ 本地 CV 项目 = 在 agritech / 政府这两个池里立刻形成差异化。

## 3. 项目目标(Project Goal)

> **一句话目标:** 用免费的 Sentinel-2 卫星影像(或低成本无人机数据集)训练一个**牧草/作物健康分割模型**,部署成生产级 REST API,带完整的 MLOps observability,让候选人能在 4–6 周内交付一个 production-grade 简历项目。

**功能定义:** 输入一块 NZ 草地/农田的卫星影像(或 RGB 无人机图),输出 segmentation mask(健康/胁迫/裸地/水体)+ 整体健康评分(0–100)+ NDVI 衍生指标。

## 4. 技术方案

### 4.1 数据源(全部免费、合法)

| 数据源 | 类型 | NZ 适配度 | 用途 |
|---|---|---|---|
| **Sentinel-2 L2A**(Copernicus 通过 AWS Open Data 或 Sentinel Hub free tier) | 多光谱卫星(10m 分辨率) | ⭐⭐⭐⭐⭐ NZ 全国覆盖 | NDVI / NDRE / NDWI 计算,粗粒度分割 |
| **Agriculture-Vision Dataset**(2020 CVPR Challenge) | 标注的 RGB+NIR 农田瓦片 | ⭐⭐⭐ US 数据但形态相似 | 监督学习的标签源 |
| **LandCoverNet** | 全球土地覆盖标注 | ⭐⭐⭐⭐ 含 Australasia | 预训练 backbone |
| **NZ LRIS Portal**(landcareresearch.co.nz/tools/lris) | NZ 土地资源/覆盖矢量数据 | ⭐⭐⭐⭐⭐ 官方 | 验证集 / 真值 |
| **DroneDeploy Open Crop Dataset** | 高分辨率无人机 RGB | ⭐⭐⭐ | 高分辨率 fine-tune |

> **入门路径:** 直接用 **Sentinel Hub Python API** 下载 NZ 区域(比如 Waikato 或 Canterbury)2024 年 4 个季度的 Sentinel-2 L2A 影像,配 LRIS 的土地覆盖向量做 weak supervision,先做"草地 vs 非草地"二分类作为 v0,再加"健康/胁迫"。

### 4.2 模型架构

**Baseline → Better → Best 三档,按时间预算选:**

| 档位 | 模型 | 训练时间(单卡 T4) | 性能预期 |
|---|---|---|---|
| Baseline(必须) | **U-Net** + ResNet50 backbone(timm)+ NDVI 通道 | 2 天 | mIoU ~0.55 |
| Better(目标) | **DeepLabV3+** with EfficientNet backbone | 3 天 | mIoU ~0.65 |
| Best(stretch) | **YOLOv8-seg** fine-tune,或 **SegFormer-B0**(transformer-based,2024 SOTA) | 4 天 | mIoU ~0.70 |

> **建议:** 至少做 baseline + better 两档,体现"做了 ablation"的科研感觉,跟候选人现有 Retail ML Platform 的"benchmarked 8 models"叙事一致。

### 4.3 服务化与 MLOps(必须复用候选人的标准)

```
[ Sentinel-2 tile ingest ]
        ↓
[ FastAPI inference service ]  ← Docker Compose
        ↓
[ Mask + Health score JSON output ]
        ↓
[ PostgreSQL: 历史推理结果 + drift metrics ]
        ↓
[ Prometheus → Grafana dashboard ]
        ↓
[ Drift alerts: NDVI 分布漂移 / 推理时延异常 ]
```

**关键 deliverables:**
- `POST /infer` 接收 base64 影像或 S3 URI,返回 `{mask: ..., health_score: ..., ndvi_stats: {...}}`
- pandera schema 验证输入/输出
- MLflow 跟踪所有训练实验(数据版本、超参、metric)
- Grafana 面板至少包含:推理时延、每日 NDVI 分布漂移、健康评分时序
- README 里给出**复现命令**(`docker compose up` + 一条 curl)

### 4.4 简历叙事(Resume Bullets,最终成稿应该长这样)

```
Pasture Health Segmentation — Production CV Pipeline
Sentinel-2 · PyTorch · DeepLabV3+ · FastAPI · MLflow · Prometheus · Grafana

· Trained DeepLabV3+ (EfficientNet backbone) on Sentinel-2 multispectral tiles
  + NZ LRIS land-cover labels for pasture/crop health segmentation;
  benchmarked 3 architectures (U-Net / DeepLabV3+ / SegFormer), best mIoU 0.6X.
· Production MLOps pipeline: FastAPI inference service with structured JSON
  contracts, MLflow experiment tracking, pandera input validation, Docker Compose
  full-stack deployment.
· Prometheus + Grafana observability with NDVI distribution drift alerting and
  per-region health-score time series — directly applicable to NZ agritech /
  precision farming use cases.
```

## 5. 时间线(4–6 周,假设每周 8–12 小时)

| 周 | Milestone | 检查点 |
|:-:|---|---|
| 1 | 数据获取 + EDA;Sentinel Hub 配好,下载 NZ Waikato + Canterbury 4 季度数据;LRIS 土地覆盖矢量对齐 | 一份 Quarto / Jupyter EDA notebook |
| 2 | Baseline U-Net 跑通 + 端到端训练流程 + MLflow 接入 | mIoU 数字在 wandb/mlflow 里可查 |
| 3 | DeepLabV3+ + ablation;固定最优模型 | benchmark table |
| 4 | FastAPI 服务 + Docker Compose + REST 测试 | curl 能拿到合理 JSON |
| 5 | Prometheus + Grafana 观测 + drift alert 规则 | Grafana dashboard 截图 |
| 6 | README、demo 视频(2 分钟)、GitHub repo cleanup;最后写简历 bullet | 公开 repo + LinkedIn 帖子 |

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Sentinel-2 数据下载/对齐复杂 | 用 [`sentinelhub-py`](https://sentinelhub-py.readthedocs.io/) free tier,先做单瓦片 PoC 再扩 |
| 没有像素级真值标签 | weak supervision: 把 LRIS 矢量栅格化成弱标签,再加少量人工修正 |
| 模型训练显卡需求 | Colab Pro 单 T4 即可;也可用 Kaggle 30h/周免费 GPU |
| 4 周做不完所有 milestone | 切到 baseline-only 路径,至少跑通 U-Net + FastAPI 两件事,简历仍可挂 |
| Sentinel-2 分辨率不够细(10m) | 备选方案:用 DroneDeploy Open Crop dataset 做高分辨率版本 |

## 7. 候选人需要决定的事

在新对话开始时,先回答这几个问题(可以让 Claude 帮你定):

1. **数据源选型:** Sentinel-2(广覆盖、低分辨率) vs 无人机数据集(高分辨率、地理范围窄)?
2. **目标雇主类型:** agritech 公司(Halter/LIC,看重 edge + 牲畜)还是政府/保育(DOC/MPI,看重广区 + 地理空间)?会影响 v1 数据集选择。
3. **算力:** Colab Pro($10/月) vs 本地 GPU vs 完全 CPU+小数据
4. **可投入时间:** 每周 8 小时(6 周完成) vs 每周 15 小时(4 周完成)?
5. **是否要把训练好的模型放 HuggingFace Hub** 一并展示(增强可信度,但多花半天)?

## 8. 给新对话的开场指令(直接复制给对方)

> "我要做一个 Computer Vision 项目,叫 Pasture / Crop Health Detection,目的是给我的 AI Engineer 简历补一个非 NLP 的、有 NZ 行业角度的项目。我已经有这份简报(附件)。请先帮我:(1) 决定第 7 节里的 5 个开放问题;(2) 把第 4.1 节的数据获取脚本写出来,我今晚就要能拉到第一批 Sentinel-2 影像在 Jupyter 里看到 NDVI 图。"
