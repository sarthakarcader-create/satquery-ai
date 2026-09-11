# 🛰️ SatQuery AI — Earth Observation Intelligence Portal

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Smart India Hackathon 2026](https://img.shields.io/badge/SIH-2026_Selection-F37021?style=flat&logo=satellite)](https://www.sih.gov.in/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Rasterio](https://img.shields.io/badge/Geospatial-Rasterio_&_GDAL-228B22?style=flat)](https://rasterio.readthedocs.io/)

**SatQuery AI** is an autonomous, evidence-grounded remote sensing intelligence platform and GIS workstation developed by **Team Aarohan** for the **Smart India Hackathon (SIH)**.

Instead of requiring complex desktop GIS software, manual band mathematics, or proprietary remote sensing tools, SatQuery AI allows users to query Earth Observation imagery using conversational, natural-language questions. Every response is verified through physical spectral indices (NDWI, NDVI, SAR dB backscatter) and localized with pixel-accurate bounding box coordinates.

---

## 🚀 Key Innovations

1. **Dual-Stream Multi-Modal Ingestion**: Handles both Sentinel-2 (12-band multispectral optical) and Sentinel-1 (C-band dual-polarization SAR radar).
2. **Autonomous Agentic Controller**: Routes queries to specialized neural vision heads while validating raster georeferencing, spectral bands, and query intents.
3. **48.7M Parameter PyTorch VLM**: Custom Vision Transformer (ViT) architecture featuring cross-modal attention fusion between text questions and satellite image embeddings.
4. **Physical Spectral Grounding**: Combines neural logits with real-time remote sensing analytics (NDWI water masking, NDVI vegetation vigor, and SAR polarimetric surface roughness) for 100% verifiable outputs.
5. **Interactive 3-Panel GIS Workstation**: Features multi-band composite switching (True Color RGB vs False Color NIR), optical/SAR cross-fade opacity blending, real-time extent metadata, and visual evidence overlays.
6. **1-Click SIH Live Demo Mode**: Built-in sample validation scenes allow immediate, zero-friction demonstration without requiring external GeoTIFF downloads.

---

## 🎯 Supported Analytical Tasks

| Task Type | Target Sensors | Model Head & Grounding | Example Question |
|-----------|----------------|------------------------|------------------|
| **Binary VQA** | S1 (SAR) + S2 (Optical) | Binary Head + NDWI / NDVI verification | *"Is there water in this image?"* |
| **Bounding Box Grounding** | S1 (SAR) + S2 (Optical) | BBox Regressor + Spectral Cluster Mask | *"Highlight the forested area"* |
| **Scene Captioning** | S1 (SAR) + S2 (Optical) | Vision-Language Decoder + Scene Stats | *"Describe the land cover in this scene"* |
| **Infrastructure Detection** | S1 (SAR) Radar | Polarimetric Double-Bounce Extraction | *"Are there buildings visible?"* |
| **Crop & Agricultural Health** | S2 (NIR / Red) | Chlorophyll Vigor Index | *"Locate agricultural fields"* |

---

## 🏗️ System Architecture

```
                               ┌───────────────────────────────────────────────────────────┐
                               │                    SATQUERY AI PORTAL                     │
                               │   (Landing Page + 3-Panel GIS Operations Workstation)     │
                               └─────────────────────────────┬─────────────────────────────┘
                                                             │
                                ┌────────────────────────────┴────────────────────────────┐
                                │                                                         │
                     [Streamlit Entrypoint: app.py]                       [FastAPI Gateway: api.py]
                     (Direct Streamlit Cloud Deploy)                       (REST API & Headless)
                                │                                                         │
                                └────────────────────────────┬────────────────────────────┘
                                                             │
                                                             ▼
                                      ┌─────────────────────────────────────────┐
                                      │        GEOSPATIAL INGESTION ENGINE      │
                                      │   - Rasterio GeoTIFF Parser             │
                                      │   - Metadata Extractor (CRS/Transform)  │
                                      │   - Multi-spectral (12-Band S2) Loader  │
                                      │   - SAR Dual-Pol (VV/VH S1) Loader      │
                                      │   - Percentile Histogram Stretcher      │
                                      └──────────────────────┬──────────────────┘
                                                             │
                                                             ▼
                                      ┌─────────────────────────────────────────┐
                                      │        SATQUERY AGENTIC CONTROLLER      │
                                      │       (src/agents/controller.py)        │
                                      │   1. Input Validator                    │
                                      │   2. Query Intent Classifier            │
                                      │   3. Multi-Specialist Dispatcher        │
                                      │   4. Execution Tracer                   │
                                      └──────────────────────┬──────────────────┘
                                                             │
                                                             ▼
                                      ┌─────────────────────────────────────────┐
                                      │        48.7M PARAMETER PYTORCH VLM      │
                                      │      (src/models/satquery_model.py)     │
                                      │   - Vision Transformer (ViT S1+S2)      │
                                      │   - Text Transformer Query Encoder      │
                                      │   - Cross-Modal Attention Fusion        │
                                      │   - Task Heads: Binary, BBox, Caption   │
                                      └──────────────────────┬──────────────────┘
                                                             │
                                                             ▼
                                      ┌─────────────────────────────────────────┐
                                      │        EVIDENCE & RESULT SYNTHESIS      │
                                      │   - Bounding Box Regressor              │
                                      │   - Grounded Spectral Masks (NDVI/NDWI) │
                                      │   - Scientific Confidence Estimation    │
                                      │   - Millisecond Step-by-Step Trace      │
                                      └──────────────────────┬──────────────────┘
```

---

## ⚡ Quickstart Guide

### 1. Local Python Environment
```bash
# Clone the repository
git clone https://github.com/sarthakarcader-create/satquery-ai.git
cd satquery-ai

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the interactive GIS Workstation
streamlit run app.py
```

### 2. Run Headless REST API Gateway
```bash
python3 api.py
# API is live at http://localhost:8000
# Interactive Swagger docs at http://localhost:8000/docs
```

### 3. Docker Deployment
```bash
docker-compose up --build
```

---

## 🎬 3-Minute SIH Presentation Script

1. **The Hook (30 sec)**: Open the live landing page. Present the hero video and explain that remote sensing analysts waste days manually downloading gigabyte-scale GeoTIFFs and running complex band mathematics. SatQuery AI solves this with conversational, multimodal satellite intelligence.
2. **Launch Portal (30 sec)**: Click **"Launch SatQuery Workstation"**. Show the 3-panel operations workspace (Data Ingestion on left, Interactive Geospatial Canvas in center, Agentic Query Inspector on right).
3. **1-Click Ingestion (30 sec)**: Click **"Load S2 Optical"** (or **"Load Both"**). Demonstrate real-time metadata extraction (12 bands, `EPSG:32643` UTM projection, 10m GSD spatial resolution). Switch between **Natural Color RGB** and **Color Infrared NIR** to show vegetative vigor.
4. **Natural Language Query (45 sec)**:
   - Select query preset: *"Is there water in this image?"*
   - Click **"Execute Agentic Analysis"**.
   - Show the sequential progress bar (Validation → Classification → Specialist Routing → PyTorch Inference → Spectral Grounding).
   - Display the final result: `Yes, water body detected` with **88%+ confidence** and physical **NDWI surface percentage**.
5. **Localization & Evidence (45 sec)**:
   - Ask: *"Highlight the forested area"*.
   - Point to the center viewer: The bounding box and cyan overlay highlight the exact spatial forest canopy in real time.
   - Expand the **Millisecond Execution Trace** to show jury members the transparent breakdown of latency for each stage.

---

## 👥 Team Aarohan
* Developed for **Smart India Hackathon (SIH)**.
* Focus Area: Earth Observation, Multimodal Remote Sensing, and Agentic AI.
