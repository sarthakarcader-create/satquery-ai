# 🛰️ SatQuery AI

**An Agentic Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis**

SatQuery AI is an intelligent satellite imagery analysis system that accepts natural-language queries and provides evidence-grounded responses about observed scenes. It processes Sentinel-1 (SAR) and Sentinel-2 (optical) satellite imagery through a custom Vision-Language Model with 48.7M parameters.

## 🎯 Supported Tasks

| Task | Input | Output | Example Query |
|------|-------|--------|---------------|
| **Binary VQA** | S1+S2 image + question | Yes/No + confidence | "Is there water in this image?" |
| **MCQ VQA** | S1+S2 image + choices | Selected option | "Which covers more area: forest or water?" |
| **Bounding Box** | S1+S2 image + question | [x1,y1,x2,y2] coordinates | "Highlight the forested area" |
| **Captioning** | S1+S2 image | Natural language description | "Describe the land cover in this scene" |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 USER INTERFACE (Gradio Web App)                  │
│          Upload S1/S2 Images + Type NL Question                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AGENTIC CONTROLLER (Query Router)                │
│  Step 1: Input Validation → Step 2: Query Classification        │
│  Step 3: Model Selection → Step 4: Execution → Step 5: Output   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│            SATQUERY VISION-LANGUAGE MODEL (48.7M params)        │
│                                                                  │
│  S2 (12 bands) → Projection ─┐                                  │
│                                ├→ ViT Encoder (6 layers)         │
│  S1 (2 bands)  → Projection ─┘         │                        │
│                                         ▼                        │
│  Question → Text Encoder (4 layers) → Cross-Modal Fusion        │
│                                         │                        │
│                        ┌────────────────┼────────────────┐       │
│                        ▼                ▼                ▼       │
│                   Binary VQA         BBox           Captioning   │
│                   Head               Head           Head         │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
satquery-ai/
├── app.py                          # Gradio web interface
├── requirements.txt                # Python dependencies
├── SatQuery_AI_Colab.ipynb         # Google Colab notebook (complete pipeline)
├── data/
│   ├── processed/                  # Annotation data
│   │   ├── prototype_pairs.csv     # 100 S1/S2 pairs
│   │   └── prototype_annotations.parquet  # 2,085 Q&A annotations
│   └── raw/                        # Downloaded GeoTIFF patches
├── src/
│   ├── data/
│   │   ├── download_patches.py     # Streaming tar.zst extractor
│   │   ├── image_loader.py         # Rasterio S1+S2 loader
│   │   ├── annotation_loader.py    # Parquet annotation loader
│   │   └── satquery_dataset.py     # PyTorch Dataset + DataLoader
│   ├── models/
│   │   ├── satquery_model.py       # Vision-Language Model (48.7M)
│   │   └── trainer.py              # Training loop + loss + metrics
│   └── agents/
│       └── controller.py           # Agentic query router
└── start_s2_download.sh            # Background download launcher
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `SatQuery_AI_Colab.ipynb`
3. Enable GPU: Runtime → Change runtime type → T4 GPU
4. Run all cells: Runtime → Run all
5. Gradio app launches with public shareable link

### Option 2: Local Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/satquery-ai.git
cd satquery-ai

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch web interface
python app.py
```

### Option 3: Download Real Satellite Data
```bash
# Start downloading Sentinel-2 patches from Zenodo (~63GB archive)
./start_s2_download.sh

# Monitor progress
tail -f data/raw/download_s2.log
```

## 🛠️ Technologies Used

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.12 | Core language |
| **Deep Learning** | PyTorch 2.13 | Model building, training, inference |
| **Transformers** | PyTorch nn.TransformerEncoder | Vision/Text encoders, Fusion layer |
| **Geospatial I/O** | Rasterio | Reads GeoTIFF satellite bands |
| **Data Processing** | Pandas, NumPy | Annotation handling, array operations |
| **Web Interface** | Gradio 5.x | Interactive web GUI |
| **Data Storage** | Parquet | Efficient annotation storage |
| **Archive Streaming** | zstandard + tarfile | Streams Zenodo tar.zst archives |

## 📊 Model Details

- **Total Parameters:** 48.7M
- **Vision Encoder:** Vision Transformer (ViT) with 6 layers, 8 attention heads
- **Text Encoder:** Transformer with 4 layers, 8 attention heads
- **Fusion Layer:** Cross-attention mechanism
- **Task Heads:** Binary VQA, BBox regression, Autoregressive captioning
- **Input:** 12-band Sentinel-2 + 2-band Sentinel-1 (14 channels total)
- **Image Size:** 120×120 pixels (configurable)

## 📈 Training

The model trains on BigEarthNet.txt dataset with:
- **Optimizer:** AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler:** Cosine Annealing
- **Loss Functions:**
  - Binary/MCQ VQA: Binary Cross-Entropy
  - Bounding Box: Smooth L1 Loss
  - Captioning: Cross-Entropy (next token prediction)
- **Gradient Clipping:** max_norm=1.0

## 🎓 Datasets

| Dataset | Purpose | Source |
|---------|---------|--------|
| BigEarthNet.txt | Primary training data | HuggingFace (BIFOLD-BigEarthNetv2-0) |
| BigEarthNet-S1 | Sentinel-1 SAR imagery | Zenodo (10891137) |
| BigEarthNet-S2 | Sentinel-2 optical imagery | Zenodo (10891137) |

## 📝 SIH Problem Statement

SatQuery AI addresses the Smart India Hackathon problem of developing an agentic vision-language assistant for analyzing single and paired remote-sensing images through natural-language queries.

**Key Requirements Met:**
- ✅ Single-image VQA (mandatory baseline)
- ✅ Captioning (additional single-image task)
- ✅ Bounding Box grounding (alternative additional task)
- ✅ Remote-sensing adaptation via BigEarthNet.txt
- ✅ Agentic orchestration with execution traces
- ✅ Interactive GUI (Gradio web interface)

**Planned Extensions:**
- 🔜 Multi-temporal change detection
- 🔜 Cross-modal S1+S2 specialized analysis
- 🔜 Fine-tuning on real satellite imagery
- 🔜 Evaluation on VRSBench/RSVQA benchmarks

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BigEarthNet Dataset](https://bigearth.net/) for remote-sensing training data
- [Zenodo](https://zenodo.org/) for satellite imagery archives
- [HuggingFace](https://huggingface.co/) for dataset hosting
- [Sentinel Hub](https://www.sentinel-hub.com/) for satellite data access

---

**Built for Smart India Hackathon (SIH)**
