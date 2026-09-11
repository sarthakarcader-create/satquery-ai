"""
SatQuery AI — Headless Geospatial REST API Gateway
==================================================
Exposes the 48.7M parameter VLM and Agentic Controller as a high-performance REST API.
"""

import sys
import os
import io
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.satquery_model import SatQueryModel, SimpleTokenizer
from src.agents.controller import SatQueryController
from src.utils.geo_processor import read_raster_file, analyze_spectral_evidence

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("SatQuery-API")

app = FastAPI(
    title="SatQuery AI - Remote Sensing Intelligence API",
    description="REST API Gateway for the 48.7M Parameter Earth Observation VLM and Agentic Controller.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory raster registry (keyed by upload ID)
RASTER_CACHE: Dict[str, Dict[str, Any]] = {}

# Lazy system loader
_MODEL = None
_TOKENIZER = None
_CONTROLLER = None

def get_system():
    global _MODEL, _TOKENIZER, _CONTROLLER
    if _CONTROLLER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = SatQueryModel(embed_dim=256, num_heads=8, num_vision_layers=4, num_text_layers=3, patch_size=16)
        _TOKENIZER = SimpleTokenizer()
        _CONTROLLER = SatQueryController(model=_MODEL, tokenizer=_TOKENIZER, device=device)
        logger.info(f"Initialized SatQuery VLM on {device}")
    return _MODEL, _TOKENIZER, _CONTROLLER


class QueryRequest(BaseModel):
    query: str
    s2_id: Optional[str] = None
    s1_id: Optional[str] = None
    use_samples: Optional[bool] = False


@app.get("/api/health")
def health_check():
    cuda_avail = torch.cuda.is_available()
    device = "cuda" if cuda_avail else "cpu"
    return {
        "status": "operational",
        "service": "SatQuery AI Remote Sensing Gateway",
        "version": "2.0.0",
        "device": device,
        "cuda_available": cuda_avail,
    }


@app.post("/api/upload")
async def upload_raster(
    file: UploadFile = File(...),
    sensor: Optional[str] = Form("S2"),
):
    try:
        raw_bytes = await file.read()
        data, meta = read_raster_file(raw_bytes)
        if data is None:
            raise HTTPException(status_code=400, detail="Unable to decode raster image or GeoTIFF.")

        upload_id = str(uuid.uuid4())
        RASTER_CACHE[upload_id] = {
            "filename": file.filename,
            "sensor": sensor,
            "data": data,
            "meta": meta,
        }
        return {
            "upload_id": upload_id,
            "filename": file.filename,
            "sensor": sensor,
            "metadata": meta,
        }
    except Exception as exc:
        logger.error(f"Upload failure: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/metadata/{upload_id}")
def get_metadata(upload_id: str):
    if upload_id not in RASTER_CACHE:
        raise HTTPException(status_code=404, detail="Raster upload ID not found.")
    return RASTER_CACHE[upload_id]["meta"]


@app.get("/api/samples")
def list_samples():
    sample_dir = PROJECT_ROOT / "samples"
    s2_exists = (sample_dir / "sentinel2_sample.tif").exists()
    s1_exists = (sample_dir / "sentinel1_sample.tif").exists()
    return {
        "sentinel2_sample": s2_exists,
        "sentinel1_sample": s1_exists,
    }


@app.post("/api/analyze")
def analyze_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    _, _, controller = get_system()

    s2_data = None
    s1_data = None

    if req.use_samples:
        sample_dir = PROJECT_ROOT / "samples"
        s2_path = sample_dir / "sentinel2_sample.tif"
        if s2_path.exists():
            s2_data, _ = read_raster_file(s2_path.read_bytes())
        s1_path = sample_dir / "sentinel1_sample.tif"
        if s1_path.exists():
            s1_data, _ = read_raster_file(s1_path.read_bytes())
    else:
        if req.s2_id and req.s2_id in RASTER_CACHE:
            s2_data = RASTER_CACHE[req.s2_id]["data"]
        if req.s1_id and req.s1_id in RASTER_CACHE:
            s1_data = RASTER_CACHE[req.s1_id]["data"]

    s2_tensor = None
    if s2_data is not None:
        s2_arr = s2_data.copy()
        if s2_arr.shape[0] < 12:
            reps = int(np.ceil(12 / s2_arr.shape[0]))
            s2_arr = np.concatenate([s2_arr] * reps, axis=0)
        s2_tensor = torch.from_numpy(s2_arr[:12]).float()

    s1_tensor = None
    if s1_data is not None:
        s1_arr = s1_data.copy()
        if s1_arr.shape[0] < 2:
            reps = int(np.ceil(2 / s1_arr.shape[0]))
            s1_arr = np.concatenate([s1_arr] * reps, axis=0)
        s1_tensor = torch.from_numpy(s1_arr[:2]).float()

    result = controller.process_query(
        query=req.query,
        s2_image=s2_tensor,
        s1_image=s1_tensor,
        raw_s2_raster=s2_data,
        raw_s1_raster=s1_data,
    )
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
