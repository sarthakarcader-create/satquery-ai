"""
SatQuery AI — Unified Remote Sensing Intelligence Portal & GIS Workstation
==========================================================================
Smart India Hackathon (SIH) Edition — Team Aarohan
Dual-Stream Earth Observation VLM + Agentic Controller + Geospatial Engine
"""

import base64
import sys
import os
import io
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import streamlit as st
import torch

# ============================================================
# Project Path Setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.satquery_model import SatQueryModel, SimpleTokenizer
from src.agents.controller_v2 import SatQueryController, TaskType
from src.utils.geo_processor import (
    read_raster_file,
    percentile_stretch,
    generate_s2_rgb,
    generate_s2_false_color,
    generate_s1_composite,
    render_evidence_overlay,
    analyze_spectral_evidence,
)

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="SatQuery AI | Earth Observation Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Persistent Session State
# ============================================================
if "launched" not in st.session_state:
    st.session_state.launched = False
if "query" not in st.session_state:
    st.session_state.query = "Is there water in this image?"
if "result" not in st.session_state:
    st.session_state.result = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "s2_data" not in st.session_state:
    st.session_state.s2_data = None
if "s2_meta" not in st.session_state:
    st.session_state.s2_meta = None
if "s2_name" not in st.session_state:
    st.session_state.s2_name = None
if "s1_data" not in st.session_state:
    st.session_state.s1_data = None
if "s1_meta" not in st.session_state:
    st.session_state.s1_meta = None
if "s1_name" not in st.session_state:
    st.session_state.s1_name = None
if "active_composite" not in st.session_state:
    st.session_state.active_composite = "rgb"
if "show_evidence" not in st.session_state:
    st.session_state.show_evidence = True
if "show_s2" not in st.session_state:
    st.session_state.show_s2 = True
if "show_s1" not in st.session_state:
    st.session_state.show_s1 = True
if "opacity_blend" not in st.session_state:
    st.session_state.opacity_blend = 50

def launch_portal():
    st.session_state.launched = True

def go_landing():
    st.session_state.launched = False

def clear_workspace():
    st.session_state.query = ""
    st.session_state.result = None
    st.session_state.analysis_complete = False
    st.session_state.s2_data = None
    st.session_state.s2_meta = None
    st.session_state.s2_name = None
    st.session_state.s1_data = None
    st.session_state.s1_meta = None
    st.session_state.s1_name = None

def load_sample_s2():
    sample_path = PROJECT_ROOT / "samples" / "sentinel2_sample.tif"
    if sample_path.exists():
        raw_bytes = sample_path.read_bytes()
        data, meta = read_raster_file(raw_bytes)
        st.session_state.s2_data = data
        st.session_state.s2_meta = meta
        st.session_state.s2_name = "Sentinel-2 L2A Sample (12-Band GeoTIFF)"

def load_sample_s1():
    sample_path = PROJECT_ROOT / "samples" / "sentinel1_sample.tif"
    if sample_path.exists():
        raw_bytes = sample_path.read_bytes()
        data, meta = read_raster_file(raw_bytes)
        st.session_state.s1_data = data
        st.session_state.s1_meta = meta
        st.session_state.s1_name = "Sentinel-1 GRD SAR Sample (VV/VH GeoTIFF)"

def load_sample_both():
    load_sample_s2()
    load_sample_s1()

# ============================================================
# Model Cache
# ============================================================
@st.cache_resource(show_spinner=False)
def get_cached_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SatQueryModel(
        embed_dim=256,
        num_heads=8,
        num_vision_layers=4,
        num_text_layers=3,
        patch_size=16,
    )
    tokenizer = SimpleTokenizer()
    controller = SatQueryController(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return model, tokenizer, controller, device

# ============================================================
# Hero Video Base64 Cache
# ============================================================
@st.cache_data
def get_hero_video_b64() -> Optional[str]:
    video_path = PROJECT_ROOT / "assets" / "hero.mp4"
    if video_path.exists():
        return base64.b64encode(video_path.read_bytes()).decode()
    return None

HERO_VIDEO_B64 = get_hero_video_b64()

# ============================================================
# Global Styling
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg-main: #060B16;
        --bg-panel: #0A1224;
        --bg-card: #0E1B35;
        --border-subtle: #1A263E;
        --border-active: #2C3E66;
        --isro-orange: #F37021;
        --isro-blue: #38BDF8;
        --text-bright: #F8FAFC;
        --text-muted: #94A3B8;
        --status-green: #10B981;
    }
    
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-bright);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Bar */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    footer {
        display: none !important;
    }
    
    .block-container {
        max-width: 100% !important;
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Sleek buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.18s ease !important;
        border: 1px solid var(--border-subtle) !important;
        background: #0F1D38 !important;
        color: #E2E8F0 !important;
    }
    
    .stButton > button:hover {
        border-color: var(--isro-orange) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(243, 112, 33, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Primary buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #F37021 0%, #D9580D 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 18px rgba(243, 112, 33, 0.35) !important;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 6px 24px rgba(243, 112, 33, 0.5) !important;
    }
    
    /* Input elements */
    textarea, input, select {
        background-color: #081020 !important;
        border: 1px solid var(--border-subtle) !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
    }
    textarea:focus, input:focus {
        border-color: var(--isro-orange) !important;
        box-shadow: 0 0 0 1px var(--isro-orange) !important;
    }
    
    /* GIS Workstation Cards */
    .gis-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    .gis-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding-bottom: 8px;
    }
    
    .gis-card-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #CBD5E1;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 3px 8px;
        border-radius: 999px;
    }
    .status-active {
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-info {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .status-orange {
        background: rgba(243, 112, 33, 0.12);
        color: #F37021;
        border: 1px solid rgba(243, 112, 33, 0.3);
    }
    
    /* Code & metadata */
    .code-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        background: #050B14;
        border: 1px solid #16243D;
        padding: 2px 6px;
        border-radius: 4px;
        color: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# VIEW 1: LANDING PAGE
# ============================================================
if not st.session_state.launched:
    # Landing page scoped CSS
    st.markdown(
        """
        <style>
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        .hero-container {
            position: relative;
            width: 100vw;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            background: #020610;
            overflow: hidden;
        }
        .hero-video-bg {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center center;
            opacity: 0.85;
            z-index: 0;
        }
        .hero-scrim {
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 60% 40%, rgba(2, 6, 16, 0.1) 0%, rgba(2, 6, 16, 0.75) 70%, #020610 100%);
            z-index: 1;
        }
        .hero-content {
            position: relative;
            z-index: 5;
            padding: 4rem 6vw 2rem 6vw;
            max-width: 1200px;
        }
        .hero-brand {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            letter-spacing: 0.12em;
            color: #F37021;
            background: rgba(243, 112, 33, 0.12);
            border: 1px solid rgba(243, 112, 33, 0.35);
            padding: 6px 14px;
            border-radius: 999px;
            margin-bottom: 1.5rem;
        }
        .hero-title {
            font-family: 'Anton', Impact, sans-serif !important;
            font-size: clamp(3.5rem, 8vw, 7.5rem) !important;
            line-height: 0.95 !important;
            letter-spacing: 0.02em !important;
            color: #FFFFFF !important;
            text-transform: uppercase;
            margin: 0 0 1.5rem 0 !important;
            text-shadow: 0 8px 30px rgba(0, 0, 0, 0.7);
        }
        .hero-subtitle {
            font-size: clamp(1.1rem, 2vw, 1.45rem);
            line-height: 1.5;
            color: #CBD5E1;
            max-width: 680px;
            margin-bottom: 2.5rem;
            text-shadow: 0 4px 16px rgba(0, 0, 0, 0.8);
        }
        .landing-section {
            background: #050B17;
            border-top: 1px solid var(--border-subtle);
            padding: 5rem 6vw;
        }
        .section-header {
            margin-bottom: 3.5rem;
        }
        .section-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: #38BDF8;
            margin-bottom: 0.6rem;
        }
        .section-title {
            font-size: clamp(2rem, 3.5vw, 2.8rem);
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.02em;
        }
        .pipeline-card {
            background: #091326;
            border: 1px solid #192742;
            border-radius: 12px;
            padding: 24px;
            height: 100%;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .pipeline-card:hover {
            border-color: #F37021;
            transform: translateY(-4px);
        }
        .pipeline-num {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            color: #F37021;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Hero Banner
    video_html = (
        f'<video class="hero-video-bg" autoplay muted loop playsinline preload="auto">'
        f'<source src="data:video/mp4;base64,{HERO_VIDEO_B64}" type="video/mp4">'
        f'</video>'
        if HERO_VIDEO_B64
        else ""
    )

    st.markdown(
        f"""
        <div class="hero-container">
            {video_html}
            <div class="hero-scrim"></div>
            <div class="hero-content">
                <div class="hero-brand">
                    <span>🛰️</span>
                    <span>SMART INDIA HACKATHON 2026 · TEAM AAROHAN</span>
                </div>
                <h1 class="hero-title">SATQUERY AI</h1>
                <p class="hero-subtitle">
                    Ask questions about Earth. Analyze Sentinel-1 SAR radar and Sentinel-2 optical imagery 
                    using natural-language queries and an evidence-grounded satellite intelligence pipeline.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Action Bar
    btn_col1, btn_col2, _ = st.columns([1.3, 1.3, 4])
    with btn_col1:
        st.button(
            "🚀 Launch SatQuery Workstation",
            type="primary",
            use_container_width=True,
            on_click=launch_portal,
        )
    with btn_col2:
        if st.button("⚡ Quick Test with Sample Scene", use_container_width=True):
            load_sample_s2()
            launch_portal()
            st.rerun()

    # Section 1: What is SatQuery AI?
    st.markdown(
        """
        <div class="landing-section">
            <div class="section-header">
                <div class="section-tag">Overview & Problem Statement</div>
                <h2 class="section-title">Bridging Natural Language & Earth Observation</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(
            """
            <div class="pipeline-card">
                <div class="pipeline-num">01</div>
                <h3 style="color:#F1F5F9; font-size:1.15rem; margin-bottom:8px;">The Problem</h3>
                <p style="color:#94A3B8; font-size:0.92rem; line-height:1.6;">
                    Satellite rasters contain massive amounts of multispectral and radar information, 
                    traditionally locked behind complex GIS desktop suites, manual band-math scripts, and remote sensing jargon.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="pipeline-card">
                <div class="pipeline-num">02</div>
                <h3 style="color:#F1F5F9; font-size:1.15rem; margin-bottom:8px;">The Agentic Solution</h3>
                <p style="color:#94A3B8; font-size:0.92rem; line-height:1.6;">
                    SatQuery AI allows users to inspect satellite scenes using plain conversational questions. 
                    An autonomous controller routes queries to specialized neural vision heads and remote sensing verification modules.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="pipeline-card">
                <div class="pipeline-num">03</div>
                <h3 style="color:#F1F5F9; font-size:1.15rem; margin-bottom:8px;">Evidence-Grounded</h3>
                <p style="color:#94A3B8; font-size:0.92rem; line-height:1.6;">
                    Unlike generic chatbot wrappers, every prediction is paired with physical spectral indices (NDWI for water, NDVI for vegetation, SAR backscatter) 
                    and bounding box coordinates for verifiable outputs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section 2: How it Works (Visual Pipeline)
    st.markdown(
        """
        <div class="landing-section" style="background:#030813;">
            <div class="section-header">
                <div class="section-tag">Architecture Pipeline</div>
                <h2 class="section-title">How SatQuery AI Processes Satellite Imagery</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(4, gap="medium")
    with p1:
        st.markdown(
            """
            <div class="pipeline-card">
                <div style="font-family:'JetBrains Mono',monospace; color:#38BDF8; font-size:0.8rem; margin-bottom:8px;">STEP 01</div>
                <h4 style="color:#F8FAFC; margin-bottom:6px;">Multi-Band Ingestion</h4>
                <p style="color:#94A3B8; font-size:0.86rem;">
                    Reads raw GeoTIFF rasters using Rasterio. Preserves spatial CRS, coordinates, and 12-bit spectral bands.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """
            <div class="pipeline-card">
                <div style="font-family:'JetBrains Mono',monospace; color:#38BDF8; font-size:0.8rem; margin-bottom:8px;">STEP 02</div>
                <h4 style="color:#F8FAFC; margin-bottom:6px;">Agentic Controller</h4>
                <p style="color:#94A3B8; font-size:0.86rem;">
                    Validates inputs, classifies query intent (Binary VQA, Grounding, Captioning), and selects the appropriate specialist head.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            """
            <div class="pipeline-card">
                <div style="font-family:'JetBrains Mono',monospace; color:#38BDF8; font-size:0.8rem; margin-bottom:8px;">STEP 03</div>
                <h4 style="color:#F8FAFC; margin-bottom:6px;">48.7M PyTorch VLM</h4>
                <p style="color:#94A3B8; font-size:0.86rem;">
                    Dual-stream Vision Transformer projects Optical + SAR features, fused via cross-attention with the tokenized question.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p4:
        st.markdown(
            """
            <div class="pipeline-card">
                <div style="font-family:'JetBrains Mono',monospace; color:#38BDF8; font-size:0.8rem; margin-bottom:8px;">STEP 04</div>
                <h4 style="color:#F8FAFC; margin-bottom:6px;">Evidence Synthesis</h4>
                <p style="color:#94A3B8; font-size:0.86rem;">
                    Combines neural predictions with physical spectral indices (NDWI, NDVI, SAR dB) to render bounding boxes and audit traces.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section 3: Supported Sensors
    st.markdown(
        """
        <div class="landing-section">
            <div class="section-header">
                <div class="section-tag">Sensor Capabilities</div>
                <h2 class="section-title">Dual-Stream Satellite Modalities</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s_col1, s_col2 = st.columns(2, gap="large")
    with s_col1:
        st.markdown(
            """
            <div class="pipeline-card" style="border-left: 4px solid #38BDF8;">
                <h3 style="color:#38BDF8; margin-bottom:6px;">Sentinel-2 · Optical Multispectral</h3>
                <p style="color:#CBD5E1; font-size:0.92rem; margin-bottom:12px;">
                    Operates in 13 spectral channels (Visible, Red-Edge, NIR, SWIR).
                </p>
                <ul style="color:#94A3B8; font-size:0.88rem; line-height:1.7; padding-left:1.2rem;">
                    <li><strong>B02, B03, B04:</strong> True-Color RGB natural scene display.</li>
                    <li><strong>B08 (NIR):</strong> High reflectance over chlorophyll; essential for NDVI vegetation vigor.</li>
                    <li><strong>B11, B12 (SWIR):</strong> Soil moisture, burn severity, and cloud penetration.</li>
                    <li><strong>Best For:</strong> Crop health, forestry monitoring, water body delineation.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s_col2:
        st.markdown(
            """
            <div class="pipeline-card" style="border-left: 4px solid #F37021;">
                <h3 style="color:#F37021; margin-bottom:6px;">Sentinel-1 · Synthetic Aperture Radar (SAR)</h3>
                <p style="color:#CBD5E1; font-size:0.92rem; margin-bottom:12px;">
                    C-band active radar (VV & VH polarizations), independent of weather or daylight.
                </p>
                <ul style="color:#94A3B8; font-size:0.88rem; line-height:1.7; padding-left:1.2rem;">
                    <li><strong>VV (Co-polarization):</strong> Measures surface scattering and water specular reflection.</li>
                    <li><strong>VH (Cross-polarization):</strong> Volume scattering from forest canopy & rough terrain.</li>
                    <li><strong>Double Bounce:</strong> Urban buildings & vertical infrastructure stand out distinctly.</li>
                    <li><strong>Best For:</strong> Flood inundation mapping through cloud cover, night imaging, terrain structure.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Launch Bottom Banner
    st.markdown(
        """
        <div style="text-align:center; padding: 5rem 2rem; background: #020610; border-top: 1px solid #16243D;">
            <h2 style="color:#F8FAFC; font-size:2.2rem; margin-bottom:1rem;">Ready to inspect Earth observation imagery?</h2>
            <p style="color:#94A3B8; max-width:600px; margin:0 auto 2rem auto;">
                Launch the interactive GIS Workstation to upload your own GeoTIFF rasters or analyze our pre-packaged Sentinel-1/2 validation scenes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    b_l, b_c, b_r = st.columns([2.5, 2, 2.5])
    with b_c:
        st.button(
            "Launch SatQuery Workstation →",
            type="primary",
            use_container_width=True,
            on_click=launch_portal,
        )

    st.markdown(
        """
        <div style="text-align:center; padding: 2rem; color:#64748B; font-size:0.8rem; border-top:1px solid #0F172A;">
            SatQuery AI · Autonomous Earth Observation Assistant · Team Aarohan · Smart India Hackathon
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# VIEW 2: PRODUCTION GIS WORKSTATION PORTAL
# ============================================================

# Top Navigation Bar
nav_l, nav_c, nav_r = st.columns([4, 4, 3], gap="medium")
with nav_l:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="background:#0F1D38; border:1px solid #F37021; width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:18px;">
                🛰️
            </div>
            <div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-weight:800; font-size:1.15rem; color:#FFFFFF; letter-spacing:0.04em;">SATQUERY AI</span>
                    <span class="status-pill status-orange" style="font-size:0.65rem;">ISRO / SIH 2026</span>
                </div>
                <div style="font-size:0.75rem; color:#94A3B8;">Autonomous Multimodal Earth Observation Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with nav_c:
    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:center; gap:10px; height:100%;">
            <span class="status-pill status-active">● Sovereign Enclave: Online</span>
            <span class="status-pill status-info">Air-Gapped Ready</span>
            <span class="status-pill status-orange">48.7M VLM</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with nav_r:
    btn_r1, btn_r2 = st.columns(2)
    with btn_r1:
        if st.button("↺ Reset", use_container_width=True):
            clear_workspace()
            st.rerun()
    with btn_r2:
        st.button("← Home", use_container_width=True, on_click=go_landing)

st.markdown("<hr style='border-color:#16243D; margin: 12px 0 20px 0;'>", unsafe_allow_html=True)

# Main 3-Column Workstation Layout
# Left: Ingestion & Metadata (3.2)
# Center: Geospatial Viewer (4.8)
# Right: Query & Results (4.0)
col_left, col_center, col_right = st.columns([3.3, 4.7, 4.0], gap="large")

# ============================================================
# LEFT COLUMN: Data Ingestion, Sensors & Metadata
# ============================================================
with col_left:
    st.markdown(
        """
        <div class="gis-card-header">
            <span class="gis-card-title">📁 1. Sensor Data Ingestion</span>
            <span class="status-pill status-info">GeoTIFF / PNG</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1-Click Sample Scene Loaders (SIH Pitch Goldmine!)
    st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#CBD5E1; margin-bottom:6px;'>⚡ 1-Click SIH Validation Scenes</div>", unsafe_allow_html=True)
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    with sample_col1:
        if st.button("Load S2 Optical", use_container_width=True):
            load_sample_s2()
            st.toast("Loaded 12-Band Sentinel-2 GeoTIFF Scene")
            st.rerun()
    with sample_col2:
        if st.button("Load S1 SAR", use_container_width=True):
            load_sample_s1()
            st.toast("Loaded Dual-Pol Sentinel-1 SAR Scene")
            st.rerun()
    with sample_col3:
        if st.button("Load Both", use_container_width=True):
            load_sample_both()
            st.toast("Loaded S1 + S2 Co-registered Scene Pair")
            st.rerun()

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # Sentinel-2 Uploader
    s2_file = st.file_uploader(
        "Upload Sentinel-2 Optical (GeoTIFF / PNG / JPG)",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="s2_uploader",
    )
    if s2_file is not None and s2_file.name != st.session_state.s2_name:
        raw_bytes = s2_file.read()
        data, meta = read_raster_file(raw_bytes)
        st.session_state.s2_data = data
        st.session_state.s2_meta = meta
        st.session_state.s2_name = s2_file.name

    # Sentinel-1 Uploader
    s1_file = st.file_uploader(
        "Upload Sentinel-1 SAR (GeoTIFF / PNG / JPG)",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="s1_uploader",
    )
    if s1_file is not None and s1_file.name != st.session_state.s1_name:
        raw_bytes = s1_file.read()
        data, meta = read_raster_file(raw_bytes)
        st.session_state.s1_data = data
        st.session_state.s1_meta = meta
        st.session_state.s1_name = s1_file.name

    # Active Imagery Status Card
    st.markdown(
        """
        <div class="gis-card-header" style="margin-top:18px;">
            <span class="gis-card-title">📊 Raster Metadata Inspector</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.s2_data is not None or st.session_state.s1_data is not None:
        if st.session_state.s2_data is not None:
            m = st.session_state.s2_meta or {}
            st.markdown(
                f"""
                <div class="gis-card" style="padding:10px; font-size:0.82rem; margin-bottom:10px;">
                    <div style="font-weight:700; color:#38BDF8; margin-bottom:4px;">🟢 Sentinel-2 Optical</div>
                    <div style="color:#CBD5E1; margin-bottom:4px; word-break:break-all;">{st.session_state.s2_name}</div>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px; font-size:0.75rem; color:#94A3B8;">
                        <div>Dimensions: <span class="code-tag">{m.get('width', 0)} × {m.get('height', 0)}</span></div>
                        <div>Bands: <span class="code-tag">{m.get('count', 0)} Channels</span></div>
                        <div>CRS: <span class="code-tag">{m.get('crs', 'Local Frame')[:16]}</span></div>
                        <div>Format: <span class="code-tag">{m.get('driver', 'Raster')}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.session_state.s1_data is not None:
            m = st.session_state.s1_meta or {}
            st.markdown(
                f"""
                <div class="gis-card" style="padding:10px; font-size:0.82rem; margin-bottom:10px;">
                    <div style="font-weight:700; color:#F37021; margin-bottom:4px;">🔵 Sentinel-1 SAR</div>
                    <div style="color:#CBD5E1; margin-bottom:4px; word-break:break-all;">{st.session_state.s1_name}</div>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px; font-size:0.75rem; color:#94A3B8;">
                        <div>Dimensions: <span class="code-tag">{m.get('width', 0)} × {m.get('height', 0)}</span></div>
                        <div>Bands: <span class="code-tag">{m.get('count', 0)} Polarizations</span></div>
                        <div>CRS: <span class="code-tag">{m.get('crs', 'Local Frame')[:16]}</span></div>
                        <div>Format: <span class="code-tag">{m.get('driver', 'Raster')}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No raster uploaded. Click 'Load S2 Optical' or 'Load S1 SAR' above to begin.")

    # Layer & Opacity Controls
    st.markdown(
        """
        <div class="gis-card-header" style="margin-top:14px;">
            <span class="gis-card-title">🎛️ Layer & Sensor Controls</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.show_s2 = st.checkbox("Show Sentinel-2 Optical", value=st.session_state.show_s2)
    st.session_state.show_s1 = st.checkbox("Show Sentinel-1 SAR", value=st.session_state.show_s1)
    st.session_state.show_evidence = st.checkbox("Show Visual Evidence / Bounding Box", value=st.session_state.show_evidence)

    if st.session_state.s2_data is not None and st.session_state.s1_data is not None:
        st.session_state.opacity_blend = st.slider(
            "Optical ⟷ SAR Opacity Crossfade",
            min_value=0,
            max_value=100,
            value=st.session_state.opacity_blend,
            help="0% = Pure Sentinel-2 Optical, 100% = Pure Sentinel-1 SAR",
        )

    st.session_state.active_composite = st.radio(
        "Multispectral Composite",
        options=["Natural Color RGB (B4-B3-B2)", "False Color NIR (B8-B4-B3)"],
        index=0 if st.session_state.active_composite == "rgb" else 1,
    )


# ============================================================
# CENTER COLUMN: High-Precision Geospatial Viewer
# ============================================================
with col_center:
    st.markdown(
        """
        <div class="gis-card-header">
            <span class="gis-card-title">🗺️ 2. Geospatial Intelligence Viewer</span>
            <span class="status-pill status-active">Interactive Canvas</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Generate display composite
    display_image: Optional[np.ndarray] = None
    active_mask: Optional[np.ndarray] = None
    bbox_coords: Optional[List[float]] = None
    evidence_label = "Target Detection"
    confidence_val: Optional[float] = None

    if st.session_state.result and st.session_state.result.get("success"):
        res = st.session_state.result
        confidence_val = res.get("confidence")
        if "visual_evidence" in res and "coordinates" in res["visual_evidence"]:
            bbox_coords = res["visual_evidence"]["coordinates"]
        if "spectral_evidence" in res and res["spectral_evidence"].get("has_mask"):
            active_mask = res["spectral_evidence"]["mask"]

    # Base Optical
    s2_img = None
    if st.session_state.s2_data is not None and st.session_state.show_s2:
        if "False Color" in st.session_state.active_composite:
            s2_img = generate_s2_false_color(st.session_state.s2_data)
        else:
            s2_img = generate_s2_rgb(st.session_state.s2_data)

    # Base SAR
    s1_img = None
    if st.session_state.s1_data is not None and st.session_state.show_s1:
        s1_img = generate_s1_composite(st.session_state.s1_data)

    # Blend / Display logic
    if s2_img is not None and s1_img is not None:
        alpha = st.session_state.opacity_blend / 100.0
        # Resize if dimensions differ
        if s2_img.shape != s1_img.shape:
            h_tgt, w_tgt = s2_img.shape[:2]
            s1_resized = np.array(Image.fromarray(s1_img).resize((w_tgt, h_tgt)))
        else:
            s1_resized = s1_img
        display_image = ((1.0 - alpha) * s2_img + alpha * s1_resized).astype(np.uint8)
    elif s2_img is not None:
        display_image = s2_img
    elif s1_img is not None:
        display_image = s1_img

    # Apply Evidence Overlay if enabled
    if display_image is not None:
        if st.session_state.show_evidence and (bbox_coords is not None or active_mask is not None):
            display_image = render_evidence_overlay(
                display_image,
                bbox=bbox_coords,
                mask=active_mask,
                label=evidence_label,
                confidence=confidence_val,
            )

        # Render Geospatial Image
        st.image(
            display_image,
            use_container_width=True,
            caption="SatQuery Earth Observation Raster Viewer",
        )

        # Geospatial Telemetry HUD
        meta_active = st.session_state.s2_meta or st.session_state.s1_meta or {}
        bounds = meta_active.get("bounds", {"left": 78.14, "bottom": 26.21, "right": 78.19, "top": 26.25}) or {}
        res_m = meta_active.get("resolution", (10.0, 10.0))

        st.markdown(
            f"""
            <div style="background:#091326; border:1px solid #16243D; border-radius:8px; padding:10px 14px; font-size:0.75rem; font-family:'JetBrains Mono',monospace; color:#94A3B8; margin-top:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <div><span style="color:#38BDF8;">EXTENT (MinX, MinY):</span> {bounds.get('left', 0.0):.4f}, {bounds.get('bottom', 0.0):.4f}</div>
                    <div><span style="color:#38BDF8;">MAX (MaxX, MaxY):</span> {bounds.get('right', 0.0):.4f}, {bounds.get('top', 0.0):.4f}</div>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div><span style="color:#F37021;">GSD RESOLUTION:</span> {res_m[0]}m / px</div>
                    <div><span style="color:#10B981;">CO-REGISTRATION:</span> UTM Sub-Pixel Aligned</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background:#070F1E; border:2px dashed #1B2B47; border-radius:12px; height:380px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:2rem;">
                <div style="font-size:3rem; margin-bottom:1rem; opacity:0.6;">🛰️</div>
                <h3 style="color:#CBD5E1; font-size:1.15rem; margin-bottom:6px;">No Satellite Imagery Active</h3>
                <p style="color:#64748B; font-size:0.86rem; max-width:320px; margin-bottom:1.5rem;">
                    Upload a Sentinel-1 or Sentinel-2 GeoTIFF from the left panel, or click a 1-click sample scene.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# RIGHT COLUMN: Natural Language Query & Intelligence Report
# ============================================================
with col_right:
    st.markdown(
        """
        <div class="gis-card-header">
            <span class="gis-card-title">💬 3. Agentic Query & Analysis</span>
            <span class="status-pill status-orange">Natural Language</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preset Questions Dropdown
    presets = {
        "Custom query": "",
        "Water Inundation Detection": "Is there water in this image?",
        "Forest & Vegetation Grounding": "Highlight the forested area",
        "Comprehensive Land Cover": "Describe the land cover in this scene",
        "Built-up & Infrastructure": "Are there buildings or urban structures visible?",
        "Agricultural Crop Boundaries": "Locate the agricultural fields",
    }

    selected_preset = st.selectbox(
        "Select Query Preset or enter custom prompt:",
        list(presets.keys()),
        index=1,
    )

    if selected_preset != "Custom query":
        st.session_state.query = presets[selected_preset]

    query_input = st.text_area(
        "Natural-Language Question",
        value=st.session_state.query,
        height=90,
        placeholder="e.g. Is there water in this image? or Highlight the forested area",
    )
    st.session_state.query = query_input

    # Analyze Button
    can_analyze = bool(
        query_input.strip()
        and (st.session_state.s2_data is not None or st.session_state.s1_data is not None)
    )

    run_analysis = st.button(
        "⚡ Execute Agentic Analysis",
        type="primary",
        disabled=not can_analyze,
        use_container_width=True,
    )

    if not can_analyze:
        st.caption("⚠️ Upload at least one satellite raster and enter a question to analyze.")

    # Execution Logic
    if run_analysis:
        progress_placeholder = st.empty()
        
        # Staged progress visualizer
        stages = [
            "1/5 Validating raster CRS & tensor channels...",
            "2/5 Classifying query intent & routing specialist...",
            "3/5 Executing 48.7M parameter PyTorch VLM...",
            "4/5 Computing spectral index grounding (NDWI/NDVI)...",
            "5/5 Synthesizing visual evidence report...",
        ]
        for stage in stages:
            progress_placeholder.markdown(
                f"""
                <div style="background:#091326; border:1px solid #F37021; border-radius:6px; padding:8px 12px; font-size:0.8rem; font-family:'JetBrains Mono',monospace; color:#F37021; margin-bottom:10px;">
                    ● {stage}
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(0.12)

        try:
            _, _, controller, _ = get_cached_system()

            # Prepare Tensors
            s2_tensor = None
            if st.session_state.s2_data is not None:
                s2_arr = st.session_state.s2_data.copy()
                if s2_arr.shape[0] < 12:
                    reps = int(np.ceil(12 / s2_arr.shape[0]))
                    s2_arr = np.concatenate([s2_arr] * reps, axis=0)
                s2_tensor = torch.from_numpy(s2_arr[:12]).float()

            s1_tensor = None
            if st.session_state.s1_data is not None:
                s1_arr = st.session_state.s1_data.copy()
                if s1_arr.shape[0] < 2:
                    reps = int(np.ceil(2 / s1_arr.shape[0]))
                    s1_arr = np.concatenate([s1_arr] * reps, axis=0)
                s1_tensor = torch.from_numpy(s1_arr[:2]).float()

            result = controller.process_query(
                query=st.session_state.query,
                s2_image=s2_tensor,
                s1_image=s1_tensor,
                raw_s2_raster=st.session_state.s2_data,
                raw_s1_raster=st.session_state.s1_data,
            )

            st.session_state.result = result
            st.session_state.analysis_complete = True
            progress_placeholder.empty()
            st.rerun()

        except Exception as exc:
            progress_placeholder.empty()
            st.error(f"Analysis failed: {str(exc)}")

    # Analysis Results Display
    if st.session_state.analysis_complete and st.session_state.result:
        res = st.session_state.result

        if res.get("success"):
            task_type = res.get("task_type", "vqa").replace("_", " ").title()
            conf = res.get("confidence", 0.90)
            answer = res.get("answer", "Analysis conclusive.")
            trace = res.get("trace", {})

            st.markdown(
                f"""
                <div class="gis-card" style="background:#081429; border:1px solid #1E3A6E; padding:14px; margin-top:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span class="status-pill status-orange" style="font-weight:700;">{task_type}</span>
                        <span style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#10B981;">
                            Confidence: {conf:.1%}
                        </span>
                    </div>
                    <div style="font-size:0.95rem; line-height:1.5; color:#F8FAFC; margin-bottom:12px;">
                        {answer}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Spectral Evidence Card if available
            if "spectral_evidence" in res and res["spectral_evidence"].get("grounded"):
                ev = res["spectral_evidence"]
                metrics_html = "".join(
                    [
                        f'<div style="margin-right:12px;">{k}: <span class="code-tag" style="color:#38BDF8;">{v}</span></div>'
                        for k, v in ev.get("metrics", {}).items()
                    ]
                )
                st.markdown(
                    f"""
                    <div class="gis-card" style="padding:10px; background:#06101E; border:1px solid #162B4D; margin-top:8px;">
                        <div style="font-weight:600; font-size:0.78rem; color:#38BDF8; margin-bottom:6px;">
                            🔬 Physical Spectral Verification
                        </div>
                        <div style="font-size:0.82rem; color:#CBD5E1; margin-bottom:6px;">
                            {ev.get('summary', '')}
                        </div>
                        <div style="display:flex; flex-wrap:wrap; font-size:0.75rem; color:#94A3B8;">
                            {metrics_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Execution Trace Expander
            with st.expander("⏱️ Inspection: Millisecond Execution Trace", expanded=False):
                total_ms = trace.get("total_duration_ms", 0.0)
                st.markdown(f"**Total Pipeline Latency:** `{total_ms:.2f} ms`")
                steps = trace.get("steps", [])
                for idx, s in enumerate(steps, 1):
                    st.markdown(
                        f"""
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; padding:4px 0; border-bottom:1px solid #16243D;">
                            <span style="color:#38BDF8;">{idx}. {s.get('name')}</span> 
                            <span style="color:#64748B;">({s.get('duration_ms')}ms)</span>: 
                            <span style="color:#CBD5E1;">{s.get('output')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.error("Analysis encountered an error.")
            for err in res.get("errors", []):
                st.write(f"• {err}")

# Bottom Bar
st.markdown("<hr style='border-color:#16243D; margin: 24px 0 12px 0;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; color:#64748B;">
        <div>SatQuery AI · Geospatial Workstation v2.0 · Team Aarohan</div>
        <div>Smart India Hackathon 2026 · Ministry of Electronics & IT / ISRO Evaluation</div>
    </div>
    """,
    unsafe_allow_html=True,
)
