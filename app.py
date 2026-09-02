"""
SatQuery AI - Streamlit Web Application
========================================

Interactive web interface for SatQuery AI.

Run locally:
    streamlit run app.py
"""
import streamlit as st
import torch
import textwrap
import numpy as np
from pathlib import Path
import sys
import json
import io


# ============================================================
# Project Setup
# ============================================================

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.satquery_model import SatQueryModel, SimpleTokenizer
from src.agents.controller import SatQueryController


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    textwrap.dedent("""
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 80% 0%,
                rgba(70, 110, 160, 0.12),
                transparent 35%
            ),
            #07111f;
        color: #e8eef7;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #091522;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* ---------- Typography ---------- */

    h1, h2, h3 {
        color: #f5f8fc !important;
        letter-spacing: -0.02em;
    }

    p, label, span {
        color: #b9c5d4;
    }

    /* ---------- Cards ---------- */

    .card {
        background: rgba(13, 27, 43, 0.82);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.16);
    }

    .hero {
        padding: 12px 0 25px 0;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1.05;
        font-weight: 800;
        color: #f7fafc;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #91a2b6;
        max-width: 850px;
        line-height: 1.65;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f2f6fa;
        margin-bottom: 12px;
    }

    .muted {
        color: #8293a7;
        font-size: 0.9rem;
    }

    /* ---------- Status ---------- */

    .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(193,255,114,0.08);
        border: 1px solid rgba(193,255,114,0.18);
        color: #c1ff72;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #c1ff72;
        display: inline-block;
    }

    /* ---------- File cards ---------- */

    .file-card {
        background: #0c1b2c;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .file-name {
        color: #edf3f8;
        font-weight: 650;
        word-break: break-word;
    }

    .file-meta {
        color: #8293a7;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 44px;
    }

    /* ---------- Inputs ---------- */

    textarea,
    input {
        border-radius: 10px !important;
    }

    /* ---------- Metrics ---------- */

    div[data-testid="stMetric"] {
        background: #0c1b2c;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 15px;
        border-radius: 12px;
    }

    /* ---------- Divider ---------- */

    hr {
        border-color: rgba(255,255,255,0.07);
    }

    </style>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

def format_bytes(size):
    """Convert bytes to a human-readable size."""
    if size < 1024:
        return f"{size} B"

    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"

    if size < 1024 ** 3:
        return f"{size / (1024 ** 2):.1f} MB"

    return f"{size / (1024 ** 3):.2f} GB"


def uploaded_file_to_numpy(uploaded_file):
    """
    Convert an uploaded image into a numpy array.

    Supports normal image files such as PNG/JPG.
    GeoTIFF handling is attempted separately.
    """
    if uploaded_file is None:
        return None

    try:
        from PIL import Image

        image = Image.open(uploaded_file)
        return np.array(image)

    except Exception:
        return None


def image_to_tensor(image, bands):
    """
    Convert an image array into a tensor with the requested
    number of bands.
    """

    if image is None:
        return None

    img = np.asarray(image)

    # Remove alpha channel if present
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]

    # Grayscale
    if img.ndim == 2:
        img = np.stack([img] * bands, axis=-1)

    # RGB / multi-channel
    elif img.ndim == 3:

        channels = img.shape[-1]

        if channels < bands:
            repeats = int(np.ceil(bands / channels))
            img = np.concatenate([img] * repeats, axis=-1)

        img = img[:, :, :bands]

    else:
        return None

    # Convert to float
    img = img.astype(np.float32)

    # Normalize sensibly
    max_value = np.nanmax(img) if img.size else 1.0

    if max_value > 1.0:
        img = img / 255.0

    img = np.nan_to_num(img)

    # HWC -> CHW
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return tensor


def read_uploaded_raster(uploaded_file):
    """
    Attempt to read a raster file using rasterio.

    Returns:
        numpy array or None
    """

    if uploaded_file is None:
        return None

    try:
        import rasterio

        data = uploaded_file.getvalue()

        with rasterio.MemoryFile(data) as memfile:
            with memfile.open() as dataset:
                raster = dataset.read()

        return raster

    except Exception:
        return None


def raster_to_model_tensor(raster, bands):
    """
    Convert rasterio output (C,H,W) to model tensor.
    """

    if raster is None:
        return None

    raster = np.asarray(raster).astype(np.float32)

    if raster.ndim == 2:
        raster = raster[np.newaxis, :, :]

    channels = raster.shape[0]

    if channels < bands:
        repeats = int(np.ceil(bands / channels))
        raster = np.concatenate([raster] * repeats, axis=0)

    raster = raster[:bands]

    raster = np.nan_to_num(raster)

    # Normalize if values appear to be byte-scaled
    max_value = np.nanmax(raster) if raster.size else 1.0

    if max_value > 1.0:
        if max_value <= 255:
            raster = raster / 255.0
        else:
            # Generic min-max normalization
            minimum = np.nanmin(raster)
            maximum = np.nanmax(raster)

            if maximum > minimum:
                raster = (raster - minimum) / (maximum - minimum)

    return torch.from_numpy(raster).float()


def create_preview(raster):
    """
    Create an RGB preview from a raster.
    """

    if raster is None:
        return None

    arr = np.asarray(raster)

    if arr.ndim == 3:

        # Rasterio format: C,H,W
        if arr.shape[0] >= 3:
            rgb = arr[:3].transpose(1, 2, 0)

        # Image format: H,W,C
        elif arr.shape[-1] >= 3:
            rgb = arr[:, :, :3]

        else:
            rgb = arr[0]

    else:
        rgb = arr

    rgb = np.nan_to_num(rgb).astype(np.float32)

    minimum = np.percentile(rgb, 2)
    maximum = np.percentile(rgb, 98)

    if maximum > minimum:
        rgb = np.clip(
            (rgb - minimum) / (maximum - minimum),
            0,
            1,
        )

    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)

    return (rgb * 255).astype(np.uint8)


# ============================================================
# Model Initialization
# ============================================================

@st.cache_resource(show_spinner="Loading SatQuery AI...")
def initialize_model():

    model = SatQueryModel()
    tokenizer = SimpleTokenizer()

    controller = SatQueryController(
        model=model,
        tokenizer=tokenizer,
    )

    return model, tokenizer, controller


# ============================================================
# Session State
# ============================================================

if "query" not in st.session_state:
    st.session_state.query = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False


# ============================================================
# Header
# ============================================================

st.markdown(
    textwrap.dedent("""
        <div class="hero">

            <div class="status">
                <span class="dot"></span>
                SATQUERY AI
            </div>

            <div class="hero-title">
                Ask questions about Earth.
            </div>

            <div class="hero-subtitle">
                Analyze Sentinel-1 SAR and Sentinel-2 optical imagery
                using natural-language queries and an evidence-grounded
                satellite intelligence pipeline.
            </div>

        </div>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown("## 🛰️ SatQuery AI")

    st.markdown(
        '<div class="muted">Satellite intelligence workspace</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Data Sources")

    s2_file = st.file_uploader(
        "Sentinel-2 · Optical",
        type=[
            "tif",
            "tiff",
            "png",
            "jpg",
            "jpeg",
        ],
        key="s2_upload",
        help="Upload a Sentinel-2 optical image or GeoTIFF.",
    )

    s1_file = st.file_uploader(
        "Sentinel-1 · SAR",
        type=[
            "tif",
            "tiff",
            "png",
            "jpg",
            "jpeg",
        ],
        key="s1_upload",
        help="Upload a Sentinel-1 SAR image or GeoTIFF.",
    )

    st.divider()

    st.markdown("### Viewer")

    show_optical = st.checkbox(
        "Show optical imagery",
        value=True,
    )

    show_sar = st.checkbox(
        "Show SAR imagery",
        value=True,
    )

    opacity = st.slider(
        "Overlay opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
        step=0.1,
    )

    st.divider()

    if st.button(
        "Clear workspace",
        use_container_width=True,
    ):
        st.session_state.query = ""
        st.session_state.result = None
        st.session_state.analysis_complete = False

        st.rerun()


# ============================================================
# Main Layout
# ============================================================

left, right = st.columns(
    [1.15, 0.85],
    gap="large",
)


# ============================================================
# Query Panel
# ============================================================

with left:

    st.markdown(
        '<div class="section-title">Natural Language Query</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="muted">Ask the satellite imagery what you want to know.</div>',
        unsafe_allow_html=True,
    )

    presets = {
        "Custom query": "",
        "Water detection": "Is there water in this image?",
        "Forest detection": "Highlight the forested area",
        "Land cover": "Describe the land cover in this scene",
        "Buildings": "Are there any buildings visible?",
        "Agriculture": "Locate the agricultural fields",
        "Terrain": "What type of terrain is shown?",
    }

    preset = st.selectbox(
        "Query preset",
        list(presets.keys()),
        label_visibility="collapsed",
    )

    if preset != "Custom query" and not st.session_state.query:
        st.session_state.query = presets[preset]

    query = st.text_area(
        "Query",
        value=st.session_state.query,
        height=130,
        placeholder=(
            "Example: Is there water in this image?"
        ),
        label_visibility="collapsed",
    )

    st.session_state.query = query

    st.markdown("")

    analyze = st.button(
        "Analyze imagery  →",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# Input Inventory
# ============================================================

with right:

    st.markdown(
        '<div class="section-title">Data Inventory</div>',
        unsafe_allow_html=True,
    )

    if s2_file is not None:

        st.markdown(
            textwrap.dedent(f"""
            <div class="file-card">
                <div class="file-name">
                    🟢 {s2_file.name}
                </div>
                <div class="file-meta">
                    Sentinel-2 · {format_bytes(s2_file.size)}
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            textwrap.dedent("""
            <div class="file-card">
                <div class="file-name">
                    Sentinel-2
                </div>
                <div class="file-meta">
                    No optical imagery uploaded
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    if s1_file is not None:

        st.markdown(
            textwrap.dedent(f"""
            <div class="file-card">
                <div class="file-name">
                    🔵 {s1_file.name}
                </div>
                <div class="file-meta">
                    Sentinel-1 · {format_bytes(s1_file.size)}
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            textwrap.dedent("""
            <div class="file-card">
                <div class="file-name">
                    Sentinel-1
                </div>
                <div class="file-meta">
                    No SAR imagery uploaded
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )


# ============================================================
# Analysis
# ============================================================

if analyze:

    if not query.strip():

        st.warning(
            "Enter a question before starting the analysis."
        )

    elif s2_file is None and s1_file is None:

        st.warning(
            "Upload at least one Sentinel-1 or Sentinel-2 image."
        )

    else:

        with st.spinner("Analyzing satellite imagery..."):

            try:

                model, tokenizer, controller = initialize_model()

                s2_tensor = None
                s1_tensor = None

                # --------------------------------------------
                # Sentinel-2
                # --------------------------------------------

                if s2_file is not None:

                    s2_raster = read_uploaded_raster(s2_file)

                    if s2_raster is not None:
                        s2_tensor = raster_to_model_tensor(
                            s2_raster,
                            bands=12,
                        )

                    else:

                        s2_file.seek(0)

                        s2_image = uploaded_file_to_numpy(
                            s2_file
                        )

                        s2_tensor = image_to_tensor(
                            s2_image,
                            bands=12,
                        )

                # --------------------------------------------
                # Sentinel-1
                # --------------------------------------------

                if s1_file is not None:

                    s1_raster = read_uploaded_raster(s1_file)

                    if s1_raster is not None:

                        s1_tensor = raster_to_model_tensor(
                            s1_raster,
                            bands=2,
                        )

                    else:

                        s1_file.seek(0)

                        s1_image = uploaded_file_to_numpy(
                            s1_file
                        )

                        s1_tensor = image_to_tensor(
                            s1_image,
                            bands=2,
                        )

                if s2_tensor is None and s1_tensor is None:

                    st.error(
                        "The uploaded imagery could not be read."
                    )

                else:

                    result = controller.process_query(
                        query=query,
                        s2_image=s2_tensor,
                        s1_image=s1_tensor,
                    )

                    st.session_state.result = result
                    st.session_state.analysis_complete = True

            except Exception as exc:

                st.session_state.result = {
                    "success": False,
                    "errors": [str(exc)],
                }

                st.session_state.analysis_complete = True


# ============================================================
# Results
# ============================================================

if st.session_state.analysis_complete:

    result = st.session_state.result

    st.divider()

    st.markdown(
        '<div class="section-title">Analysis Output</div>',
        unsafe_allow_html=True,
    )

    if result and result.get("success"):

        answer = result.get(
            "answer",
            "No answer returned.",
        )

        confidence = result.get(
            "confidence",
            None,
        )

        task_type = result.get(
            "task_type",
            "unknown",
        )

        col1, col2, col3 = st.columns(
            [2.2, 1, 1],
            gap="medium",
        )

        with col1:

            st.markdown("#### Answer")

            st.markdown(
                textwrap.dedent(f"""
                <div class="card">
                    <div style="
                        font-size:1.15rem;
                        line-height:1.65;
                        color:#edf3f8;
                    ">
                        {answer}
                    </div>
                </div>
                """),
                unsafe_allow_html=True,
            )

        with col2:

            st.metric(
                "Task",
                str(task_type).replace("_", " ").title(),
            )

        with col3:

            if confidence is not None:

                try:
                    confidence_value = float(confidence)

                    st.metric(
                        "Confidence",
                        f"{confidence_value:.1%}",
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    st.metric(
                        "Confidence",
                        str(confidence),
                    )

            else:

                st.metric(
                    "Confidence",
                    "—",
                )

        # --------------------------------------------
        # Execution trace
        # --------------------------------------------

        trace = result.get("trace")

        if trace:

            with st.expander(
                "View execution details"
            ):

                if isinstance(trace, str):

                    st.code(
                        trace,
                        language="json",
                    )

                else:

                    st.json(trace)

    else:

        errors = (
            result.get("errors", [])
            if result
            else ["Unknown processing error."]
        )

        st.error(
            "Analysis failed."
        )

        for error in errors:

            st.write(
                f"• {error}"
            )


# ============================================================
# Geospatial Viewer
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Geospatial Viewer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    textwrap.dedent("""
    <div class="muted">
        Uploaded imagery is available to the analysis pipeline.
        Visualization below provides a quick raster preview.
    </div>
    """),
    unsafe_allow_html=True,
)

viewer_left, viewer_right = st.columns(
    2,
    gap="medium",
)


with viewer_left:

    if s2_file is not None and show_optical:

        s2_file.seek(0)

        s2_raster = read_uploaded_raster(
            s2_file
        )

        if s2_raster is not None:

            preview = create_preview(
                s2_raster
            )

            if preview is not None:

                st.image(
                    preview,
                    caption="Sentinel-2 Optical Preview",
                    use_container_width=True,
                )

            else:

                st.info(
                    "Sentinel-2 uploaded, but a preview could not be generated."
                )

        else:

            s2_file.seek(0)

            image = uploaded_file_to_numpy(
                s2_file
            )

            if image is not None:

                st.image(
                    image,
                    caption="Sentinel-2 Optical Preview",
                    use_container_width=True,
                )

            else:

                st.info(
                    "Sentinel-2 uploaded, but a preview could not be generated."
                )

    else:

        st.markdown(
            textwrap.dedent("""
            <div class="card" style="
                min-height:260px;
                display:flex;
                align-items:center;
                justify-content:center;
                text-align:center;
            ">
                <div>
                    <div style="
                        font-size:2rem;
                        margin-bottom:10px;
                    ">
                        🛰️
                    </div>
                    <div style="
                        color:#edf3f8;
                        font-weight:650;
                    ">
                        Optical imagery
                    </div>
                    <div class="muted">
                        Upload Sentinel-2 to preview imagery
                    </div>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )


with viewer_right:

    if s1_file is not None and show_sar:

        s1_file.seek(0)

        s1_raster = read_uploaded_raster(
            s1_file
        )

        if s1_raster is not None:

            preview = create_preview(
                s1_raster
            )

            if preview is not None:

                st.image(
                    preview,
                    caption="Sentinel-1 SAR Preview",
                    use_container_width=True,
                )

            else:

                st.info(
                    "Sentinel-1 uploaded, but a preview could not be generated."
                )

        else:

            s1_file.seek(0)

            image = uploaded_file_to_numpy(
                s1_file
            )

            if image is not None:

                st.image(
                    image,
                    caption="Sentinel-1 SAR Preview",
                    use_container_width=True,
                )

            else:

                st.info(
                    "Sentinel-1 uploaded, but a preview could not be generated."
                )

    else:

        st.markdown(
            textwrap.dedent("""
            <div class="card" style="
                min-height:260px;
                display:flex;
                align-items:center;
                justify-content:center;
                text-align:center;
            ">
                <div>
                    <div style="
                        font-size:2rem;
                        margin-bottom:10px;
                    ">
                        📡
                    </div>
                    <div style="
                        color:#edf3f8;
                        font-weight:650;
                    ">
                        SAR imagery
                    </div>
                    <div class="muted">
                        Upload Sentinel-1 to preview imagery
                    </div>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )


# ============================================================
# Footer
# ============================================================

st.markdown(
    textwrap.dedent("""
    <div style="
        margin-top:35px;
        padding-top:18px;
        border-top:1px solid rgba(255,255,255,0.06);
        color:#647589;
        font-size:0.78rem;
        text-align:center;
    ">
        SatQuery AI · Earth Observation Intelligence
    </div>
    """),
    unsafe_allow_html=True,
)
