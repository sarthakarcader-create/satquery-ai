import base64
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch


# ============================================================
# Project Setup
# ============================================================

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.satquery_model import SatQueryModel, SimpleTokenizer
from src.agents.controller import SatQueryController


# ============================================================
# Landing Page State
# (checked before set_page_config so we can control the
# sidebar's initial state depending on whether the hero or
# the tool is being shown)
# ============================================================

if "launched" not in st.session_state:
    st.session_state.launched = False


def launch():
    st.session_state.launched = True


def go_home():
    st.session_state.launched = False


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded" if st.session_state.launched else "collapsed",
)


# ============================================================
# Hero Image
# ============================================================

@st.cache_data
def get_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


HERO_VIDEO = get_base64("assets/hero.mp4")


# ============================================================
# Styling
# ============================================================

# Base dark theme — applies on both the landing hero and the
# tool view.
st.markdown(
    """<style>
    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(70, 110, 160, 0.12), transparent 34%),
            #07111f;
        color: #e8eef7;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: #091522;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    h1, h2, h3, h4 {
        color: #f5f8fc !important;
        letter-spacing: -0.02em;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 44px;
    }

    div[data-testid="stMetric"] {
        background: #0c1b2c;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 15px;
    }

    textarea, input {
        border-radius: 10px !important;
    }

    hr {
        border-color: rgba(255,255,255,0.07);
    }
    </style>""",
    unsafe_allow_html=True,
)

if not st.session_state.launched:
    # Landing-only overrides: full-screen video hero with custom overlay content.
    st.markdown(
        f"""<style>
        @import url('https://fonts.googleapis.com/css2?family=Anton&display=swap');

        [data-testid="stHeader"], footer, section[data-testid="stSidebar"] {{
            display: none !important;
        }}

        .block-container {{
            padding: 0 !important;
            max-width: 100% !important;
        }}

        .stApp {{
            overflow: hidden;
        }}

        /* Full-screen hero */
        .hero {{
            position: relative;
            width: 100vw;
            height: 100vh;
            margin: 0;
            overflow: hidden;
            box-sizing: border-box;
            padding: 4rem 4.5rem;
        }}

        /* Full-screen background video */
        .hero-video {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center;
            z-index: 0;
        }}

        /* Cinematic readability overlay */
        .hero-overlay {{
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg,
                    rgba(0,0,0,0.55) 0%,
                    rgba(0,0,0,0.16) 55%,
                    rgba(0,0,0,0.38) 100%),
                linear-gradient(0deg,
                    rgba(0,0,0,0.52) 0%,
                    rgba(0,0,0,0.04) 55%,
                    rgba(0,0,0,0.22) 100%);
            z-index: 1;
        }}

        .hero-content {{
            position: relative;
            z-index: 2;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .hero h1 {{
            font-family: 'Anton', sans-serif;
            font-size: clamp(3.5rem, 7vw, 7rem);
            font-weight: 400;
            color: white !important;
            margin: 0;
            padding: 0;
            letter-spacing: 2px;
            line-height: 0.95;
            text-shadow: 3px 3px 14px rgba(0,0,0,0.70);
        }}

        /* Bottom-right text: slightly smaller than the main heading and lifted upward */
        .tagline-block {{
            align-self: flex-end;
            text-align: right;
            margin-bottom: 8.5rem;
            margin-right: 1rem;
            max-width: 650px;
        }}

        .tagline-block .tagline {{
            font-family: 'Anton', sans-serif;
            font-size: clamp(2rem, 4.2vw, 4.2rem);
            font-weight: 400;
            color: white;
            line-height: 1.05;
            letter-spacing: 1px;
            text-shadow: 2px 2px 12px rgba(0,0,0,0.85);
        }}

        /* Custom launch button */
        div[data-testid="stButton"] {{
            position: fixed;
            right: 4.5rem;
            bottom: 3rem;
            z-index: 20;
            width: auto !important;
        }}

        div[data-testid="stButton"] button {{
            min-height: 58px;
            padding: 0.7rem 1.9rem;
            border-radius: 999px;
            border: 1.5px solid rgba(255,255,255,0.82);
            background: rgba(255,255,255,0.10);
            color: white;
            font-family: 'Anton', sans-serif;
            font-size: 1.05rem;
            font-weight: 400;
            letter-spacing: 2px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.28);
            transition: all 0.25s ease;
        }}

        div[data-testid="stButton"] button:hover {{
            background: white;
            color: #111;
            border-color: white;
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.38);
        }}

        div[data-testid="stButton"] button:active {{
            transform: translateY(0);
        }}

        @media (max-width: 768px) {{
            .hero {{
                padding: 2rem;
            }}

            .hero h1 {{
                font-size: clamp(3rem, 13vw, 5rem);
            }}

            .tagline-block {{
                margin-bottom: 7rem;
                margin-right: 0;
            }}

            .tagline-block .tagline {{
                font-size: clamp(1.8rem, 8vw, 3rem);
            }}

            div[data-testid="stButton"] {{
                right: 2rem;
                bottom: 2rem;
            }}
        }}
        </style>""",
        unsafe_allow_html=True,
    )


# ============================================================
# Landing / Hero View
# ============================================================

if not st.session_state.launched:
    st.markdown(
        f"""
        <div class="hero">
            <video
                class="hero-video"
                autoplay
                muted
                loop
                playsinline
                preload="auto"
            >
                <source
                    src="data:video/mp4;base64,{HERO_VIDEO}"
                    type="video/mp4"
                >
            </video>

            <div class="hero-overlay"></div>

            <div class="hero-content">
                <h1>SAT QUERY AI</h1>

                <div class="tagline-block">
                    <div class="tagline">
                        Talk to the Earth<br>
                        in Plain Language.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "LAUNCH SATQUERY  →",
        key="launch_btn",
        on_click=launch,
    )
    st.stop()


# ============================================================
# Helpers
# ============================================================

def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    if size < 1024**3:
        return f"{size / 1024**2:.1f} MB"
    return f"{size / 1024**3:.2f} GB"


def read_uploaded_raster(uploaded_file):
    """Read a GeoTIFF/raster into C,H,W format using rasterio."""
    if uploaded_file is None:
        return None

    try:
        import rasterio

        data = uploaded_file.getvalue()
        with rasterio.MemoryFile(data) as memfile:
            with memfile.open() as dataset:
                return dataset.read()
    except Exception:
        return None


def uploaded_file_to_numpy(uploaded_file):
    """Read PNG/JPG-style uploads into a numpy array."""
    if uploaded_file is None:
        return None

    try:
        from PIL import Image

        uploaded_file.seek(0)
        return np.array(Image.open(uploaded_file))
    except Exception:
        return None


def image_to_tensor(image, bands: int):
    """Convert H,W,C or H,W image data into C,H,W tensor."""
    if image is None:
        return None

    img = np.asarray(image)

    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]

    if img.ndim == 2:
        img = np.stack([img] * bands, axis=-1)
    elif img.ndim == 3:
        channels = img.shape[-1]
        if channels < bands:
            repeats = int(np.ceil(bands / channels))
            img = np.concatenate([img] * repeats, axis=-1)
        img = img[:, :, :bands]
    else:
        return None

    img = img.astype(np.float32)
    max_value = np.nanmax(img) if img.size else 1.0

    if max_value > 1.0 and max_value <= 255.0:
        img /= 255.0

    img = np.nan_to_num(img)
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def raster_to_model_tensor(raster, bands: int):
    """Convert rasterio C,H,W output into the model tensor format."""
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

    max_value = np.nanmax(raster) if raster.size else 1.0

    if max_value > 1.0:
        if max_value <= 255.0:
            raster /= 255.0
        else:
            minimum = np.nanmin(raster)
            maximum = np.nanmax(raster)
            if maximum > minimum:
                raster = (raster - minimum) / (maximum - minimum)

    return torch.from_numpy(raster).float()


def create_preview(raster):
    """Create a display-friendly RGB preview from a raster."""
    if raster is None:
        return None

    arr = np.asarray(raster)

    if arr.ndim == 3:
        if arr.shape[0] >= 3:
            rgb = arr[:3].transpose(1, 2, 0)
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
        rgb = np.clip((rgb - minimum) / (maximum - minimum), 0, 1)

    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)

    return (rgb * 255).astype(np.uint8)


# ============================================================
# Model Initialization
# ============================================================

@st.cache_resource(show_spinner="Loading SatQuery AI model...")
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
# Header - NATIVE STREAMLIT ONLY
# ============================================================

header_l, header_r = st.columns([6, 1])
with header_l:
    st.markdown("### 🛰️ SATQUERY AI")
with header_r:
    st.button("← Home", on_click=go_home)

st.title("Ask questions about Earth.")
st.markdown(
    "Analyze Sentinel-1 SAR and Sentinel-2 optical imagery using "
    "natural-language queries and an evidence-grounded satellite "
    "intelligence pipeline."
)

st.divider()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🛰️ SatQuery AI")
    st.caption("Satellite intelligence workspace")
    st.divider()

    st.markdown("### Data Sources")

    s2_file = st.file_uploader(
        "Sentinel-2 · Optical",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="s2_upload",
        help="Upload a Sentinel-2 optical image or GeoTIFF.",
    )

    s1_file = st.file_uploader(
        "Sentinel-1 · SAR",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="s1_upload",
        help="Upload a Sentinel-1 SAR image or GeoTIFF.",
    )

    st.divider()
    st.markdown("### Viewer")

    show_optical = st.checkbox("Show optical imagery", value=True)
    show_sar = st.checkbox("Show SAR imagery", value=True)

    st.divider()

    if st.button("Clear workspace", use_container_width=True):
        st.session_state.query = ""
        st.session_state.result = None
        st.session_state.analysis_complete = False
        st.rerun()


# ============================================================
# Main Input Area
# ============================================================

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Natural Language Query")
    st.caption("Ask the satellite imagery what you want to know.")

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
    )

    if preset != "Custom query":
        st.session_state.query = presets[preset]

    query = st.text_area(
        "Your question",
        value=st.session_state.query,
        height=130,
        placeholder="Example: Is there water in this image?",
    )
    st.session_state.query = query

    analyze = st.button(
        "Analyze imagery →",
        type="primary",
        use_container_width=True,
    )

with right:
    st.subheader("Data Inventory")

    with st.container(border=True):
        if s2_file is not None:
            st.markdown(f"**🟢 Sentinel-2**")
            st.caption(f"{s2_file.name} · {format_bytes(s2_file.size)}")
        else:
            st.markdown("**Sentinel-2**")
            st.caption("No optical imagery uploaded")

    with st.container(border=True):
        if s1_file is not None:
            st.markdown("**🔵 Sentinel-1**")
            st.caption(f"{s1_file.name} · {format_bytes(s1_file.size)}")
        else:
            st.markdown("**Sentinel-1**")
            st.caption("No SAR imagery uploaded")


# ============================================================
# Analysis
# ============================================================

if analyze:
    if not query.strip():
        st.warning("Enter a question before starting the analysis.")

    elif s2_file is None and s1_file is None:
        st.warning("Upload at least one Sentinel-1 or Sentinel-2 image.")

    else:
        with st.spinner("Analyzing satellite imagery..."):
            try:
                _, _, controller = initialize_model()

                s2_tensor = None
                s1_tensor = None

                if s2_file is not None:
                    s2_raster = read_uploaded_raster(s2_file)

                    if s2_raster is not None:
                        s2_tensor = raster_to_model_tensor(s2_raster, bands=12)
                    else:
                        s2_image = uploaded_file_to_numpy(s2_file)
                        s2_tensor = image_to_tensor(s2_image, bands=12)

                if s1_file is not None:
                    s1_raster = read_uploaded_raster(s1_file)

                    if s1_raster is not None:
                        s1_tensor = raster_to_model_tensor(s1_raster, bands=2)
                    else:
                        s1_image = uploaded_file_to_numpy(s1_file)
                        s1_tensor = image_to_tensor(s1_image, bands=2)

                if s2_tensor is None and s1_tensor is None:
                    st.error("The uploaded imagery could not be read.")
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
# Analysis Results
# ============================================================

if st.session_state.analysis_complete:
    result = st.session_state.result
    st.divider()
    st.subheader("Analysis Output")

    if result and result.get("success"):
        answer = result.get("answer", "No answer returned.")
        confidence = result.get("confidence")
        task_type = result.get("task_type", "unknown")

        c1, c2, c3 = st.columns([2.2, 1, 1], gap="medium")

        with c1:
            with st.container(border=True):
                st.markdown("#### Answer")
                st.write(answer)

        with c2:
            st.metric(
                "Task",
                str(task_type).replace("_", " ").title(),
            )

        with c3:
            if confidence is None:
                st.metric("Confidence", "—")
            else:
                try:
                    st.metric("Confidence", f"{float(confidence):.1%}")
                except (ValueError, TypeError):
                    st.metric("Confidence", str(confidence))

        trace = result.get("trace")
        if trace:
            with st.expander("View execution details"):
                if isinstance(trace, str):
                    st.code(trace, language="json")
                else:
                    st.json(trace)

    else:
        st.error("Analysis failed.")
        errors = result.get("errors", ["Unknown processing error."]) if result else ["Unknown processing error."]
        for error in errors:
            st.write(f"• {error}")


# ============================================================
# Geospatial Viewer
# ============================================================

st.divider()
st.subheader("Geospatial Viewer")
st.caption("Preview the uploaded satellite imagery used by the analysis pipeline.")

viewer_left, viewer_right = st.columns(2, gap="medium")

with viewer_left:
    st.markdown("#### Sentinel-2 · Optical")

    if s2_file is not None and show_optical:
        s2_raster = read_uploaded_raster(s2_file)

        if s2_raster is not None:
            preview = create_preview(s2_raster)
            if preview is not None:
                st.image(preview, use_container_width=True)
            else:
                st.info("Sentinel-2 was uploaded, but a preview could not be generated.")
        else:
            s2_image = uploaded_file_to_numpy(s2_file)
            if s2_image is not None:
                st.image(s2_image, use_container_width=True)
            else:
                st.info("Sentinel-2 was uploaded, but a preview could not be generated.")
    elif s2_file is not None and not show_optical:
        st.info("Optical imagery is hidden. Enable it from the Viewer controls.")
    else:
        st.info("Upload Sentinel-2 imagery to preview it here.")

with viewer_right:
    st.markdown("#### Sentinel-1 · SAR")

    if s1_file is not None and show_sar:
        s1_raster = read_uploaded_raster(s1_file)

        if s1_raster is not None:
            preview = create_preview(s1_raster)
            if preview is not None:
                st.image(preview, use_container_width=True)
            else:
                st.info("Sentinel-1 was uploaded, but a preview could not be generated.")
        else:
            s1_image = uploaded_file_to_numpy(s1_file)
            if s1_image is not None:
                st.image(s1_image, use_container_width=True)
            else:
                st.info("Sentinel-1 was uploaded, but a preview could not be generated.")
    elif s1_file is not None and not show_sar:
        st.info("SAR imagery is hidden. Enable it from the Viewer controls.")
    else:
        st.info("Upload Sentinel-1 imagery to preview it here.")


# ============================================================
# Footer
# ============================================================

st.divider()
st.caption("SatQuery AI · Earth Observation Intelligence")
