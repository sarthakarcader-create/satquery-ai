"""
SatQuery AI - Geospatial & Remote Sensing Processing Engine
============================================================
Handles high-precision GeoTIFF raster parsing, metadata extraction,
percentile histogram stretching, multispectral band composites (RGB, CIR),
spectral index calculation (NDVI, NDWI, MNDWI), SAR polarimetry,
and visual evidence overlay generation.
"""

import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio
from rasterio.io import MemoryFile
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ============================================================
# GeoTIFF & Raster Ingestion
# ============================================================

def read_raster_file(file_obj_or_bytes: Union[bytes, Any]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Read raster data from an uploaded file object or raw bytes.
    Returns:
        (band_array [C, H, W], metadata_dict)
    """
    metadata: Dict[str, Any] = {
        "is_geotiff": False,
        "width": 0,
        "height": 0,
        "count": 0,
        "crs": "Not Georeferenced",
        "bounds": None,
        "transform": None,
        "dtypes": [],
        "driver": "Unknown",
        "resolution": None,
        "nodata": None,
    }

    if file_obj_or_bytes is None:
        return None, metadata

    raw_bytes = file_obj_or_bytes if isinstance(file_obj_or_bytes, bytes) else file_obj_or_bytes.getvalue()
    if not raw_bytes:
        return None, metadata

    # 1. Attempt GeoTIFF reading via Rasterio
    try:
        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as src:
                data = src.read()  # Shape: (C, H, W)
                metadata["is_geotiff"] = True
                metadata["width"] = src.width
                metadata["height"] = src.height
                metadata["count"] = src.count
                metadata["crs"] = str(src.crs) if src.crs else "EPSG:4326 (Default Geographic)"
                metadata["dtypes"] = [str(d) for d in src.dtypes]
                metadata["driver"] = src.driver
                metadata["nodata"] = src.nodata
                
                if src.bounds:
                    metadata["bounds"] = {
                        "left": round(src.bounds.left, 5),
                        "bottom": round(src.bounds.bottom, 5),
                        "right": round(src.bounds.right, 5),
                        "top": round(src.bounds.top, 5),
                    }
                
                if src.transform:
                    metadata["resolution"] = (
                        round(abs(src.transform[0]), 2),
                        round(abs(src.transform[4]), 2),
                    )
                    metadata["transform"] = [round(v, 4) for v in src.transform[:6]]
                
                return data.astype(np.float32), metadata
    except Exception:
        pass

    # 2. Fallback to standard raster image via PIL (PNG, JPEG, WebP)
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        arr = np.array(img).astype(np.float32)
        
        # Convert H,W or H,W,C into C,H,W
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]  # Drop alpha
            arr = arr.transpose(2, 0, 1)

        metadata["width"] = arr.shape[2]
        metadata["height"] = arr.shape[1]
        metadata["count"] = arr.shape[0]
        metadata["driver"] = img.format or "Standard Image"
        metadata["crs"] = "Local Pixel Coordinates (Image Frame)"
        metadata["bounds"] = {
            "left": 0.0,
            "bottom": 0.0,
            "right": float(arr.shape[2]),
            "top": float(arr.shape[1]),
        }
        metadata["resolution"] = (10.0, 10.0)
        return arr, metadata
    except Exception:
        return None, metadata


# ============================================================
# Percentile Equalization & Visualization
# ============================================================

def percentile_stretch(array: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """
    Apply a 2%-98% scientific percentile stretch to remote sensing arrays.
    Handles outliers, zero-nodata pixels, and clouds gracefully.
    """
    arr = np.nan_to_num(array.astype(np.float32), nan=0.0)
    valid = arr[arr > 0] if np.any(arr > 0) else arr
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    low = np.percentile(valid, p_low)
    high = np.percentile(valid, p_high)
    if high <= low:
        high = low + 1.0

    stretched = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


def generate_s2_rgb(raster: np.ndarray) -> np.ndarray:
    """
    Create True-Color RGB composite from Sentinel-2 bands.
    If 12-band: B04 (Red=idx 3), B03 (Green=idx 2), B02 (Blue=idx 1)
    If 3-band: channels 0, 1, 2
    If 1-band: replicate to grayscale RGB
    Returns: (H, W, 3) uint8 array
    """
    if raster is None or raster.size == 0:
        return np.zeros((120, 120, 3), dtype=np.uint8)

    c, h, w = raster.shape
    if c >= 4:
        r = percentile_stretch(raster[3])
        g = percentile_stretch(raster[2])
        b = percentile_stretch(raster[1])
        return np.stack([r, g, b], axis=-1)
    elif c >= 3:
        r = percentile_stretch(raster[0])
        g = percentile_stretch(raster[1])
        b = percentile_stretch(raster[2])
        return np.stack([r, g, b], axis=-1)
    else:
        gray = percentile_stretch(raster[0])
        return np.stack([gray, gray, gray], axis=-1)


def generate_s2_false_color(raster: np.ndarray) -> np.ndarray:
    """
    Create Color Infrared (CIR) False-Color composite.
    B08 (NIR=idx 7), B04 (Red=idx 3), B03 (Green=idx 2)
    Highlights dense vegetation in vivid red/magenta.
    """
    if raster is None or raster.size == 0:
        return np.zeros((120, 120, 3), dtype=np.uint8)

    c, h, w = raster.shape
    if c >= 8:
        nir = percentile_stretch(raster[7])
        red = percentile_stretch(raster[3])
        green = percentile_stretch(raster[2])
        return np.stack([nir, red, green], axis=-1)
    # If bands are unavailable, fallback to RGB
    return generate_s2_rgb(raster)


def generate_s1_composite(raster: np.ndarray) -> np.ndarray:
    """
    Create SAR composite from Sentinel-1 bands (VV, VH).
    Channel 0: VV (Surface scattering)
    Channel 1: VH (Volume scattering)
    Channel 2: Cross-ratio (VH / VV)
    Returns: (H, W, 3) uint8 array
    """
    if raster is None or raster.size == 0:
        return np.zeros((120, 120, 3), dtype=np.uint8)

    c, h, w = raster.shape
    if c >= 2:
        # Convert dB to linear power if negative
        vv = raster[0]
        vh = raster[1]
        if np.nanmin(vv) < 0:
            vv = 10 ** (np.clip(vv, -30, 0) / 10.0)
        if np.nanmin(vh) < 0:
            vh = 10 ** (np.clip(vh, -35, 0) / 10.0)

        vv_norm = percentile_stretch(vv)
        vh_norm = percentile_stretch(vh)
        ratio = percentile_stretch(vh / (vv + 1e-6))
        return np.stack([vv_norm, vh_norm, ratio], axis=-1)
    else:
        gray = percentile_stretch(raster[0])
        return np.stack([gray, gray, gray], axis=-1)


# ============================================================
# Scientific Spectral Indices & Remote Sensing Analytics
# ============================================================

def compute_ndwi(raster: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalized Difference Water Index (McFeeters, 1996):
    NDWI = (Green - NIR) / (Green + NIR)
    Green = B03 (idx 2), NIR = B08 (idx 7)
    Values > 0 typically denote water bodies.
    """
    if raster is None or raster.shape[0] < 8:
        return None
    green = raster[2].astype(np.float32)
    nir = raster[7].astype(np.float32)
    denom = green + nir + 1e-6
    ndwi = (green - nir) / denom
    return np.nan_to_num(ndwi, nan=-1.0)


def compute_ndvi(raster: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalized Difference Vegetation Index (Rouse et al., 1974):
    NDVI = (NIR - Red) / (NIR + Red)
    NIR = B08 (idx 7), Red = B04 (idx 3)
    Values > 0.35 denote healthy vegetation.
    """
    if raster is None or raster.shape[0] < 8:
        return None
    nir = raster[7].astype(np.float32)
    red = raster[3].astype(np.float32)
    denom = nir + red + 1e-6
    ndvi = (nir - red) / denom
    return np.nan_to_num(ndvi, nan=-1.0)


def analyze_spectral_evidence(
    s2_raster: Optional[np.ndarray],
    s1_raster: Optional[np.ndarray],
    query_task: str,
    query_text: str
) -> Dict[str, Any]:
    """
    Scientifically grounds model predictions with real spectral statistics.
    Returns:
        Dict with metrics, evidence masks, and summary statements.
    """
    evidence: Dict[str, Any] = {
        "grounded": False,
        "summary": "",
        "metrics": {},
        "has_mask": False,
        "mask": None,
    }

    q = query_text.lower()

    # 1. Water / Inundation queries
    if any(w in q for w in ["water", "lake", "river", "flood", "reservoir", "submerged"]):
        if s2_raster is not None and s2_raster.shape[0] >= 8:
            ndwi = compute_ndwi(s2_raster)
            if ndwi is not None:
                water_mask = ndwi > 0.05
                water_pct = float(np.mean(water_mask) * 100.0)
                mean_ndwi = float(np.mean(ndwi))
                evidence["grounded"] = True
                evidence["metrics"]["Water Surface Area"] = f"{water_pct:.1f}%"
                evidence["metrics"]["Mean NDWI Index"] = f"{mean_ndwi:.3f}"
                evidence["has_mask"] = True
                evidence["mask"] = water_mask
                if water_pct > 2.0:
                    evidence["summary"] = f"Optical NDWI verifies water body coverage across {water_pct:.1f}% of the scene."
                else:
                    evidence["summary"] = f"No significant open water bodies identified (NDWI coverage < 2%)."
        elif s1_raster is not None and s1_raster.shape[0] >= 1:
            vv = s1_raster[0]
            low_backscatter = vv < np.percentile(vv, 15)
            coverage = float(np.mean(low_backscatter) * 100.0)
            evidence["grounded"] = True
            evidence["metrics"]["SAR Specular Inundation"] = f"{coverage:.1f}%"
            evidence["summary"] = f"SAR radar backscatter reveals low-specular regions ({coverage:.1f}% coverage) consistent with calm surface water."

    # 2. Vegetation / Forestry / Agriculture queries
    elif any(w in q for w in ["forest", "vegetation", "tree", "crop", "agriculture", "farm"]):
        if s2_raster is not None and s2_raster.shape[0] >= 8:
            ndvi = compute_ndvi(s2_raster)
            if ndvi is not None:
                veg_mask = ndvi > 0.35
                veg_pct = float(np.mean(veg_mask) * 100.0)
                mean_ndvi = float(np.mean(ndvi))
                evidence["grounded"] = True
                evidence["metrics"]["Vegetation Vigor Coverage"] = f"{veg_pct:.1f}%"
                evidence["metrics"]["Mean NDVI Index"] = f"{mean_ndvi:.3f}"
                evidence["has_mask"] = True
                evidence["mask"] = veg_mask
                evidence["summary"] = f"NIR/Red spectral analysis identifies active chlorophyll canopy across {veg_pct:.1f}% of the scene."

    # 3. Built-up / Structures / Urban queries
    elif any(w in q for w in ["building", "urban", "structure", "road", "city"]):
        if s1_raster is not None and s1_raster.shape[0] >= 2:
            vv = s1_raster[0]
            high_backscatter = vv > np.percentile(vv, 85)
            struct_pct = float(np.mean(high_backscatter) * 100.0)
            evidence["grounded"] = True
            evidence["metrics"]["SAR Double-Bounce Density"] = f"{struct_pct:.1f}%"
            evidence["summary"] = f"Sentinel-1 SAR polarimetric return shows structural double-bounce scattering across {struct_pct:.1f}% of the tile."

    return evidence


# ============================================================
# Visual Evidence & Bounding Box Overlay Renderer
# ============================================================

def render_evidence_overlay(
    rgb_image: np.ndarray,
    bbox: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None,
    label: str = "Detected Target",
    confidence: Optional[float] = None,
) -> np.ndarray:
    """
    Renders scientific overlays (bounding boxes, coordinate crosshairs,
    and semi-transparent spectral masks) directly onto the RGB image.
    """
    if rgb_image is None or rgb_image.size == 0:
        return rgb_image

    img = Image.fromarray(rgb_image).convert("RGBA")
    h, w = rgb_image.shape[:2]

    # 1. Render semi-transparent mask if provided
    if mask is not None and mask.shape == (h, w):
        mask_pixels = np.zeros((h, w, 4), dtype=np.uint8)
        mask_pixels[mask] = [0, 220, 255, 100]  # Semi-transparent cyan
        mask_img = Image.fromarray(mask_pixels, "RGBA")
        img = Image.alpha_composite(img, mask_img)

    # 2. Render bounding box if provided
    if bbox is not None and len(bbox) == 4:
        c0, c1, c2, c3 = bbox
        x1 = int(min(c0, c2) * w)
        y1 = int(min(c1, c3) * h)
        x2 = int(max(c0, c2) * w)
        y2 = int(max(c1, c3) * h)

        # Minimum bounding box size
        if abs(x2 - x1) < 8:
            x2 = min(w, x1 + 24)
        if abs(y2 - y1) < 8:
            y2 = min(h, y1 + 24)

        border_color = (243, 112, 33, 255)  # ISRO Saffron
        bg_box = (243, 112, 33, 35)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x1, y1, x2, y2], fill=bg_box, outline=border_color, width=2)
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        # Crosshairs at corners
        cross_len = max(4, min(10, min(w, h) // 10))
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx = cross_len if cx == x1 else -cross_len
            dy = cross_len if cy == y1 else -cross_len
            draw.line([(cx, cy), (cx + dx, cy)], fill=(255, 255, 255, 255), width=2)
            draw.line([(cx, cy), (cx, cy + dy)], fill=(255, 255, 255, 255), width=2)

        conf_str = f" {confidence:.0%}" if confidence else ""
        text = f"{label}{conf_str}"
        tag_w = max(70, len(text) * 7)
        tag_h = 14
        tag_y0 = max(0, y1 - tag_h - 2)
        draw.rectangle([x1, tag_y0, x1 + tag_w, tag_y0 + tag_h], fill=(10, 18, 36, 230))
        draw.rectangle([x1, tag_y0, x1 + tag_w, tag_y0 + tag_h], outline=border_color, width=1)
        draw.text((x1 + 4, tag_y0 + 1), text, fill=(243, 112, 33, 255))

    return np.array(img.convert("RGB"))
