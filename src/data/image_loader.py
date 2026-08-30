"""
Satellite Image Loader for SatQuery AI
========================================

Loads Sentinel-1 (SAR) and Sentinel-2 (optical) GeoTIFF bands,
normalizes them, and stacks them into tensors ready for the model.

Key concepts:
  - Sentinel-2 has 13 spectral bands (each a separate .tif file)
  - Sentinel-1 has 2 polarization bands (VV, VH)
  - Raw values are 12-bit (0-4095) for S2, float32 for S1
  - We normalize to [0, 1] range for model input
  - All bands are resized to the same spatial dimensions

Architecture:
  [GeoTIFF files] → [Rasterio read] → [NumPy arrays] → [Normalize] → [Stack] → [Tensor]
"""

import os
import numpy as np
import rasterio
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ============================================================
# Sentinel-2 Band Configuration
# ============================================================

# Sentinel-2 L2A bands and their wavelengths
S2_BANDS = {
    "B01": {"wavelength": 443, "resolution": 60, "name": "Coastal aerosol"},
    "B02": {"wavelength": 490, "resolution": 10, "name": "Blue"},
    "B03": {"wavelength": 560, "resolution": 10, "name": "Green"},
    "B04": {"wavelength": 665, "resolution": 10, "name": "Red"},
    "B05": {"wavelength": 705, "resolution": 20, "name": "Vegetation Red Edge"},
    "B06": {"wavelength": 740, "resolution": 20, "name": "Vegetation Red Edge"},
    "B07": {"wavelength": 783, "resolution": 20, "name": "Vegetation Red Edge"},
    "B08": {"wavelength": 842, "resolution": 10, "name": "NIR"},
    "B8A": {"wavelength": 865, "resolution": 20, "name": "Narrow NIR"},
    "B09": {"wavelength": 945, "resolution": 60, "name": "Water Vapour"},
    "B11": {"wavelength": 1610, "resolution": 20, "name": "SWIR 1"},
    "B12": {"wavelength": 2190, "resolution": 20, "name": "SWIR 2"},
}

# Bands we actually use (skip BCL - classification map)
S2_BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                 "B08", "B8A", "B09", "B11", "B12"]

# Sentinel-1 bands
S1_BAND_NAMES = ["VV", "VH"]


# ============================================================
# Core Loading Functions
# ============================================================

def load_s2_bands(patch_dir: Path,
                  target_size: Optional[Tuple[int, int]] = None,
                  bands: Optional[List[str]] = None) -> np.ndarray:
    """
    Load Sentinel-2 bands from a patch directory.
    
    Args:
        patch_dir: Path to the patch directory containing band .tif files
        target_size: (H, W) to resize all bands to. If None, use native size.
        bands: List of band names to load. If None, load all S2 bands.
    
    Returns:
        Array of shape (num_bands, H, W), dtype float32, values in [0, 1]
    
    Example:
        >>> data = load_s2_bands(Path("data/raw/sentinel2/BigEarthNet-S2/S2A_..."))
        >>> print(data.shape)  # (12, 120, 120) or similar
    """
    if bands is None:
        bands = S2_BAND_NAMES
    
    arrays = []
    target_h, target_w = target_size if target_size else (120, 120)
    
    for band_name in bands:
        # Find the .tif file for this band
        tif_path = _find_band_file(patch_dir, band_name)
        
        if tif_path is None:
            # If band not found, create empty array
            print(f"  Warning: Band {band_name} not found in {patch_dir}")
            arrays.append(np.zeros((target_h, target_w), dtype=np.float32))
            continue
        
        # Read with Rasterio
        with rasterio.open(tif_path) as src:
            band_data = src.read(1)  # Read first (only) band
            
            # Resize if needed
            if band_data.shape != (target_h, target_w):
                from rasterio.warp import reproject, Resampling
                resized = np.zeros((target_h, target_w), dtype=band_data.dtype)
                reproject(
                    source=band_data,
                    destination=resized,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=rasterio.transform.from_bounds(
                        0, 0, 1, 1, target_w, target_h
                    ),
                    resampling=Resampling.nearest
                )
                band_data = resized
        
        # Normalize to [0, 1]
        band_data = _normalize_s2(band_data)
        arrays.append(band_data)
    
    # Stack into (num_bands, H, W)
    return np.stack(arrays, axis=0).astype(np.float32)


def load_s1_bands(patch_dir: Path,
                  target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load Sentinel-1 SAR bands (VV, VH) from a patch directory.
    
    Args:
        patch_dir: Path to the patch directory containing VV.tif and VH.tif
        target_size: (H, W) to resize to. If None, use native size.
    
    Returns:
        Array of shape (2, H, W), dtype float32
    """
    arrays = []
    target_h, target_w = target_size if target_size else (120, 120)
    
    for band_name in S1_BAND_NAMES:
        tif_path = _find_band_file(patch_dir, band_name)
        
        if tif_path is None:
            print(f"  Warning: Band {band_name} not found in {patch_dir}")
            arrays.append(np.zeros((target_h, target_w), dtype=np.float32))
            continue
        
        with rasterio.open(tif_path) as src:
            band_data = src.read(1)
            
            if band_data.shape != (target_h, target_w):
                from rasterio.warp import reproject, Resampling
                resized = np.zeros((target_h, target_w), dtype=band_data.dtype)
                reproject(
                    source=band_data,
                    destination=resized,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=rasterio.transform.from_bounds(
                        0, 0, 1, 1, target_w, target_h
                    ),
                    resampling=Resampling.nearest
                )
                band_data = resized
        
        # SAR normalization (dB to linear, then normalize)
        band_data = _normalize_s1(band_data)
        arrays.append(band_data)
    
    return np.stack(arrays, axis=0).astype(np.float32)


# ============================================================
# Normalization
# ============================================================

def _normalize_s2(band: np.ndarray, percentile_clip: float = 98.0) -> np.ndarray:
    """
    Normalize Sentinel-2 band to [0, 1].
    
    Sentinel-2 L2A values are typically 0-4095 (12-bit).
    We clip at a percentile to handle outliers, then scale.
    
    Why percentile clipping?
      - Satellite images have extreme values (clouds, shadows)
      - Simple min-max would squish everything to a narrow range
      - Percentile clipping keeps 98% of data in a nice range
    """
    band = band.astype(np.float32)
    
    # Clip outliers
    max_val = np.percentile(band[band > 0], percentile_clip) if np.any(band > 0) else 1.0
    if max_val == 0:
        max_val = 1.0
    
    band = np.clip(band, 0, max_val)
    band = band / max_val
    
    return band


def _normalize_s1(band: np.ndarray, percentile_clip: float = 98.0) -> np.ndarray:
    """
    Normalize Sentinel-1 SAR band to [0, 1].
    
    SAR data is in dB scale (typically -25 to 0 dB).
    We convert to linear power, then normalize.
    """
    band = band.astype(np.float32)
    
    # If values are in dB (negative), convert to linear
    if band.min() < 0:
        band = 10 ** (band / 10.0)
    
    # Clip and normalize
    max_val = np.percentile(band[band > 0], percentile_clip) if np.any(band > 0) else 1.0
    if max_val == 0:
        max_val = 1.0
    
    band = np.clip(band, 0, max_val)
    band = band / max_val
    
    return band


# ============================================================
# Visualization
# ============================================================

def s2_to_rgb(bands: np.ndarray) -> np.ndarray:
    """
    Convert Sentinel-2 bands to an RGB image for visualization.
    
    Uses bands B04 (Red), B03 (Green), B02 (Blue) — the natural color composite.
    Input: (12, H, W) array from load_s2_bands
    Output: (H, W, 3) array for display
    """
    # B04=Red(index 3), B03=Green(index 2), B02=Blue(index 1)
    red = bands[3]    # B04
    green = bands[2]   # B03
    blue = bands[1]    # B02
    
    rgb = np.stack([red, green, blue], axis=-1)
    return np.clip(rgb, 0, 1)


def false_color_composite(bands: np.ndarray) -> np.ndarray:
    """
    Create a false-color composite using NIR-Red-Green.
    Useful for vegetation analysis.
    
    Uses B08 (NIR), B04 (Red), B03 (Green).
    """
    nir = bands[7]    # B08
    red = bands[3]    # B04
    green = bands[2]   # B03
    
    rgb = np.stack([nir, red, green], axis=-1)
    return np.clip(rgb, 0, 1)


# ============================================================
# Utility Functions
# ============================================================

def _find_band_file(patch_dir: Path, band_name: str) -> Optional[Path]:
    """
    Find a band .tif file in a patch directory.
    
    Handles different naming conventions:
      - <patch_id>_B01.tif
      - <patch_id>_VV.tif  
      - B01.tif
    """
    patch_dir = Path(patch_dir)
    
    # Try common patterns
    patterns = [
        f"*_{band_name}.tif",
        f"*_{band_name.upper()}.tif",
        f"{band_name}.tif",
        f"{band_name.upper()}.tif",
        f"*{band_name}*.tif",
    ]
    
    for pattern in patterns:
        matches = list(patch_dir.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def get_patch_info(patch_dir: Path) -> Dict:
    """Get metadata about a patch (band count, dimensions, etc.)."""
    tif_files = list(Path(patch_dir).glob("*.tif"))
    
    info = {
        "path": str(patch_dir),
        "num_bands": len(tif_files),
        "band_files": [f.name for f in tif_files],
    }
    
    if tif_files:
        with rasterio.open(tif_files[0]) as src:
            info["height"] = src.height
            info["width"] = src.width
            info["dtype"] = str(src.dtypes[0])
            info["crs"] = str(src.crs)
    
    return info


# ============================================================
# Demo / Test
# ============================================================

if __name__ == "__main__":
    print("Satellite Image Loader - SatQuery AI")
    print("=" * 50)
    
    # Show band configuration
    print("\nSentinel-2 Bands:")
    for band_name, info in S2_BANDS.items():
        print(f"  {band_name}: {info['name']} ({info['wavelength']}nm, {info['resolution']}m)")
    
    print(f"\nUsing {len(S2_BAND_NAMES)} bands: {S2_BAND_NAMES}")
    print(f"S1 bands: {S1_BAND_NAMES}")
    
    # Try loading from downloaded data (if available)
    s2_dir = Path("/Users/adityaupadhyaya/Desktop/satquery-ai/data/raw/sentinel2")
    if s2_dir.exists():
        # Find first patch directory
        patches = list(s2_dir.rglob("*.tif"))
        if patches:
            patch_dir = patches[0].parent
            print(f"\nLoading from: {patch_dir}")
            
            try:
                s2_data = load_s2_bands(patch_dir)
                print(f"S2 shape: {s2_data.shape}")
                print(f"S2 dtype: {s2_data.dtype}")
                print(f"S2 range: [{s2_data.min():.3f}, {s2_data.max():.3f}]")
                
                rgb = s2_to_rgb(s2_data)
                print(f"RGB shape: {rgb.shape}")
            except Exception as e:
                print(f"Error loading: {e}")
        else:
            print("\nNo patches downloaded yet.")
    else:
        print("\nNo data directory found. Run download first.")
