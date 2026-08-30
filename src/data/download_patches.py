"""
BigEarthNet Streaming Patch Extractor
=====================================

Problem: We need ~300 patches from 63GB (S2) and 54GB (S1) tar.zst archives.
Can't download the full archives — not enough disk space.

Solution: Stream-decompress the archive and extract ONLY matching patches.

How tar.zst works:
  [Compressed archive on Zenodo] → [zstd decompress] → [tar stream] → [scan & extract]

Key insight: A tar file is a SEQUENCE of:
  [512-byte header] [file data padded to 512 bytes] [next header] [next data] ...

We can read this sequentially, check each filename against our target list,
and only write the files we care about.
"""

import os
import io
import tarfile
import hashlib
import struct
import requests
import zstandard as zstd
import pandas as pd
from pathlib import Path
from typing import Set, Optional


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/Users/adityaupadhyaya/Desktop/satquery-ai/data/raw")
PROCESSED_DIR = Path("/Users/adityaupadhyaya/Desktop/satquery-ai/data/processed")

ZENODO_BASE = "https://zenodo.org/records/10891137/files"
S2_URL = f"{ZENODO_BASE}/BigEarthNet-S2.tar.zst?download=1"
S1_URL = f"{ZENODO_BASE}/BigEarthNet-S1.tar.zst?download=1"

# Read our 300 target pairs
PAIRS_CSV = PROCESSED_DIR / "prototype_pairs.csv"


# ============================================================
# STEP 1: Build target sets from our prototype pairs
# ============================================================

def load_target_names(pairs_csv: Path) -> tuple[Set[str], Set[str]]:
    """
    Parse our prototype pairs CSV to get the exact patch names we need.
    
    Returns:
        (s2_targets, s1_targets) — sets of patch directory names
    """
    df = pd.read_csv(pairs_csv)
    
    # S2 targets: the patch_id column contains the full S2 identifier
    # Archive path: BigEarthNet-S2/<patch_id>/<band_files>.tif
    s2_targets = set(df["patch_id"].astype(str))
    
    # S1 targets: the s1_name column contains the full S1 identifier
    # Archive path: BigEarthNet-S1/<s1_name>/<band_files>.tif
    s1_targets = set(df["s1_name"].astype(str))
    
    print(f"Loaded {len(s2_targets)} S2 targets and {len(s1_targets)} S1 targets")
    return s2_targets, s1_targets


# ============================================================
# STEP 2: Streaming tar.zst extraction
# ============================================================

class StreamingTarZstExtractor:
    """
    Streams a tar.zst archive from a URL, decompresses on-the-fly,
    and extracts only files matching a target set.
    
    Concept:
    ┌─────────────────────────────────────────────────────────┐
    │  HTTP Range Request (10MB chunks)                       │
    │       │                                                 │
    │       ▼                                                 │
    │  zstd.ZstdDecompressor.stream_reader()                  │
    │       │                                                 │
    │       ▼                                                 │
    │  tarfile.open(fileobj=reader, mode='r|')                │
    │       │                                                 │
    │       ▼                                                 │
    │  for member in tar:  ← reads headers sequentially       │
    │      if member.name matches targets:                    │
    │          extract to disk                                │
    │      else:                                              │
    │          skip (tarfile auto-seeks past data)            │
    └─────────────────────────────────────────────────────────┘
    
    The 'r|' mode means "read tar sequentially" — no seeking required.
    """
    
    def __init__(self, url: str, target_names: Set[str], output_dir: Path,
                 archive_prefix: str, chunk_size: int = 10 * 1024 * 1024):
        """
        Args:
            url: Zenodo download URL for the tar.zst archive
            target_names: Set of patch directory names to extract
            output_dir: Where to save extracted patches
            archive_prefix: Top-level dir in archive (e.g., "BigEarthNet-S2")
            chunk_size: How many bytes to fetch per HTTP request (default 10MB)
        """
        self.url = url
        self.target_names = target_names
        self.output_dir = output_dir
        self.archive_prefix = archive_prefix
        self.chunk_size = chunk_size
        
        # Tracking
        self.extracted_count = 0
        self.scanned_count = 0
        self.bytes_downloaded = 0
        
    def extract(self) -> int:
        """
        Main extraction loop. Returns number of patches extracted.
        
        This works by:
        1. Opening an HTTP session with the Zenodo server
        2. Creating a zstd decompressor
        3. Opening the decompressed stream as a tar archive
        4. Iterating through tar entries and extracting matches
        """
        print(f"\n{'='*60}")
        print(f"Starting extraction: {len(self.target_names)} targets")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open HTTP session — Zenodo supports streaming
        session = requests.Session()
        
        # First, get total size for progress reporting
        head = session.head(self.url, allow_redirects=True)
        total_size = int(head.headers.get("Content-Length", 0))
        print(f"Archive size: {total_size / 1e9:.1f} GB")
        
        # Create zstd decompressor
        dctx = zstd.ZstdDecompressor()
        
        # Stream the archive
        # We use a custom file-like wrapper that reads from HTTP in chunks
        # and feeds into the zstd decompressor
        response = session.get(self.url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        # Wrap the response body in a zstd stream reader
        reader = dctx.stream_reader(response.raw)
        
        # Open as sequential tar archive
        # mode='r|' = sequential read (no seeking)
        with tarfile.open(fileobj=reader, mode='r|') as tar:
            for member in tar:
                self.scanned_count += 1
                
                # Check if this file belongs to one of our target patches
                if self._should_extract(member.name):
                    self._extract_member(tar, member)
                
                # Progress update every 1000 scanned entries
                if self.scanned_count % 1000 == 0:
                    print(f"  Scanned: {self.scanned_count:,} | "
                          f"Extracted: {self.extracted_count}/{len(self.target_names)}")
                
                # Early stop if we've extracted everything
                if self.extracted_count >= len(self.target_names):
                    print(f"\nAll {self.extracted_count} patches extracted!")
                    break
        
        print(f"\n{'='*60}")
        print(f"Extraction complete!")
        print(f"  Scanned: {self.scanned_count:,} entries")
        print(f"  Extracted: {self.extracted_count}/{len(self.target_names)} patches")
        print(f"{'='*60}")
        
        return self.extracted_count
    
    def _should_extract(self, tar_name: str) -> bool:
        """
        Check if a tar entry belongs to one of our target patches.
        
        Archive structure:
          BigEarthNet-S2/<tile>/<patch_id>/<band_files>.tif
          e.g. BigEarthNet-S2/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP/
               S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57/
               S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B01.tif
        
        Our target set contains patch_ids like:
          S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
        
        So the patch_id is at path index 2 (third component).
        """
        parts = tar_name.split("/")
        if len(parts) < 4:  # Need at least: prefix/tile/patch/file
            return False
        
        # parts[0] = "BigEarthNet-S2"
        # parts[1] = "<tile>"
        # parts[2] = "<patch_id>"
        # parts[3] = "<band_file>.tif"
        patch_id = parts[2]
        return patch_id in self.target_names
    
    def _extract_member(self, tar: tarfile.TarFile, member: tarfile.TarInfo):
        """Extract a single tar member to disk."""
        if member.isdir():
            # Create directory
            dir_path = self.output_dir / member.name
            dir_path.mkdir(parents=True, exist_ok=True)
            return
        
        # Only extract .tif files (band data)
        if not member.name.endswith('.tif'):
            return
        
        # Extract the file
        # tar.extractfile() returns a file-like object with the content
        f = tar.extractfile(member)
        if f is None:
            return
        
        out_path = self.output_dir / member.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'wb') as out:
            out.write(f.read())
        
        self.extracted_count += 1


# ============================================================
# STEP 3: Convenience functions for S1 and S2
# ============================================================

def download_s2_patches(pairs_csv: Path = PAIRS_CSV,
                        output_dir: Path = BASE_DIR / "sentinel2",
                        max_retries: int = 3) -> int:
    """Download Sentinel-2 patches for our 300 pairs."""
    _, s2_targets = load_target_names(pairs_csv)
    
    extractor = StreamingTarZstExtractor(
        url=S2_URL,
        target_names=s2_targets,
        output_dir=output_dir,
        archive_prefix="BigEarthNet-S2",
    )
    return extractor.extract()


def download_s1_patches(pairs_csv: Path = PAIRS_CSV,
                        output_dir: Path = BASE_DIR / "sentinel1",
                        max_retries: int = 3) -> int:
    """Download Sentinel-1 patches for our 300 pairs."""
    s1_targets, _ = load_target_names(pairs_csv)
    
    extractor = StreamingTarZstExtractor(
        url=S1_URL,
        target_names=s1_targets,
        output_dir=output_dir,
        archive_prefix="BigEarthNet-S1",
    )
    return extractor.extract()


# ============================================================
# STEP 4: Verify downloads
# ============================================================

def verify_downloads(pairs_csv: Path = PAIRS_CSV,
                     s2_dir: Path = BASE_DIR / "sentinel2",
                     s1_dir: Path = BASE_DIR / "sentinel1"):
    """Check which patches we successfully downloaded."""
    s2_targets, s1_targets = load_target_names(pairs_csv)
    
    # Check S2
    print("\n--- Sentinel-2 Verification ---")
    s2_found = 0
    s2_missing = []
    for patch_id in sorted(s2_targets):
        # Search for the patch directory anywhere under s2_dir
        matches = list(s2_dir.rglob(f"{patch_id}"))
        if matches:
            patch_dir = matches[0]
            tifs = list(patch_dir.glob("*.tif"))
            if len(tifs) >= 10:  # Should have ~13 bands
                s2_found += 1
            else:
                s2_missing.append(f"{patch_id} ({len(tifs)} bands)")
        else:
            s2_missing.append(f"{patch_id} (not found)")
    
    print(f"  Found: {s2_found}/{len(s2_targets)}")
    if s2_missing:
        print(f"  Missing: {len(s2_missing)}")
        for m in s2_missing[:5]:
            print(f"    - {m}")
    
    # Check S1
    print("\n--- Sentinel-1 Verification ---")
    s1_found = 0
    s1_missing = []
    for s1_name in sorted(s1_targets):
        matches = list(s1_dir.rglob(f"{s1_name}"))
        if matches:
            patch_dir = matches[0]
            tifs = list(patch_dir.glob("*.tif"))
            if len(tifs) >= 1:  # Should have 2 bands (VV, VH)
                s1_found += 1
            else:
                s1_missing.append(f"{s1_name} ({len(tifs)} bands)")
        else:
            s1_missing.append(f"{s1_name} (not found)")
    
    print(f"  Found: {s1_found}/{len(s1_targets)}")
    if s1_missing:
        print(f"  Missing: {len(s1_missing)}")
        for m in s1_missing[:5]:
            print(f"    - {m}")
    
    return s2_found, s1_found


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download BigEarthNet patches for SatQuery prototype")
    parser.add_argument("--s2-only", action="store_true", help="Download only Sentinel-2 patches")
    parser.add_argument("--s1-only", action="store_true", help="Download only Sentinel-1 patches")
    parser.add_argument("--verify", action="store_true", help="Verify existing downloads")
    args = parser.parse_args()
    
    if args.verify:
        verify_downloads()
    elif args.s2_only:
        download_s2_patches()
    elif args.s1_only:
        download_s1_patches()
    else:
        print("Starting S2 download (this will take a while — ~63GB archive)...")
        download_s2_patches()
        print("\nStarting S1 download (~54GB archive)...")
        download_s1_patches()
        verify_downloads()
