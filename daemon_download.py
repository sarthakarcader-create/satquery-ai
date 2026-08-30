#!/usr/bin/env python3
"""Optimized download daemon for BigEarthNet S2 patches."""
import os, sys, time

def main():
    pid = os.fork()
    if pid > 0:
        print(f"Download daemon started with PID {pid}")
        with open("data/raw/download_s2.pid", "w") as f:
            f.write(str(pid))
        sys.exit(0)
    
    os.setsid()
    log = open("data/raw/download_s2.log", "w")
    os.dup2(log.fileno(), sys.stdout.fileno())
    os.dup2(log.fileno(), sys.stderr.fileno())
    
    sys.path.insert(0, ".")
    from src.data.download_patches import (
        load_target_names, StreamingTarZstExtractor, S2_URL, PAIRS_CSV, BASE_DIR
    )
    from pathlib import Path
    import requests
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting optimized S2 download daemon...")
    
    pairs_csv = Path("data/processed/prototype_pairs.csv")
    _, s2_targets = load_target_names(pairs_csv)
    
    output_dir = BASE_DIR / "sentinel2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = StreamingTarZstExtractor(
        url=S2_URL,
        target_names=s2_targets,
        output_dir=output_dir,
        archive_prefix="BigEarthNet-S2",
    )
    
    # Use optimized session
    extractor.session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=10,
        max_retries=3,
    )
    extractor.session.mount("https://", adapter)
    
    result = extractor.extract()
    print(f"[{time.strftime('%H:%M:%S')}] Done! Extracted {result} patches")
    
    # Start S1 download next
    print(f"[{time.strftime('%H:%M:%S')}] Starting S1 download...")
    from src.data.download_patches import S1_URL
    
    s1_targets, _ = load_target_names(pairs_csv)
    s1_output = BASE_DIR / "sentinel1"
    s1_output.mkdir(parents=True, exist_ok=True)
    
    s1_extractor = StreamingTarZstExtractor(
        url=S1_URL,
        target_names=s1_targets,
        output_dir=s1_output,
        archive_prefix="BigEarthNet-S1",
    )
    s1_extractor.session = extractor.session
    s1_result = s1_extractor.extract()
    print(f"[{time.strftime('%H:%M:%S')}] S1 Done! Extracted {s1_result} patches")
    
    log.close()

if __name__ == "__main__":
    main()
