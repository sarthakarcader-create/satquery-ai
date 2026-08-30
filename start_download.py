#!/usr/bin/env python3
"""Start the S2 download in background."""
import subprocess
import sys
import os

# Change to project directory
os.chdir('/Users/adityaupadhyaya/Desktop/satquery-ai')

# Run the download script
cmd = [
    sys.executable,
    'src/data/download_patches.py',
    '--s2-only'
]

# Start as background process
log_file = open('data/raw/download_s2.log', 'w')
proc = subprocess.Popen(
    cmd,
    stdout=log_file,
    stderr=subprocess.STDOUT,
    cwd='/Users/adityaupadhyaya/Desktop/satquery-ai'
)

print(f"Download started! PID: {proc.pid}")
print(f"Log: data/raw/download_s2.log")
print(f"Monitor: tail -f data/raw/download_s2.log")
