#!/bin/bash
cd /Users/adityaupadhyaya/Desktop/satquery-ai
nohup /Users/adityaupadhyaya/Desktop/SatQueryAI\ Prototype/.venv/bin/python3 src/data/download_patches.py --s2-only >> data/raw/download_s2.log 2>&1 &
echo $! > data/raw/download_s2.pid
echo "Download started with PID $(cat data/raw/download_s2.pid)"
