"""Reassemble chunked base64+gzip result files into CSVs.

The repo is maintained through a text-only API, so raw simulation
outputs are stored in results/data/ as gzip+base64 text chunks
(files larger than one chunk are split into .00, .01, ... parts).

Usage:
    python results/decode.py
"""
import base64
import gzip
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

TARGETS = {
    "l2_phase_results.csv.gz.b64": "l2_phase_results.csv",
    "l2_verify_results.csv.gz.b64": "l2_verify_results.csv",
}

for stem, out_name in TARGETS.items():
    parts = sorted(DATA.glob(stem + "*"))
    if not parts:
        raise FileNotFoundError(f"no parts found for {stem} in {DATA}")
    b64 = "".join(p.read_text().strip() for p in parts)
    csv_bytes = gzip.decompress(base64.b64decode(b64))
    out = HERE / out_name
    out.write_bytes(csv_bytes)
    print(f"{out.name}: {len(csv_bytes)} bytes from {len(parts)} part(s)")
