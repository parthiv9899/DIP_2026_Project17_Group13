"""Runs the whole Group 13 pipeline: feature cache -> nested CV -> Excel, figures and overlays.

Usage: python Codes/run_all.py [--data DATA_DIR]   (defaults to ./data next to Codes/)
Every stage resumes from its cache, so an interrupted run can simply be restarted.
"""

import argparse
import time
from pathlib import Path

import afid
import pipeline
import report

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()
    recs = afid.list_images(args.data)
    print(f"{len(recs)} strips ({sum(r['label'] for r in recs)} defective)")
    cache, work = ROOT / "cache" / "features", ROOT / "cache" / "cv"
    t = time.time()
    pipeline.extract(recs, cache)
    print(f"features done in {time.time() - t:.0f} s", flush=True)
    pipeline.cross_validate(recs, cache, work)
    print(f"cross-validation done in {time.time() - t:.0f} s", flush=True)
    report.build(recs, work, ROOT / "Results")
    print(f"results written in {time.time() - t:.0f} s")
