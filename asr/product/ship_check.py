"""The shipping run's own marks (ship-eval.sh) at the app's thresholds, beside the published check:
the Rust pipeline should reproduce the offline policy (prodforce kaws0 1,0,0,0.2 at det 0.6 / score 6.0).
usage (from asr/): python tools/ship_check.py <name>
"""
import os
import sys

sys.argv = [sys.argv[0], "--gate"] + sys.argv[1:]
sys.path.insert(0, os.path.dirname(__file__))
import prod  # noqa: E402

name = sys.argv[2]
prod.GATE = False
prod.show("D today", prod.row(prod.load("eval-val", "d"), prod.load("..", "rust"), prod.load("eval-kokoro", "rustd"), 0.4707, 5.13))
prod.GATE = True
prod.show(f"{name} 0.6/6.0", prod.row(prod.load("eval-val", name), prod.load("..", name), prod.load("eval-kokoro", name), 0.6, 6.0))
