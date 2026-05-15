"""Orchestrator stub. Phase B will wire all modules."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.parse_args()
    print("SCAFFOLD ONLY -- Phase B pending. "
          "Run `python scripts/preflight.py` to verify infra.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
