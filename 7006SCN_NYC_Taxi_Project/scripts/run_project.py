"""Backwards-compatible CLI alias for the numbered root Python entry point.

Old command: spark-submit scripts/run_project.py --stage task1
Recommended: spark-submit NYC_Taxi_TR04.py --stage task1
Both use the same tested coursework/*.py implementation. Running a CLI stage is
not a substitute for executing the four notebook cells and saving YOUR evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from NYC_Taxi_TR04 import main

if __name__ == "__main__":
    main()
