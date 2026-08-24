"""Console entry point, mirroring run_experiment.py for installed usage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_experiment import main  # noqa: E402

__all__ = ["main"]
