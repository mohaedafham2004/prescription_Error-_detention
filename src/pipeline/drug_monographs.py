"""
src/pipeline/drug_monographs.py
================================
Drug monograph lookup runner with graceful fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from member3_safety_app.error_detection.drug_monographs import *
except ImportError:
    def get_drug_monograph(medicine_name: str) -> dict | None:
        """Return monograph details for a medicine if available."""
        return None
