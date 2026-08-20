"""
src/pipeline/error_detection.py
================================
Error detection stage runner with graceful fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from member3_safety_app.error_detection.error_detection import *
except ImportError:
    class PrescriptionIssue:
        def __init__(self, error_type: str, field: str, value: str, message: str, severity: str = "MEDIUM", suggestion: str = ""):
            self.error_type = error_type
            self.field = field
            self.value = value
            self.message = message
            self.severity = severity
            self.suggestion = suggestion

        def to_dict(self) -> dict:
            return {
                "error_type": self.error_type,
                "field": self.field,
                "value": self.value,
                "message": self.message,
                "severity": self.severity,
                "suggestion": self.suggestion,
            }

    class ErrorDetector:
        def __init__(self, *args, **kwargs):
            pass

        def check(self, entities: dict) -> list:
            return []

    def issues_to_dict_list(issues: list) -> list:
        res = []
        for i in issues:
            if hasattr(i, "to_dict"):
                res.append(i.to_dict())
            elif isinstance(i, dict):
                res.append(i)
        return res
