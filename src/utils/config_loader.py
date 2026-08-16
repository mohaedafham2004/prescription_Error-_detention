"""
src/utils/config_loader.py
===========================
Loads and validates config.yaml from the project root.

Usage
-----
    from src.utils.config_loader import load_config

    cfg = load_config()
    print(cfg["active_ocr_model"])   # "trocr"
    print(cfg["trocr_model_path"])   # "models/trocr_finetuned"

The loader:
  - Searches for config.yaml starting from this file's location up to project root
  - Falls back to safe defaults for every key if the key is missing in the file
  - Returns a plain dict (no external config library needed)
  - Results are cached so the file is only read once per process
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ── Default configuration values ──────────────────────────────────────────────
DEFAULTS: Dict[str, Any] = {
    "active_ocr_model":       "trocr",
    "trocr_model_path":       "models/trocr_finetuned",
    "trocr_use_pretrained":   False,
    "trocr_model_name":       "microsoft/trocr-base-handwritten",
    "trocr_max_new_tokens":   64,
    "confidence_threshold":   0.6,
    "ner_model_path":         "models/ner_model",
    "ner_enabled":            True,
    "error_rules_dir":        "data/error_rules",
    "fuzzy_match_threshold":  80,
    "dashboard_title":        "Smart Prescription Error Detection",
}


def _find_config_file() -> Path | None:
    """Walk up from this file's directory to find config.yaml."""
    search = Path(__file__).resolve()
    for _ in range(6):   # max 6 levels up
        candidate = search / "config.yaml"
        if candidate.exists():
            return candidate
        search = search.parent
    return None


@functools.lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load config.yaml and return a merged dict (file values + defaults).

    Parameters
    ----------
    config_path : Optional explicit path to config.yaml.
                  If None, searches upward from this file.

    Returns
    -------
    cfg : Dict with all keys from DEFAULTS, overridden by file values.
    """
    cfg = dict(DEFAULTS)   # start with defaults

    # Resolve config file path
    if config_path:
        path = Path(config_path)
    else:
        path = _find_config_file()

    if path is None or not path.exists():
        print("  [config] config.yaml not found — using built-in defaults.")
        return cfg

    if not _HAS_YAML:
        print("  [config] PyYAML not installed — using defaults. "
              "Run: pip install pyyaml")
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        file_cfg = yaml.safe_load(f) or {}

    cfg.update({k: v for k, v in file_cfg.items() if v is not None})

    # Resolve relative paths to absolute (relative to config file directory)
    root = path.parent
    for key in ("trocr_model_path", "ner_model_path", "error_rules_dir"):
        val = cfg.get(key, "")
        if val and not Path(val).is_absolute():
            cfg[key] = str(root / val)

    return cfg


def reload_config(config_path: str | None = None) -> Dict[str, Any]:
    """Force a fresh reload (clears the lru_cache)."""
    load_config.cache_clear()
    return load_config(config_path)


def get(key: str, default: Any = None) -> Any:
    """Convenience shorthand: get a single config value."""
    return load_config().get(key, default)
