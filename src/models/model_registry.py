"""
src/models/model_registry.py
==============================
Factory functions that return the active OCR and NER model instances from config.

Usage
-----
    from src.models.model_registry import get_ocr_model, get_ner_model
    from src.utils.config_loader import load_config

    cfg = load_config()

    ocr = get_ocr_model(cfg)
    text, conf = ocr.recognize_line("path/to/line.png")

    ner = get_ner_model(cfg)
    entities = ner.extract_entities("Amoxicillin 500mg twice daily for 7 days")

Adding a new model later
------------------------
OCR: 1. Create class in src/models/your_ocr.py, inherit OCRModel
     2. Add to _ocr_registry() below
     3. Change active_ocr_model in config.yaml

NER: 1. Create class in src/models/your_ner.py, inherit NERModel
     2. Add to _ner_registry() below
     3. Change active_ner_model in config.yaml

→ Zero other code changes required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Type

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ocr_base import OCRModel
from src.models.ner_base import NERModel


# ══════════════════════════════════════════════════════════════════════════════
# OCR REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def _ocr_registry() -> Dict[str, Type[OCRModel]]:
    # Lazy imports — avoids loading heavy model deps at import time
    from src.models.trocr_infer  import TrOCRModel
    from src.models.cnn_infer    import CNNCharacterModel
    from src.models.gemini_infer import GeminiOCRModel
    return {
        "trocr":  TrOCRModel,
        "cnn":    CNNCharacterModel,
        "gemini": GeminiOCRModel,
    }



_ocr_instances: Dict[tuple, OCRModel] = {}


def get_ocr_model(cfg: dict) -> OCRModel:
    """Return the active OCR model instance, constructed from config.

    Parameters
    ----------
    cfg : config dict from src.utils.config_loader.load_config()

    Returns
    -------
    model : An OCRModel instance for the active_ocr_model specified in cfg.
    """
    name = str(cfg.get("active_ocr_model", "trocr")).lower().strip()
    use_pre = bool(cfg.get("trocr_use_pretrained", False))
    model_path = str(cfg.get("trocr_model_path", ""))
    model_name = str(cfg.get("trocr_model_name", ""))
    gemini_model = str(cfg.get("gemini_model", "gemini-2.5-flash"))

    cache_key = (name, use_pre, model_path, model_name, gemini_model)


    if cache_key in _ocr_instances:
        return _ocr_instances[cache_key]

    registry = _ocr_registry()

    if name not in registry:
        available = ", ".join(f'"{k}"' for k in registry)
        raise ValueError(
            f'Unknown OCR model: "{name}". '
            f"Available options: {available}. "
            "Update active_ocr_model in config.yaml."
        )

    cls = registry[name]
    model = cls.from_config(cfg) if hasattr(cls, "from_config") else cls()

    _ocr_instances[cache_key] = model
    return model


# ══════════════════════════════════════════════════════════════════════════════
# NER REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def _ner_registry() -> Dict[str, Type[NERModel]]:
    # Lazy imports — avoids loading spaCy/transformers at import time
    from src.models.ner_infer    import SpacyNERModel
    from src.models.hf_ner_infer import HFClinicalNERModel
    return {
        "spacy":       SpacyNERModel,
        "hf_clinical": HFClinicalNERModel,
        # "ensemble_ner": EnsembleNERModel,  ← add here when ready
    }


_ner_instances: Dict[tuple, NERModel] = {}


def get_ner_model(cfg: dict) -> NERModel:
    """Return the active NER model instance, constructed from config.

    Parameters
    ----------
    cfg : config dict from src.utils.config_loader.load_config()

    Returns
    -------
    model : A NERModel instance for the active_ner_model specified in cfg.

    Notes
    -----
    Default is "spacy" (custom-trained model on YOUR prescription data).
    Switch to "hf_clinical" in config.yaml to use Posos/ClinicalNER.
    """
    name = str(cfg.get("active_ner_model", "spacy")).lower().strip()
    hf_model_name = str(cfg.get("hf_ner_model_name", "Posos/ClinicalNER"))
    ner_model_path = str(cfg.get("ner_model_path", "models/ner_model"))

    cache_key = (name, hf_model_name, ner_model_path)

    if cache_key in _ner_instances:
        return _ner_instances[cache_key]

    registry = _ner_registry()

    if name not in registry:
        available = ", ".join(f'"{k}"' for k in registry)
        raise ValueError(
            f'Unknown NER model: "{name}". '
            f"Available options: {available}. "
            "Update active_ner_model in config.yaml."
        )

    cls = registry[name]

    # Construct with relevant config params
    if name == "spacy":
        model = cls(model_dir=ner_model_path)
    elif name == "hf_clinical":
        model = cls(model_name=hf_model_name)
    elif hasattr(cls, "from_config"):
        model = cls.from_config(cfg)
    else:
        model = cls()

    _ner_instances[cache_key] = model
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def clear_cache() -> None:
    """Clear all model instance caches (useful for testing or config hot-reload)."""
    _ocr_instances.clear()
    _ner_instances.clear()


def list_available_ocr() -> list[str]:
    """Return all registered OCR model names."""
    return list(_ocr_registry().keys())


def list_available_ner() -> list[str]:
    """Return all registered NER model names."""
    return list(_ner_registry().keys())


# Keep the old name for any existing callers
list_available = list_available_ocr

