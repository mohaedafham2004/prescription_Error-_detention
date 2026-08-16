"""
src/models/trocr_infer.py
==========================
TrOCR-based OCR model implementing the OCRModel interface.

Loads model weights either from a local folder (models/trocr_finetuned/) or
directly from the Hugging Face Hub (e.g. "mohaedafham2004/trocr-prescription-finetuned").

Designed for deployment on Streamlit Community Cloud:
  - Fetches HF_TOKEN from st.secrets or environment variable if repository is private
  - Robust tokenizer loading with fallback for slow/fast tokenizers
  - Memory-efficient lazy loading
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

# ── Project root on path ─────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ocr_base import OCRModel, ImageInput


# ── Hallucination detection ───────────────────────────────────────────────────
import re as _re

# Known boilerplate phrases TrOCR produces when the input has little signal.
# These come from web-crawl training data leaking through when the model is
# uncertain (blank crops, borders, watermarks, heavily distorted handwriting).
_HALLUCINATION_PHRASES: tuple = (
    "jump to navigation",
    "jump to search",
    "retrieved from",
    "this page was last",
    "wikipedia",
    "free encyclopedia",
    "navigation menu",
    # Wikipedia sidebar / tool panel phrases (observed in real runs)
    "personal tools",
    "my contributions",
    "create account",
    "log in",
    "main page",
    "contents",
    "current events",
    "random article",
    "about wikipedia",
    "contact us",
)

# Pattern: 3+ groups of purely-digit tokens separated by spaces, e.g. "000 000 000"
_REPEATED_DIGITS = _re.compile(r"(\b\d+\b\s+){3,}")

# Pattern: trailing isolated digit-only token at end of line (e.g. "TD 000" or "for 1")
# Catches the "repetition_penalty helped but didn't fully remove" leftover artifact.
# Requires the token to be 2+ digits so we don't flag real single-digit doses.
_TRAILING_DIGIT = _re.compile(r"\b0{2,}\s*$")

# Pattern: same word repeated 3+ times  (e.g. "the the the")
_REPEATED_WORD   = _re.compile(r"\b(\w+)\s+\1\s+\1\b")


def _is_hallucinated(text: str) -> str | None:
    """Return a short description of the hallucination pattern, or None if clean.

    Parameters
    ----------
    text : Decoded TrOCR output string.

    Returns
    -------
    Pattern description string (truthy) if hallucinated, else None (falsy).
    """
    if not text:
        return None

    lower = text.lower()

    for phrase in _HALLUCINATION_PHRASES:
        if phrase in lower:
            return f"boilerplate phrase: {phrase!r}"

    if _REPEATED_DIGITS.search(text):
        return "repeated digit tokens"

    if _TRAILING_DIGIT.search(text):
        return "trailing zero-pad token (OCR artifact)"

    if _REPEATED_WORD.search(text):
        return "repeated word"

    return None




def _pad_to_min_height(pil_img, min_h: int = 64):
    """Pad a PIL image to at least min_h pixels tall with white background.

    Why this matters for TrOCR
    --------------------------
    TrOCR's ViT encoder resamples every input to 384×384 pixels internally.
    A crop that is 800×30 pixels gets stretched 12.8× vertically — the
    extreme aspect ratio distortion makes handwriting unrecognisable and
    causes the model to hallucinate boilerplate text.

    Padding to 64px preserves the stroke proportions so the resampled image
    still looks like text.  If lines are still being cut off, increase min_h.
    """
    from PIL import Image as PILImage
    w, h = pil_img.size
    if h >= min_h:
        return pil_img
    # Create white canvas and paste the original image centered vertically
    pad_top = (min_h - h) // 2
    canvas = PILImage.new("RGB", (w, min_h), (255, 255, 255))
    canvas.paste(pil_img, (0, pad_top))
    return canvas


def get_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Retrieve Hugging Face access token if available.

    Checks:
    1. Explicit token argument
    2. Streamlit Cloud secrets (st.secrets["HF_TOKEN"])
    3. Environment variable (HF_TOKEN)
    """
    if explicit_token:
        return explicit_token

    # 1. Streamlit Secrets (for Streamlit Community Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass

    # 2. Environment variable
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")



def _load_trocr_processor(src: str, token: Optional[str] = None):
    """Load TrOCR processor with robust multi-backend tokenizer initialization."""
    from transformers import (
        TrOCRProcessor,
        AutoImageProcessor,
        AutoTokenizer,
        RobertaTokenizer,
        XLMRobertaTokenizer,
    )

    try:
        return TrOCRProcessor.from_pretrained(src, token=token)
    except Exception:
        pass

    img_proc = AutoImageProcessor.from_pretrained(src, token=token)
    for tok_cls in (XLMRobertaTokenizer, RobertaTokenizer, AutoTokenizer):
        try:
            tok = tok_cls.from_pretrained(src, token=token)
            return TrOCRProcessor(image_processor=img_proc, tokenizer=tok)
        except Exception:
            continue

    raise RuntimeError(f"Could not load tokenizer/processor for '{src}'.")


class TrOCRModel(OCRModel):
    """TrOCR line-level OCR model.

    Wraps HuggingFace TrOCRProcessor + VisionEncoderDecoderModel.
    Supports both locally fine-tuned models and models hosted on the HF Hub.

    Parameters
    ----------
    model_dir      : Path to local fine-tuned model folder.
    use_pretrained : If True, load from HF Hub instead of local folder.
    hf_model_name  : HF Hub repo name (e.g., 'mohaedafham2004/trocr-prescription-finetuned').
    max_new_tokens : Maximum decoder output tokens per line.
    hf_token       : Optional Hugging Face token (for private repos).
    """

    _MODEL_NAME = "trocr"

    def __init__(
        self,
        model_dir: str = "models/trocr_finetuned",
        use_pretrained: bool = False,
        hf_model_name: str = "microsoft/trocr-small-handwritten",
        max_new_tokens: int = 64,
        hf_token: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.use_pretrained = use_pretrained
        self.hf_model_name = hf_model_name
        self.max_new_tokens = max_new_tokens
        self.hf_token = get_hf_token(hf_token)

        self._processor = None
        self._model = None
        self._loaded = False
        self._load_error: Optional[str] = None

    # ── OCRModel interface ────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def is_ready(self) -> bool:
        if self._load_error:
            return False
        if self._loaded:
            return True
        if self.use_pretrained:
            return True
        # Check if local model dir exists AND contains weights
        has_local = self.model_dir.exists() and (
            (self.model_dir / "config.json").exists()
            or (self.model_dir / "model.safetensors").exists()
            or (self.model_dir / "pytorch_model.bin").exists()
        )
        return has_local

    def recognize_line(self, image: ImageInput) -> tuple[str, float]:
        """Transcribe a single prescription line image."""
        self._ensure_loaded()
        if self._load_error:
            raise RuntimeError(f"TrOCR model load error: {self._load_error}")
        if self._model is None:
            raise RuntimeError("TrOCR model is not initialized.")

        try:
            from PIL import Image as PILImage
            import torch

            # Accept path or PIL Image
            if isinstance(image, (str, Path)):
                pil_img = PILImage.open(str(image)).convert("RGB")
            elif hasattr(image, "convert"):  # PIL Image
                pil_img = image.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")

            pil_img = _pad_to_min_height(pil_img)

            pixel_values = self._processor(
                images=pil_img, return_tensors="pt"
            ).pixel_values

            with torch.no_grad():
                output = self._model.generate(
                    pixel_values,
                    # ── How long a single prescription line can be.
                    # Increase if a line is getting cut off mid-word.
                    max_new_tokens=self.max_new_tokens,

                    # ── Beam search: considers 4 candidate sequences in
                    # parallel and returns the highest-scoring one.
                    # Higher = better quality but slower; 4 is a good default.
                    num_beams=4,

                    # ── Penalise repeating the same token. Values >1 reduce
                    # repetition; 2.0 strongly discourages "000 000 000" loops.
                    # Increase toward 3.0 if repetition still occurs.
                    repetition_penalty=2.0,

                    # ── Prevent any N-gram of this length from appearing twice.
                    # 3 means "000 00" won't repeat; safe for medical abbreviations.
                    # Tune down to 2 if rare multi-word drug names are cut short.
                    no_repeat_ngram_size=3,

                    # ── Stop beam search as soon as all beams hit EOS,
                    # rather than always generating max_new_tokens tokens.
                    early_stopping=True,

                    output_scores=True,
                    return_dict_in_generate=True,
                )

            text = self._processor.batch_decode(
                output.sequences, skip_special_tokens=True
            )[0].strip()

            # ── Hallucination guard ───────────────────────────────────────────
            # TrOCR occasionally hallucinates web-page boilerplate or repeated
            # tokens when the line crop has very little real signal
            # (blank area, border, smudge). Flag these explicitly so the caller
            # can surface a warning rather than silently using garbage text.
            hallucination_flag = _is_hallucinated(text)
            if hallucination_flag:
                print(
                    f"  [TrOCR] ⚠ Possible hallucination detected: {text!r:.60}"
                    f" (pattern: {hallucination_flag})"
                )

            # ── Confidence proxy ──────────────────────────────────────────────
            # We use the geometric mean of per-token softmax probabilities as a
            # proxy for overall sequence confidence.
            #
            # Interpretation:
            #   ≥ 0.70  → high confidence, reliable transcription
            #   0.40–0.69 → moderate; worth a human check
            #   < 0.40  → low; likely a poor crop or ambiguous handwriting
            #
            # Why does repetition read low? Repeated tokens arise when the model
            # is uncertain; it picks the statistically safest next token (often the
            # previous one). Each repeat step typically has moderate probability
            # PER STEP but the geometric mean collapses once several high-entropy
            # steps accumulate. So a 0.31 confidence IS a real signal — it means
            # the model was not confident step-by-step, not a scoring bug.
            confidence = 0.0
            if hasattr(output, "scores") and output.scores:
                import torch.nn.functional as F
                scores = output.scores  # list of (1, vocab) tensors
                probs = [F.softmax(s[0], dim=-1) for s in scores]
                token_confs = [
                    p[output.sequences[0, i + 1]].item()
                    for i, p in enumerate(probs)
                    if (i + 1) < output.sequences.shape[1]
                ]
                if token_confs:
                    import math
                    log_sum = sum(math.log(max(c, 1e-9)) for c in token_confs)
                    confidence = math.exp(log_sum / len(token_confs))

            # Force confidence to 0 when a hallucination is detected so the
            # pipeline surfaces it as a low-confidence line with a warning.
            if hallucination_flag:
                confidence = 0.0

            if text and not hallucination_flag and confidence == 0.0:
                confidence = 0.5

            return text, round(confidence, 4)

        except Exception as e:
            print(f"  [TrOCR] Inference error: {e}", file=sys.stderr)
            raise e

    # ── Loading ───────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._load_error:
            raise RuntimeError(self._load_error)

        try:
            from transformers import VisionEncoderDecoderModel

            has_local_weights = self.model_dir.exists() and (
                (self.model_dir / "config.json").exists()
                or (self.model_dir / "model.safetensors").exists()
                or (self.model_dir / "pytorch_model.bin").exists()
            )

            # Determine source: local folder or Hugging Face Hub
            if not self.use_pretrained and not has_local_weights:
                print(
                    f"  [TrOCR] ⚠️ Local model at '{self.model_dir}' is not found/empty. "
                    f"Loading '{self.hf_model_name}' from Hugging Face Hub instead…"
                )
                src = self.hf_model_name
            elif self.use_pretrained:
                src = self.hf_model_name
                print(f"  [TrOCR] Loading model from Hugging Face Hub: {src} …")
            else:
                src = str(self.model_dir)
                print(f"  [TrOCR] Loading fine-tuned model from: {src} …")

            self._processor = _load_trocr_processor(src, token=self.hf_token)
            self._model = VisionEncoderDecoderModel.from_pretrained(
                src, token=self.hf_token
            )
            self._model.eval()
            self._loaded = True
            print("  [TrOCR] ✅ Model loaded successfully.")

        except Exception as e:
            self._load_error = str(e)
            print(f"  [TrOCR] ❌ Failed to load model '{src}': {e}", file=sys.stderr)
            raise e

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "TrOCRModel":
        """Construct from a config dict (from config_loader.load_config())."""
        return cls(
            model_dir=cfg.get("trocr_model_path", "models/trocr_finetuned"),
            use_pretrained=cfg.get("trocr_use_pretrained", False),
            hf_model_name=cfg.get(
                "trocr_model_name", "microsoft/trocr-small-handwritten"
            ),
            max_new_tokens=cfg.get("trocr_max_new_tokens", 64),
            hf_token=cfg.get("hf_token"),
        )
