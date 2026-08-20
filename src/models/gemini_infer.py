"""
src/models/gemini_infer.py
==========================
Google Gemini Multimodal Vision inference model implementing the OCRModel interface.

Provides:
1. `GeminiOCRModel`: Inherits from OCRModel, provides `recognize_line()` and `recognize_full_prescription()`.
2. Full multimodal extraction: Transcribes handwritten prescriptions, extracts clinical entities
   (MEDICINE, DOSAGE, FREQUENCY, DURATION), and computes confidence scores.
"""

from __future__ import annotations

import io
import os
import sys
import time
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ── Project root on path ─────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ocr_base import OCRModel, ImageInput
from src.models.gemini_verifier import get_gemini_api_key

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


class GeminiOCRModel(OCRModel):
    """Gemini Multimodal OCR and clinical entity extraction model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-flash-latest",
    ):
        self._model_name = model_name
        self.api_key = api_key or get_gemini_api_key()
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY in .env"
            )
        self._init_client()

    def _init_client(self):
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._sdk_type = "google-genai"
        except ImportError:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self.legacy_client = legacy_genai.GenerativeModel(self._model_name)
                self._sdk_type = "google-generativeai"
            except ImportError:
                raise ImportError(
                    "Please install google-genai: pip install google-genai"
                )

    @classmethod
    def from_config(cls, cfg: dict) -> "GeminiOCRModel":
        model_name = cfg.get("gemini_model", "gemini-flash-latest")
        api_key = cfg.get("gemini_api_key") or get_gemini_api_key()
        return cls(api_key=api_key, model_name=model_name)

    @property
    def model_name(self) -> str:
        return f"gemini ({self._model_name})"

    @property
    def is_ready(self) -> bool:
        return bool(self.api_key)

    def _to_pil_image(self, image: ImageInput) -> PILImage.Image:
        """Convert any input image (Path, str, ndarray, or PIL Image) to an RGB PIL Image."""
        if isinstance(image, (str, Path)):
            img = PILImage.open(str(image)).convert("RGB")
        elif PILImage and isinstance(image, PILImage.Image):
            img = image.convert("RGB")
        elif hasattr(image, "shape"):  # NumPy ndarray
            import cv2
            if len(image.shape) == 2:
                img = PILImage.fromarray(image).convert("RGB")
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = PILImage.fromarray(rgb)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Resize large scans for fast upload
        max_dim = 1600
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), PILImage.Resampling.LANCZOS)
        return img

    def recognize_line(self, image: ImageInput) -> Tuple[str, float]:
        """Transcribe a single cropped line image using Gemini."""
        pil_img = self._to_pil_image(image)

        prompt = (
            "Transcribe this single line of medical handwriting accurately. "
            "Return only the exact transcribed text, nothing else."
        )

        try:
            if self._sdk_type == "google-genai":
                response = self.client.models.generate_content(
                    model=self._model_name,
                    contents=[pil_img, prompt],
                    config=dict(temperature=0.0),
                )
                text = response.text.strip() if response.text else ""
            else:
                response = self.legacy_client.generate_content([prompt, pil_img])
                text = response.text.strip() if response.text else ""

            confidence = 0.95 if text else 0.0
            return text, confidence
        except Exception as exc:
            return "", 0.0

    def recognize_full_prescription(
        self,
        image: ImageInput,
    ) -> Dict[str, Any]:
        """
        Multimodal end-to-end extraction from a full prescription image.
        
        Returns a dictionary containing:
          - extracted_text: Full transcribed text
          - lines: List of line items with text & confidence
          - entities: {"MEDICINE": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}
          - mean_confidence: float
        """
        pil_img = self._to_pil_image(image)

        system_instruction = (
            "You are an expert clinical OCR and medical prescription transcription AI. "
            "Your task is to transcribe the handwritten prescription image faithfully and extract clinical entities. "
            "Rules:\n"
            "1. Transcribe each line exactly as written on the prescription.\n"
            "2. Extract medicine names, dosages, frequencies, and durations.\n"
            "3. If a word is unclear, provide your best clinical reading with high accuracy.\n"
            "4. Return strictly valid JSON."
        )

        user_prompt = """Analyze the prescription image and return ONLY a JSON object with this schema:
{
  "lines": ["line 1 text", "line 2 text", ...],
  "full_text": "all lines joined by newline",
  "entities": {
    "MEDICINE": ["Drug Name 1", ...],
    "DOSAGE": ["500mg", "10ml", ...],
    "FREQUENCY": ["TID", "once daily", ...],
    "DURATION": ["7 days", "1 month", ...]
  },
  "confidence": 0.98
}"""

        t0 = time.perf_counter()
        candidate_models = ["gemini-2.5-flash", "gemini-1.5-flash"]
        if self._model_name not in candidate_models:
            candidate_models.insert(0, self._model_name)
        # deduplicate while keeping order
        candidate_models = list(dict.fromkeys(candidate_models))

        last_error = None
        for m in candidate_models:
            try:
                if self._sdk_type == "google-genai":
                    response = self.client.models.generate_content(
                        model=m,
                        contents=[pil_img, user_prompt],
                        config=dict(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            temperature=0.0,
                        ),
                    )
                    raw_text = response.text.strip() if response.text else ""
                else:
                    response = self.legacy_client.generate_content([system_instruction + "\n\n" + user_prompt, pil_img])
                    raw_text = response.text.strip() if response.text else ""
                    if "```json" in raw_text:
                        raw_text = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
                    elif "```" in raw_text:
                        raw_text = raw_text.split("```", 1)[1].split("```", 1)[0].strip()

                if not raw_text:
                    continue

                data = json.loads(raw_text)
                elapsed = round(time.perf_counter() - t0, 2)

                lines_list = data.get("lines", [])
                full_text = data.get("full_text", "\n".join(lines_list)).strip()
                raw_entities = data.get("entities", {})
                confidence = float(data.get("confidence", 0.98))

                # Normalize entities format
                parsed_entities = {"MEDICINE": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}
                if isinstance(raw_entities, dict):
                    parsed_entities["MEDICINE"] = raw_entities.get("MEDICINE", raw_entities.get("medicine", []))
                    parsed_entities["DOSAGE"] = raw_entities.get("DOSAGE", raw_entities.get("dosage", []))
                    parsed_entities["FREQUENCY"] = raw_entities.get("FREQUENCY", raw_entities.get("frequency", []))
                    parsed_entities["DURATION"] = raw_entities.get("DURATION", raw_entities.get("duration", []))
                elif isinstance(raw_entities, list):
                    for item in raw_entities:
                        if isinstance(item, dict):
                            med = item.get("medication") or item.get("medicine") or item.get("name")
                            if med and med not in parsed_entities["MEDICINE"]:
                                parsed_entities["MEDICINE"].append(med)
                            dos = item.get("dosage") or item.get("strength")
                            if dos and dos not in parsed_entities["DOSAGE"]:
                                parsed_entities["DOSAGE"].append(dos)
                            freq = item.get("frequency")
                            if freq and freq not in parsed_entities["FREQUENCY"]:
                                parsed_entities["FREQUENCY"].append(freq)
                            dur = item.get("duration")
                            if dur and dur not in parsed_entities["DURATION"]:
                                parsed_entities["DURATION"].append(dur)

                if full_text or lines_list:
                    return {
                        "full_text": full_text,
                        "lines": lines_list if lines_list else full_text.splitlines(),
                        "entities": parsed_entities,
                        "confidence": confidence,
                        "elapsed_s": elapsed,
                    }
            except Exception as exc:
                last_error = exc
                continue

        return {
            "full_text": "",
            "lines": [],
            "entities": {"MEDICINE": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []},
            "confidence": 0.0,
            "error": str(last_error) if last_error else "Empty response from Gemini",
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }



    def _to_image_bytes(self, image: ImageInput) -> Tuple[bytes, str]:
        """Convert input image (Path, str, or PIL Image) to bytes and mime type."""
        if isinstance(image, (str, Path)):
            p = Path(image)
            mime, _ = mimetypes.guess_type(str(p))
            mime = mime or "image/jpeg"
            with open(p, "rb") as f:
                return f.read(), mime
        elif PILImage and isinstance(image, PILImage.Image):
            buf = io.BytesIO()
            fmt = image.format or "PNG"
            image.save(buf, format=fmt)
            mime = f"image/{fmt.lower()}"
            return buf.getvalue(), mime
        elif hasattr(image, "tobytes"):  # NumPy ndarray
            import cv2
            success, enc = cv2.imencode(".png", image)
            if success:
                return enc.tobytes(), "image/png"
            raise ValueError("Failed to encode ndarray image")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
