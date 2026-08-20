"""
src/models/gemini_verifier.py
==============================
Multimodal Prescription Verification Assistant powered by Gemini 2.5 / 3.7 Flash.

Verifies OCR-extracted prescription text directly against the handwritten prescription image,
corrects misread tokens, extracts structured dosage/frequency/medications, and flags transcription errors.
"""

from __future__ import annotations

import os
import json
import mimetypes
from pathlib import Path
from typing import List, Literal, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

# ── Load environment / secrets safely ─────────────────────────────────────────
_ROOT_DIR = Path(__file__).resolve().parents[2]


def get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API Key from environment, .env file, or Streamlit secrets."""
    # 1. Environment variable
    if os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY")

    # 2. .env file
    env_path = _ROOT_DIR / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            os.environ["GEMINI_API_KEY"] = val
                            return val
        except Exception:
            pass

    # 3. Streamlit secrets
    secrets_path = _ROOT_DIR / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "GEMINI_API_KEY" in line and "=" in line:
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            os.environ["GEMINI_API_KEY"] = val
                            return val
        except Exception:
            pass

    return None


# ── Pydantic Output Schemas ───────────────────────────────────────────────────

class PatientInfo(BaseModel):
    name: str = Field(default="", description="Patient's full name if legible")
    age: str = Field(default="", description="Patient's age or DOB")
    gender: str = Field(default="", description="Patient's gender (e.g., M, F, Other)")


class VerifiedMedication(BaseModel):
    ocr_name: str = Field(..., description="Raw medicine name as extracted by initial OCR")
    verified_name: str = Field(..., description="Visually verified medicine name from handwriting")
    name_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for medicine name (0.0 - 1.0)")
    dose: str = Field(default="", description="Dosage quantity (e.g., '500', '10', '0.5')")
    dose_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence for dosage")
    unit: str = Field(default="", description="Dosage unit (e.g., 'mg', 'ml', 'mcg', 'IU')")
    frequency: str = Field(default="", description="Intake frequency (e.g., 'TID', 'twice daily', 'OD')")
    duration: str = Field(default="", description="Treatment duration (e.g., '7 days', '1 month')")
    instructions: str = Field(default="", description="Additional administration instructions (e.g., 'after meals')")
    status: Literal["MATCH", "CORRECTED", "UNCERTAIN"] = Field(..., description="Verification status")


class OCRCorrection(BaseModel):
    original: str = Field(..., description="Erroneous OCR token or phrase")
    corrected: str = Field(..., description="Corrected transcription from image")
    reason: str = Field(..., description="Explanation of error (e.g., letter misread 'rn' -> 'm', digit confusion)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Correction confidence score")


class PossibleError(BaseModel):
    field: str = Field(..., description="Affected field (e.g., 'dosage', 'drug_name', 'frequency')")
    value: str = Field(..., description="Problematic value found")
    issue: str = Field(..., description="Clinical or transcription issue detected (e.g., illegible zero, dangerous dose)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence that an error exists")


class PrescriptionVerificationResult(BaseModel):
    verification_status: Literal["MATCH", "CORRECTED", "UNCERTAIN"] = Field(
        ..., description="Overall verification outcome"
    )
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregated confidence score across all fields (0.0 - 1.0)"
    )
    patient: PatientInfo
    medications: List[VerifiedMedication] = Field(default_factory=list)
    ocr_corrections: List[OCRCorrection] = Field(default_factory=list)
    possible_errors: List[PossibleError] = Field(default_factory=list)
    human_review_required: bool = Field(
        ..., description="True if pharmacist / physician manual review is mandatory"
    )
    review_reasons: List[str] = Field(
        default_factory=list, description="Specific triggers requiring human intervention"
    )


# ── System Instruction Prompt ─────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are a prescription verification assistant in a Smart Prescription Error Detection system.

Your task is to verify the OCR result extracted from a handwritten medical prescription.

IMPORTANT:
- The prescription image is the primary source of truth.
- The OCR text may contain mistakes.
- Carefully inspect the handwriting in the provided image.
- Do NOT blindly trust the OCR text.
- Do NOT invent information that cannot be clearly seen.
- If handwriting is unclear, mark the field as "uncertain".
- Do not make a medical diagnosis.
- Do not recommend changing a doctor's prescription.
- Your role is ONLY to verify, correct, structure, and flag possible transcription errors.

Perform the following tasks:
1. Read the prescription from the image.
2. Compare the image transcription with the OCR output.
3. Identify possible OCR errors, including:
   - incorrect drug names
   - missing letters
   - extra letters
   - incorrect numbers
   - incorrect dosage
   - incorrect units
   - incorrect frequency
   - incorrect duration
   - missing medications
   - duplicated medications
   - incorrect patient information
4. Correct the OCR text ONLY when the handwriting provides sufficient evidence.
5. If a word or number cannot be confidently read:
   - do not guess
   - mark it as "uncertain"
6. Normalize medication information into structured fields where possible.
7. Calculate an OCR verification confidence score from 0 to 1.
8. Identify whether the OCR output and prescription image match.
9. Return the final verified prescription information.
10. Flag any item that requires human/pharmacist/doctor review.
"""


class GeminiPrescriptionVerifier:
    """Multimodal prescription verification client using Google GenAI SDK."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or get_gemini_api_key()
        if not self.api_key:
            raise ValueError(
                "Gemini API Key not found! Please set GEMINI_API_KEY in your environment, .env, or .streamlit/secrets.toml."
            )
        self.model = model
        self._init_client()

    def _init_client(self):
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._use_new_sdk = True
        except ImportError:
            # Fallback to google.generativeai if google-genai is not installed yet
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self.legacy_model = legacy_genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=SYSTEM_INSTRUCTION
                )
                self._use_new_sdk = False
            except ImportError:
                raise ImportError(
                    "Neither `google-genai` nor `google-generativeai` is installed. "
                    "Please install with: pip install google-genai"
                )

    def verify(
        self,
        image_path: Union[str, Path],
        ocr_text: str,
    ) -> PrescriptionVerificationResult:
        """
        Verify an OCR text extraction against the given prescription image.

        Parameters
        ----------
        image_path : str or Path
            Path to the prescription image file.
        ocr_text : str
            Raw text output from baseline OCR engine.

        Returns
        -------
        PrescriptionVerificationResult
            Parsed and validated structured verification report.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")

        mime_type, _ = mimetypes.guess_type(str(img_path))
        if not mime_type:
            mime_type = "image/jpeg"

        with open(img_path, "rb") as f:
            image_bytes = f.read()

        user_content = f"""INPUT:

OCR TEXT:
{ocr_text}

Analyze the provided prescription image together with the OCR text.
Return ONLY valid JSON matching the schema."""

        if self._use_new_sdk:
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    user_content
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=PrescriptionVerificationResult,
                    temperature=0.1,
                ),
            )
            raw_json = response.text
        else:
            from PIL import Image
            img = Image.open(img_path)
            prompt = f"{user_content}\n\nSchema:\n{PrescriptionVerificationResult.model_json_schema()}"
            response = self.legacy_model.generate_content([prompt, img])
            raw_json = response.text
            if "```json" in raw_json:
                raw_json = raw_json.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in raw_json:
                raw_json = raw_json.split("```", 1)[1].split("```", 1)[0].strip()

        return PrescriptionVerificationResult.model_validate_json(raw_json)
