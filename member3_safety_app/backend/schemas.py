"""
backend/schemas.py
==================
Pydantic schemas for FastAPI request and response models.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EntitiesDict(BaseModel):
    medicine: List[str] = Field(default_factory=list, description="Extracted medicine names")
    dosage: List[str] = Field(default_factory=list, description="Extracted dosage values")
    frequency: List[str] = Field(default_factory=list, description="Extracted frequency instructions")
    duration: List[str] = Field(default_factory=list, description="Extracted duration values")


class IssueItem(BaseModel):
    error_type: str = Field(..., description="Issue classification e.g. INTERACTION, DUPLICATE_THERAPY, OUT_OF_RANGE")
    severity: str = Field(..., description="HIGH | MEDIUM | LOW | INFO")
    field: str = Field(..., description="Field triggering issue e.g. MEDICINE, DOSAGE")
    value: str = Field(..., description="The detected entity value")
    message: str = Field(..., description="Clinical issue description")
    suggestion: Optional[str] = Field(None, description="Actionable recommendation or clinical alternative")


class LineItem(BaseModel):
    line_index: int
    text: str
    confidence: float
    model_used: Optional[str] = None


class DrugMonograph(BaseModel):
    name: str
    generic_name: str
    brand_names: Optional[str] = None
    composition: Optional[str] = None
    manufacturer: Optional[str] = None
    therapeutic_class: Optional[str] = None
    dosage_forms: Optional[str] = None
    usage: Optional[str] = None
    standard_dosage: Optional[str] = None
    precautions: Optional[str] = None


class RiskAssessment(BaseModel):
    level: str = Field("clear", description="clear | low | medium | high")
    reason: str = Field("", description="Summary explanation for risk level")
    message: str = Field("", description="User-facing clinical recommendation message")


class AnalyzeResponse(BaseModel):
    extracted_text: str = Field(..., description="Full transcribed prescription text")
    entities: EntitiesDict = Field(..., description="Structured extracted entity fields")
    monographs: Dict[str, DrugMonograph] = Field(default_factory=dict, description="Detailed pharmaceutical monographs for detected medicines")
    issues: List[IssueItem] = Field(default_factory=list, description="List of detected errors, interactions, and duplications")
    risk: RiskAssessment = Field(default_factory=RiskAssessment, description="Overall risk level assessment")
    lines: List[LineItem] = Field(default_factory=list, description="Segmented text lines with per-line confidence")
    ocr_model_used: str = Field("trocr", description="Active OCR architecture")
    ocr_confidence: float = Field(0.0, description="Overall OCR confidence (0.0 - 1.0)")
    ner_confidence: float = Field(0.0, description="Estimated NER entity extraction confidence (0.0 - 1.0)")
    ner_available: bool = Field(True, description="Whether NER model ran successfully")
    total_time_s: float = Field(0.0, description="End-to-end processing time in seconds")
    error: Optional[str] = Field(None, description="Top-level error message if processing failed")


class SampleItem(BaseModel):
    id: str
    title: str
    category: str
    description: str
    filename: str


class MetricsResponse(BaseModel):
    active_ocr_model: str
    active_ner_model: str
    trocr: Dict[str, Any]
    ner: Dict[str, Any]
    cnn: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    active_ocr_model: str
    active_ner_model: str
    version: str = "1.0.0"
    trocr_available: bool = True
    ner_available: bool = True
