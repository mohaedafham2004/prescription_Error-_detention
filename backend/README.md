# Smart Prescription NLP — FastAPI Backend

High-performance REST API wrapping the OCR, NER, and Drug Safety Error Detection pipeline.

---

## 1. Installation

Activate your virtual environment and install backend dependencies:

```powershell
# In Windows PowerShell from project root:
.\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

---

## 2. Running the Server

Start the API with Uvicorn:

```powershell
# Using Python module (Recommended):
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

The API will be available at:
- **Base URL**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 3. Endpoints

### `POST /api/analyze`
Accepts a prescription image scan via multipart form upload.

**Request**:
- `file`: `UploadFile` (PNG, JPG, JPEG)
- Optional `use_sample`: `true` to use the preloaded `sample_rx.png`

**Response Example**:
```json
{
  "extracted_text": "Amoxicillin 500mg TDS 7 days",
  "entities": {
    "medicine": ["Amoxicillin"],
    "dosage": ["500mg"],
    "frequency": ["TDS"],
    "duration": ["7 days"]
  },
  "issues": [],
  "ocr_model_used": "trocr",
  "ocr_confidence": 0.945,
  "ner_confidence": 0.920,
  "ner_available": true,
  "total_time_s": 1.45,
  "error": null
}
```

### `GET /api/metrics`
Returns benchmark performance data (CER/WER, NER Precision/Recall/F1, and CNN status) for frontend charting.

### `GET /api/health`
Returns system status and currently active OCR/NER models.

### `GET /api/sample-image`
Streams the preloaded sample prescription image (`sample_rx.png`).
