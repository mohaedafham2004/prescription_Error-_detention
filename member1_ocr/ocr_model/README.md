# prescription_ocr_pipeline

End-to-end handwritten prescription recognition pipeline integrated into **Smart-Prescription-NLP**.

## Architecture

```
prescription_ocr_pipeline/
├── data/
│   ├── drug_dictionary.json          # canonical drug names + aliases
│   └── ground_truth/
│       └── ground_truth.json         # per-image medication ground truth (fill in yours)
├── src/
│   ├── preprocess.py                 # image cleaning: grayscale, CLAHE, threshold, deskew
│   ├── recognize.py                  # dual-model OCR: TrOCR (primary) + CharCNN (fallback)
│   ├── postprocess.py                # regex + fuzzy drug extraction → structured entities
│   ├── evaluate.py                   # field-level accuracy scoring (medicine F1, dosage, freq, dur)
│   └── pipeline.py                   # CLI orchestrator tying all stages together
```

## Quick Start

```powershell
# From the project root (Smart-Prescription-NLP/)
cd "c:\Users\USER\OneDrive\Desktop\Cutermized\SLTC RESEARCH UNIVERSITY\5TH SEM\NLP\Smart-Prescription-NLP"

# Run on the 4 sample prescriptions (default hybrid mode)
.\venv\Scripts\python.exe prescription_ocr_pipeline\src\pipeline.py `
  --input data\raw\prescriptions `
  --mode hybrid `
  --verbose

# Run with accuracy evaluation
.\venv\Scripts\python.exe prescription_ocr_pipeline\src\pipeline.py `
  --input data\raw\prescriptions `
  --mode hybrid `
  --evaluate `
  --verbose

# Save preprocessing images + report
.\venv\Scripts\python.exe prescription_ocr_pipeline\src\pipeline.py `
  --input data\raw\prescriptions `
  --mode hybrid `
  --evaluate `
  --save-processed `
  --report reports `
  --verbose
```

## OCR Modes

| Mode     | Description |
|----------|-------------|
| `trocr`  | TrOCR only — best for clean/clear handwriting |
| `cnn`    | CharCNN only — fast character classifier |
| `hybrid` | TrOCR first; falls back to CharCNN if confidence < 0.45 |

## Ground Truth Format

```json
{
  "sample_rx.png": {
    "image": "sample_rx.png",
    "medications": [
      { "drug": "Amoxicillin", "strength": "500mg", "frequency": "TDS", "duration": "7 days" }
    ]
  }
}
```

Add more images by appending entries. Stub entries are provided for all 4 sample prescriptions.

## Accuracy Report

Saved to `reports/accuracy_report.json`. Contains:
- Per-image: medicine precision/recall/F1, dosage/frequency/duration hit flags, overall field score
- Aggregate: average medicine F1, dosage accuracy, frequency accuracy, duration accuracy, overall field score

## Fuzzy Threshold

Default is `70` (out of 100). Lower it to be more lenient, raise it for stricter matching:
```powershell
--threshold 60   # more lenient
--threshold 85   # stricter
```
