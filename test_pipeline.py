"""
Test script to check all pipeline components for errors.
Run from project root: python test_pipeline.py
"""
import sys
import traceback

ROOT = "c:/Users/USER/OneDrive/Desktop/Cutermized/SLTC RESEARCH UNIVERSITY/5TH SEM/NLP/Smart-Prescription-NLP"

print("=" * 60)
print("PIPELINE DIAGNOSTIC TEST")
print("=" * 60)

errors = []

# Test 1: config_loader
print("\n[1] config_loader...")
try:
    from src.utils.config_loader import load_config
    cfg = load_config()
    active_ocr = cfg.get("active_ocr_model", "N/A")
    active_ner = cfg.get("active_ner_model", "N/A")
    print(f"    OK  active_ocr={active_ocr}, active_ner={active_ner}")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("config_loader", str(e)))
    traceback.print_exc()

# Test 2: risk_assessment
print("\n[2] src.pipeline.risk_assessment...")
try:
    from src.pipeline.risk_assessment import assess_risk
    result = assess_risk([], ocr_confidence=0.95, ner_confidence=0.9)
    print(f"    OK  level={result.get('level')}")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("risk_assessment", str(e)))
    traceback.print_exc()

# Test 3: error_detection
print("\n[3] src.pipeline.error_detection...")
try:
    from src.pipeline.error_detection import ErrorDetector, issues_to_dict_list
    det = ErrorDetector()
    issues = det.check({"MEDICINE": ["Amoxicillin"], "DOSAGE": ["500mg"], "FREQUENCY": ["twice daily"], "DURATION": ["7 days"]})
    print(f"    OK  {len(issues)} issues found")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("error_detection", str(e)))
    traceback.print_exc()

# Test 4: drug_monographs
print("\n[4] src.pipeline.drug_monographs...")
try:
    from src.pipeline.drug_monographs import get_drug_monograph
    mono = get_drug_monograph("Amoxicillin")
    print(f"    OK  monograph={'found' if mono else 'not found (normal)'}")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("drug_monographs", str(e)))
    traceback.print_exc()

# Test 5: ner_pipeline
print("\n[5] src.pipeline.ner_pipeline (NER)...")
try:
    from src.pipeline.ner_pipeline import run_ner_pipeline
    from src.utils.config_loader import load_config
    cfg = load_config()
    result = run_ner_pipeline("Amoxicillin 500mg twice daily 7 days", cfg)
    print(f"    OK  ner_available={result.get('ner_available')} entities={result.get('entities')}")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("ner_pipeline", str(e)))
    traceback.print_exc()

# Test 6: medicine_matcher
print("\n[6] src.pipeline.medicine_matcher...")
try:
    from src.pipeline.medicine_matcher import *
    print("    OK")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("medicine_matcher", str(e)))
    traceback.print_exc()

# Test 7: ner_labeler
print("\n[7] src.pipeline.ner_labeler...")
try:
    from src.pipeline.ner_labeler import *
    print("    OK")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("ner_labeler", str(e)))
    traceback.print_exc()

# Test 8: full_pipeline import
print("\n[8] src.pipeline.full_pipeline import...")
try:
    from src.pipeline.full_pipeline import run_full_pipeline
    print("    OK")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("full_pipeline_import", str(e)))
    traceback.print_exc()

# Test 9: backend FastAPI imports
print("\n[9] backend FastAPI routers...")
try:
    sys.path.insert(0, "backend")
    from backend.app.routers.health import router as health_router, HealthStatus
    from backend.app.routers.analyze import router as analyze_router
    from backend.app.routers.metrics import router as metrics_router
    from backend.app.routers.samples import router as samples_router
    print("    OK  all routers imported")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("backend_routers", str(e)))
    traceback.print_exc()

# Test 10: member2/member3 module imports
print("\n[10] member2_nlp.ner_pipeline...")
try:
    from member2_nlp.ner_pipeline.ner_pipeline import run_ner_pipeline as run_ner2
    print("    OK")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("member2_ner_pipeline", str(e)))

print("\n[11] member3_safety_app.error_detection...")
try:
    from member3_safety_app.error_detection.error_detection import ErrorDetector as ED3
    print("    OK")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("member3_error_detection", str(e)))

print("\n[12] member3_safety_app.error_detection.risk_assessment...")
try:
    from member3_safety_app.error_detection.risk_assessment import assess_risk as ar3
    print("    OK")
except Exception as e:
    print(f"    ERROR: {e}")
    errors.append(("member3_risk_assessment", str(e)))

print("\n" + "=" * 60)
if errors:
    print(f"FOUND {len(errors)} ERROR(S):")
    for name, msg in errors:
        print(f"  - {name}: {msg}")
else:
    print("ALL TESTS PASSED - No pipeline errors found!")
print("=" * 60)
