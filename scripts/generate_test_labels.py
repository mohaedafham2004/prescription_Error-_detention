import json
import os
from pathlib import Path

labels_dir = Path("data/nlp_test_image/labels")
labels_dir.mkdir(parents=True, exist_ok=True)

labels = {
    "test_image_1.jpeg": {
        "image": "test_image_1.jpeg",
        "medications": [
            {"drug": "Aspirin", "strength": "75mg", "frequency": "OD", "duration": None},
            {"drug": "Atorvastatin", "strength": "40mg", "frequency": "OD", "duration": None},
            {"drug": "Losartan", "strength": "50mg", "frequency": "OD", "duration": None},
            {"drug": "Amlodipine", "strength": None, "frequency": "OD", "duration": None},
            {"drug": "Bisoprolol", "strength": "2.5mg", "frequency": "OD/BD", "duration": None}
        ]
    },
    "test_image_2.jpeg": {
        "image": "test_image_2.jpeg",
        "medications": [
            {"drug": "Iron", "strength": "50g", "frequency": "BD", "duration": None}
        ]
    },
    "test_image_3.jpeg": {
        "image": "test_image_3.jpeg",
        "medications": [
            {"drug": "Flunarizine", "strength": "10mg", "frequency": "OD", "duration": "1 month"},
            {"drug": "Diclofenac", "strength": "50mg", "frequency": "PRN", "duration": None},
            {"drug": "Domperidone", "strength": "10mg", "frequency": "PRN", "duration": None},
            {"drug": "Etoricoxib", "strength": "90mg", "frequency": "OD", "duration": "1 week"},
            {"drug": "Pantoprazole", "strength": "40mg", "frequency": "OD", "duration": "1 month"}
        ]
    },
    "test_image_4.jpeg": {
        "image": "test_image_4.jpeg",
        "medications": [
            {"drug": "Enoxaparin", "strength": "60mg", "frequency": "BD", "duration": "48 hours"},
            {"drug": "Aspirin", "strength": "75mg", "frequency": "OD", "duration": None},
            {"drug": "Clopidogrel", "strength": "75mg", "frequency": "OD", "duration": None},
            {"drug": "Atorvastatin", "strength": "40mg", "frequency": "OD", "duration": None},
            {"drug": "Bisoprolol", "strength": "2.5mg", "frequency": "BD", "duration": None},
            {"drug": "Spironolactone", "strength": "25mg", "frequency": "OD", "duration": None},
            {"drug": "Nicorandil", "strength": "10mg", "frequency": "BD", "duration": None},
            {"drug": "Pantoprazole", "strength": "40mg", "frequency": "BD", "duration": None},
            {"drug": "Glyceryl Trinitrate", "strength": "1 tab", "frequency": "PRN", "duration": None}
        ]
    },
    "test_image_5.jpeg": {
        "image": "test_image_5.jpeg",
        "medications": [
            {"drug": "Rabeprazole", "strength": None, "frequency": "HS", "duration": None},
            {"drug": "Deflazacort", "strength": "0.5mg", "frequency": "HS", "duration": None}
        ]
    },
    "test_image_6.jpeg": {
        "image": "test_image_6.jpeg",
        "medications": [
            {"drug": "Aspirin", "strength": "100mg", "frequency": None, "duration": None},
            {"drug": "Pantoprazole", "strength": "40mg", "frequency": None, "duration": None},
            {"drug": "Nermigan", "strength": None, "frequency": "OD", "duration": None}
        ]
    },
    "test_image_7.jpeg": {
        "image": "test_image_7.jpeg",
        "medications": [
            {"drug": "Atorvastatin", "strength": "20mg", "frequency": "OD", "duration": None},
            {"drug": "Gliclazide", "strength": "80mg", "frequency": "BD", "duration": None},
            {"drug": "Metformin", "strength": "500mg", "frequency": "BD", "duration": None},
            {"drug": "Insulin Mixtard", "strength": "100 IU/ml", "frequency": "BD", "duration": None}
        ]
    },
    "test_image_8.jpeg": {
        "image": "test_image_8.jpeg",
        "medications": [
            {"drug": "Fluoxetine", "strength": "20mg", "frequency": "BD", "duration": None},
            {"drug": "Clomipramine", "strength": "100mg", "frequency": "OD", "duration": None},
            {"drug": "Mirtazapine", "strength": "15mg", "frequency": "OD", "duration": None},
            {"drug": "Risperidone", "strength": "20mg", "frequency": "BD", "duration": None},
            {"drug": "Clonazepam", "strength": "1mg", "frequency": "BD", "duration": None}
        ]
    },
    "test_image_9.jpeg": {
        "image": "test_image_9.jpeg",
        "medications": [
            {"drug": "Acyclovir", "strength": "Cream", "frequency": "BD", "duration": None},
            {"drug": "Emollient", "strength": "Cream", "frequency": "TDS", "duration": None},
            {"drug": "Antifungal", "strength": "Powder", "frequency": "5 times/day", "duration": "5 days"}
        ]
    }
}

for img_name, data in labels.items():
    stem = Path(img_name).stem
    out_file = labels_dir / f"{stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_file.name} with {len(data['medications'])} medication(s)")

# Also update prescription_ocr_pipeline/data/ground_truth/ground_truth.json to include all 9 + 4 = 13 images
gt_file = Path("prescription_ocr_pipeline/data/ground_truth/ground_truth.json")
gt_data = {}
if gt_file.exists():
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
gt_data.update(labels)
with open(gt_file, "w", encoding="utf-8") as f:
    json.dump(gt_data, f, indent=2)
print(f"Updated combined ground_truth.json with {len(gt_data)} total entries")
