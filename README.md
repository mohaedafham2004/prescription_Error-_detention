# prescription-error-detection-system
This is our 5th semester NLP module project 


## Group Information

**Group Number:** (Your Group Number)

**Project Title:** Smart Prescription Error Detection Using NLP

| Student ID     | Student Name  |
| -------------- | ------------- |
| CIT-24-01-0311 | M.F.M. Afham  |
| CIT-24-01-0006 | M.L. Abdullah |
| CIT-24-01-0495 | M.U. Sahy     |

---

# Section 1 – Project Overview

## 1. What NLP problem/application are you attempting to solve?

The healthcare industry depends heavily on accurate prescriptions to ensure patient safety. However, prescription errors such as incorrect drug names, dosage mistakes, dangerous drug interactions, and missing instructions can lead to severe health complications. Manual prescription verification is time-consuming and prone to human error. Our project aims to develop an NLP-based Prescription Error Detection System that automatically analyzes prescription text and identifies potential mistakes. The system will process textual prescriptions using Natural Language Processing techniques. It will extract drug names, dosages, frequencies, and administration instructions from prescriptions. The extracted information will then be validated against medical knowledge sources and predefined rules. Machine Learning and Deep Learning models will be used to classify prescriptions as safe or potentially problematic. The system will also provide explanations for detected errors. Healthcare professionals can use the tool to improve prescription accuracy. This solution aims to reduce medication errors and enhance patient safety. Ultimately, the project demonstrates the practical application of NLP in healthcare.

---

## 2. Why is this problem important?

Medication errors are among the leading causes of preventable healthcare complications worldwide. Incorrect prescriptions may result in adverse drug reactions, overdoses, or ineffective treatment. Pharmacists and healthcare providers often handle large volumes of prescriptions daily, increasing the likelihood of mistakes.

### Intended Users

* Pharmacists
* Doctors
* Hospitals
* Medical students

### Value Provided

* Early detection of prescription errors
* Improved patient safety
* Reduced healthcare costs
* Faster prescription verification process
* Decision support for healthcare professionals

---

## 3. What will be the final output of your application?

### Final Output

The system will generate:

* Classification Label:

  * Safe Prescription
  * Risky Prescription

* Error Detection Results:

  * Wrong Dosage
  * Drug Interaction
  * Missing Information
  * Invalid Drug Name

* Confidence Score

### Example

Input:

Metformin 500mg twice daily
Warfarin 5mg daily

Output:

Risky Prescription

Reason:
Potential Drug Interaction Detected

Confidence:
95%

---

# Section 2 – Dataset Information

## 4. Dataset Details

### Dataset Name

MTSamples Medical Transcriptions Dataset

### Dataset Source

[Kaggle MTSamples Dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions?utm_source=chatgpt.com)

### Number of Records

Approximately 5,000 medical transcription records

### Number of Classes

Multiple medical specialties

---

## 5. Why is this dataset suitable?

The dataset contains real-world medical transcription text with medication-related information. It provides rich healthcare language suitable for NLP tasks such as entity extraction and classification. Prescription-related patterns can be extracted and transformed into structured information. The dataset is freely available and contains sufficient textual content for training ML and DL models.

---

## 6. What challenges exist within the dataset?

### Challenges

**Noisy Text**

* Medical abbreviations
* Typographical variations

**Class Imbalance**

* Some medical categories have fewer records

**Medical Terminology**

* Complex vocabulary

**Context Dependency**

* Similar drugs may have different meanings depending on context

**Ethical Concerns**

* Medical recommendations must not replace professional judgment

---

# Section 3 – Individual NLP Pipeline

## Member 01 – M.F.M. Afham

### NLP Pipeline

| Step                 | Justification                        |
| -------------------- | ------------------------------------ |
| Data Collection      | Gather medical text data             |
| Text Cleaning        | Remove noise and unwanted characters |
| Tokenization         | Split text into words                |
| Stop Word Removal    | Reduce unnecessary words             |
| TF-IDF Vectorization | Convert text into numerical features |
| Model Training       | Train classification model           |
| Evaluation           | Measure performance                  |

### ML Model Selected

**Support Vector Machine (SVM)**

### Why?

SVM performs well on high-dimensional text classification tasks and is widely used in NLP.

### DL Model Selected

**LSTM**

### Why?

LSTM captures long-term dependencies and sequential relationships in prescription text.

### Contribution

* Data preprocessing
* Feature engineering
* SVM implementation
* Performance evaluation

---

## Member 02 – M.L. Abdullah

### NLP Pipeline

| Step                     | Justification              |
| ------------------------ | -------------------------- |
| Data Cleaning            | Remove inconsistencies     |
| Lemmatization            | Reduce words to root form  |
| Named Entity Recognition | Extract drug names         |
| Feature Extraction       | Prepare model inputs       |
| Training                 | Build classification model |
| Validation               | Evaluate results           |

### ML Model Selected

**Random Forest**

### Why?

Random Forest is robust against overfitting and handles complex feature relationships.

### DL Model Selected

**BiLSTM**

### Why?

BiLSTM captures context from both past and future words.

### Contribution

* Named Entity Recognition
* Random Forest implementation
* Drug extraction module

---

## Member 03 – M.U. Sahy

### NLP Pipeline

| Step                  | Justification                   |
| --------------------- | ------------------------------- |
| Text Normalization    | Standardize text                |
| Word Embedding        | Create semantic representations |
| Feature Selection     | Improve performance             |
| Model Training        | Train DL model                  |
| Error Detection Rules | Create medical validation rules |
| Testing               | Verify outputs                  |

### ML Model Selected

**XGBoost**

### Why?

XGBoost provides strong classification performance and handles feature interactions effectively.

### DL Model Selected

**GRU**

### Why?

GRU requires fewer parameters and trains faster than LSTM while maintaining good performance.

### Contribution

* Rule-based error detection
* XGBoost implementation
* Application integration

---

# Section 4 – Model Comparison Plan

## 7. Selected Models

| Member   | ML Model      | DL Model |
| -------- | ------------- | -------- |
| Afham    | SVM           | LSTM     |
| Abdullah | Random Forest | BiLSTM   |
| Sahy     | XGBoost       | GRU      |

---

## 8. Evaluation Metrics

### Metrics

* Accuracy
* Precision
* Recall
* F1-Score

### Justification

Accuracy alone may be misleading for imbalanced datasets. Precision measures how many detected errors are correct. Recall ensures dangerous prescriptions are not missed. F1-score balances precision and recall, making it suitable for healthcare applications.

---

# Section 5 – Final Application Plan

## 9. Application Type

### Web Application

Free technologies:

* Frontend: HTML, CSS, JavaScript
* Backend: Python Flask
* Database: SQLite
* Deployment: Render Free Tier

---

## 10. Simple Workflow

```text
User
  │
  ▼
Upload Prescription Text
  │
  ▼
Text Preprocessing
  │
  ▼
Named Entity Recognition
  │
  ▼
ML / DL Models
  │
  ▼
Error Detection Rules
  │
  ▼
Prediction
  │
  ▼
Safe / Risky Prescription
```

---

# Section 6 – Workload & Git Management

## 11. Work Distribution

| Task                     | Responsible Member |
| ------------------------ | ------------------ |
| Dataset Identification   | Afham              |
| Data Preprocessing       | Afham              |
| ML Model Development     | Abdullah           |
| DL Model Development     | Sahy               |
| Application Development  | All Members        |
| Model Evaluation         | All Members        |
| Report Writing           | Afham & Abdullah   |
| Presentation Preparation | All Members        |

---

## 12. Git Repository

### Repository Link

Create a free repository using:

[GitHub](https://github.com?utm_source=chatgpt.com)

Example:

```text
https://github.com/afhamfaiz/prescription-error-detection
```

### Branch Structure

```text
main

afham-branch

abdullah-branch

sahy-branch
```

---

# Section 7 – Ethics & Responsible AI

## 13. Potential Biases and Ethical Problems

* Dataset may not represent all medical conditions.
* Medical terminology may vary by country.
* Incorrect predictions could affect healthcare decisions.
* Training data may contain historical biases.

---

## 14. Could the system produce harmful outputs?

Yes.

Examples:

* Missing a dangerous drug interaction.
* Incorrectly flagging a safe prescription.
* Misinterpreting abbreviations.

---

## 15. How will risks be reduced?

* Use multiple evaluation metrics.
* Include rule-based verification.
* Human pharmacist review before final decisions.
* Use confidence thresholds.
* Clearly state system limitations.

---

## 16. Limitations

* Not a replacement for medical professionals.
* Performance depends on dataset quality.
* Limited coverage of rare medications.
* Cannot guarantee 100% accuracy.
* Requires continuous updates for new drugs.

---

# Completely Free Tech Stack

| Component       | Tool                   |
| --------------- | ---------------------- |
| Programming     | Python                 |
| NLP             | NLTK, spaCy            |
| ML              | Scikit-Learn           |
| DL              | TensorFlow / Keras     |
| Dataset         | Kaggle Medical Dataset |
| Database        | SQLite                 |
| Backend         | Flask                  |
| Frontend        | HTML/CSS/JS            |
| Version Control | GitHub                 |
| Deployment      | Render Free Tier       |
| Development     | VS Code                |

This project satisfies all assignment requirements:
✅ NLP-focused
✅ Unique ML model per member
✅ Unique DL model per member
✅ Real healthcare problem
✅ Free datasets and tools only
✅ Can be completed by 3 students within a semester.
#   p r e s c r i p t i o n _ E r r o r - _ d e t e n t i o n  
 #   p r e s c r i p t i o n _ E r r o r - _ d e t e n t i o n  
 