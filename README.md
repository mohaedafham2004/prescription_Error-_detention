# 💊 Smart Prescription Error Detection Using NLP

An NLP-based healthcare application that analyzes prescription text and identifies potential medication-related errors such as incorrect dosages, invalid drug names, missing information, and potentially dangerous drug interactions.

> **5th Semester NLP Module Project**

---

## 📌 Project Overview

Medication errors are a serious healthcare concern. Incorrect drug names, dosage mistakes, drug interactions, and incomplete prescription instructions can potentially lead to adverse health outcomes.

The **Smart Prescription Error Detection System** aims to use **Natural Language Processing (NLP), Machine Learning (ML), Deep Learning (DL), and rule-based validation** to automatically analyze prescription text and identify potential risks.

The system extracts important information such as:

* 💊 Drug names
* 📏 Dosages
* ⏰ Frequencies
* 📋 Administration instructions
* ⚠️ Potential medication errors

The detected information is then analyzed using machine learning/deep learning models and predefined validation rules.

### ⚠️ Important Disclaimer

This project is an **academic prototype** developed for educational purposes. It is **not a medical diagnostic system** and must not replace doctors, pharmacists, or other qualified healthcare professionals.

---

# 👥 Group Information

**Group Number:** *To be updated*

**Project Title:** Smart Prescription Error Detection Using NLP

| Student ID     | Student Name  |
| -------------- | ------------- |
| CIT-24-01-0311 | M.F.M. Afham  |
| CIT-24-01-0006 | M.L. Abdullah |
| CIT-24-01-0495 | M.U. Sahy     |

---

# 🎯 Project Objectives

The main objectives of this project are to:

1. Process prescription-related medical text using NLP techniques.
2. Extract drug names, dosage information, frequencies, and instructions.
3. Detect potentially incorrect or risky prescription information.
4. Compare different Machine Learning classification algorithms.
5. Compare different Deep Learning models.
6. Implement rule-based medication validation.
7. Provide understandable explanations for detected risks.
8. Evaluate the performance of different models using standard classification metrics.
9. Develop a simple web-based interface for prescription analysis.

---

# 🧠 NLP Problem

The system focuses on **medical text processing and prescription error classification**.

The general process is:

```text
Prescription Text
       │
       ▼
Text Preprocessing
       │
       ▼
Named Entity Recognition
       │
       ▼
Drug / Dosage / Frequency Extraction
       │
       ▼
Feature Extraction
       │
       ▼
ML / DL Classification
       │
       ▼
Rule-Based Validation
       │
       ▼
Error Detection
       │
       ▼
Final Result
```

---

# 🔍 Types of Errors

The proposed system focuses on detecting potential prescription problems such as:

| Error Type               | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| 💊 Invalid Drug Name     | Drug name cannot be recognized or validated          |
| 📏 Wrong Dosage          | Potentially inappropriate dosage information         |
| ⚠️ Drug Interaction      | Potential interaction between prescribed medications |
| 📋 Missing Information   | Important prescription information is missing        |
| ❓ Ambiguous Instructions | Instructions are unclear or incomplete               |

---

# 🖥️ Expected Application Output

The application will provide:

### Classification

* ✅ **Safe Prescription**
* ⚠️ **Risky Prescription**

### Error Information

* Error type
* Detected medication
* Explanation
* Confidence score

### Example

**Input**

```text
Metformin 500mg twice daily
Warfarin 5mg daily
```

**Expected Output**

```text
Classification: Risky Prescription

Potential Issue:
Drug Interaction Detected

Confidence:
95%
```

> The displayed result is an example for demonstrating the system workflow and does not represent clinical advice.

---

# 📊 Dataset

## MTSamples Medical Transcriptions Dataset

The project uses the **MTSamples Medical Transcriptions Dataset** as the primary source of medical text.

**Dataset:** MTSamples Medical Transcriptions

**Source:** Kaggle

The dataset contains approximately **5,000 medical transcription records** covering multiple medical specialties.

The dataset provides realistic medical language that can be useful for experimenting with:

* Medical text preprocessing
* Medical terminology extraction
* Named Entity Recognition
* Text classification
* Feature engineering
* NLP model development

### Dataset Source

https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

---

# ⚠️ Dataset Challenges

The dataset presents several NLP challenges.

### 1. Noisy Text

Medical transcription data may contain:

* Abbreviations
* Typographical variations
* Formatting inconsistencies
* Unstructured text

### 2. Medical Terminology

Medical documents contain specialized vocabulary that can be difficult for general NLP models to interpret.

### 3. Context Dependency

The meaning of medical terms can depend heavily on the surrounding context.

### 4. Class Imbalance

Some medical categories may contain significantly more examples than others.

### 5. Prescription-Specific Data

The MTSamples dataset is primarily a **medical transcription dataset**, rather than a dedicated prescription-error dataset. Therefore, prescription-related information may need to be extracted and transformed into suitable training examples.

---

# 🤖 Machine Learning & Deep Learning Models

Each group member is responsible for implementing different ML and DL models.

| Member        | Machine Learning | Deep Learning |
| ------------- | ---------------- | ------------- |
| M.F.M. Afham  | SVM              | LSTM          |
| M.L. Abdullah | Random Forest    | BiLSTM        |
| M.U. Sahy     | XGBoost          | GRU           |

---

# 👨‍💻 Member Contributions

## Member 01 — M.F.M. Afham

### NLP Pipeline

```text
Data Collection
      ↓
Text Cleaning
      ↓
Tokenization
      ↓
Stop Word Removal
      ↓
TF-IDF Vectorization
      ↓
Model Training
      ↓
Evaluation
```

### ML Model

**Support Vector Machine (SVM)**

SVM is suitable for high-dimensional text classification problems and is commonly used with sparse features such as TF-IDF.

### DL Model

**Long Short-Term Memory (LSTM)**

LSTM networks can capture sequential dependencies within medical text.

### Responsibilities

* Dataset collection
* Data preprocessing
* Text cleaning
* Feature engineering
* TF-IDF implementation
* SVM implementation
* Model evaluation

---

## Member 02 — M.L. Abdullah

### NLP Pipeline

```text
Data Cleaning
      ↓
Lemmatization
      ↓
Named Entity Recognition
      ↓
Feature Extraction
      ↓
Model Training
      ↓
Validation
```

### ML Model

**Random Forest**

Random Forest can model nonlinear relationships between features and provides a robust baseline for classification.

### DL Model

**Bidirectional LSTM (BiLSTM)**

BiLSTM processes sequences in both forward and backward directions, allowing the model to use contextual information from both sides of a word.

### Responsibilities

* Named Entity Recognition
* Drug entity extraction
* Lemmatization
* Random Forest implementation
* Feature extraction
* Model validation

---

## Member 03 — M.U. Sahy

### NLP Pipeline

```text
Text Normalization
      ↓
Word Embedding
      ↓
Feature Selection
      ↓
Model Training
      ↓
Error Detection Rules
      ↓
Testing
```

### ML Model

**XGBoost**

XGBoost is a powerful gradient boosting algorithm that can model complex feature interactions.

### DL Model

**Gated Recurrent Unit (GRU)**

GRU is a recurrent neural network architecture that generally uses fewer parameters than LSTM while maintaining the ability to model sequential information.

### Responsibilities

* Text normalization
* Word embedding
* Feature selection
* XGBoost implementation
* Rule-based error detection
* GRU implementation
* Application integration

---

# 📈 Model Evaluation

The models will be evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

### Why These Metrics?

Accuracy alone may not be sufficient when dealing with imbalanced datasets.

**Precision** measures how many predicted errors are actually errors.

**Recall** measures how many actual errors are successfully detected.

**F1-Score** provides a balance between precision and recall.

For a healthcare-related error detection system, **recall is particularly important** because failing to detect a potentially dangerous prescription may have serious consequences.

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │        User         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Prescription Input │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Text Preprocessing  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Named Entity        │
                         │ Recognition         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Extraction  │
                         └──────────┬──────────┘
                                    │
                      ┌─────────────┴─────────────┐
                      │                           │
                      ▼                           ▼
             ┌────────────────┐          ┌─────────────────┐
             │ ML / DL Models │          │ Validation Rules│
             └────────┬───────┘          └────────┬────────┘
                      │                           │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Error Classification│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Final Result        │
                         │ Safe / Risky        │
                         └─────────────────────┘
```

---

# 🌐 Web Application

The final project will be implemented as a web application.

### Workflow

```text
User
  │
  ▼
Enter Prescription
  │
  ▼
Submit
  │
  ▼
NLP Processing
  │
  ▼
Drug & Dosage Extraction
  │
  ▼
ML / DL Prediction
  │
  ▼
Rule-Based Validation
  │
  ▼
Error Analysis
  │
  ▼
Result Dashboard
```

---

# 🛠️ Technology Stack

| Component            | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python                       |
| NLP                  | NLTK, spaCy                  |
| Machine Learning     | Scikit-learn                 |
| Deep Learning        | TensorFlow / Keras           |
| ML Models            | SVM, Random Forest, XGBoost  |
| DL Models            | LSTM, BiLSTM, GRU            |
| Backend              | Flask                        |
| Frontend             | HTML, CSS, JavaScript        |
| Database             | SQLite                       |
| Dataset              | MTSamples                    |
| Version Control      | Git & GitHub                 |
| Development          | VS Code                      |
| Deployment           | Free hosting where available |

---

# 📁 Proposed Project Structure

```text
prescription-error-detection-system/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── data-exploration.ipynb
│   ├── preprocessing.ipynb
│   ├── svm.ipynb
│   ├── random-forest.ipynb
│   ├── xgboost.ipynb
│   ├── lstm.ipynb
│   ├── bilstm.ipynb
│   └── gru.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── nlp/
│   ├── models/
│   ├── rules/
│   └── evaluation/
│
├── app/
│   ├── static/
│   ├── templates/
│   └── app.py
│
├── models/
│   └── README.md
│
├── tests/
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

# 👥 Work Distribution

| Task                    | Responsible Member |
| ----------------------- | ------------------ |
| Dataset Identification  | Afham              |
| Data Preprocessing      | Afham              |
| ML Model Development    | Abdullah           |
| DL Model Development    | Sahy               |
| Application Development | All Members        |
| Model Evaluation        | All Members        |
| Report Writing          | Afham & Abdullah   |
| Presentation            | All Members        |

---

# 🌿 Git Branch Strategy

The project uses separate branches for individual development.

```text
main
│
├── afham-branch
│
├── abdullah-branch
│
└── sahy-branch
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
#   p r e s c r i p t i o n _ E r r o r - _ d e t e n t i o n 
 
 #   p r e s c r i p t i o n _ E r r o r - _ d e t e n t i o n 
 
 