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

### Recommended Workflow

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/prescription-error-detection-system.git

# Enter project
cd prescription-error-detection-system

# Create/switch to your branch
git checkout -b afham-branch

# Check changes
git status

# Add changes
git add .

# Commit
git commit -m "Add NLP preprocessing pipeline"

# Push branch
git push -u origin afham-branch
```

Pull Requests should be used to merge completed work into `main`.

---

# 🔐 Security & Privacy

Because this project deals with healthcare-related text, privacy must be considered carefully.

The application should:

* Avoid storing real patient information.
* Avoid using identifiable patient data during development.
* Use anonymized or publicly available datasets.
* Avoid exposing sensitive information through logs.
* Clearly communicate that predictions are experimental.

---

# ⚖️ Ethics & Responsible AI

Potential risks include:

* Dataset bias
* Incorrect predictions
* False positives
* False negatives
* Misinterpretation of medical terminology
* Incomplete medication knowledge
* Differences in medical practices across countries

### Risk Mitigation

The system will attempt to reduce these risks by:

1. Evaluating multiple ML/DL models.
2. Using multiple evaluation metrics.
3. Combining machine learning with rule-based validation.
4. Providing confidence scores.
5. Clearly communicating system limitations.
6. Recommending professional verification.
7. Avoiding claims of clinical accuracy.

---

# ⚠️ Limitations

The proposed system has several limitations:

* It is not a replacement for healthcare professionals.
* The dataset may not represent all medications or medical conditions.
* Rare medications may not be adequately represented.
* Medical terminology can change over time.
* Model predictions may contain false positives and false negatives.
* The system cannot guarantee 100% accuracy.
* Drug interaction knowledge requires continuous maintenance.
* The MTSamples dataset is not specifically designed for prescription-error classification.

---

# 🚀 Future Improvements

Possible future improvements include:

* Integration with larger medical NLP datasets.
* Advanced transformer models such as BERT.
* Medical-specific language models.
* Real-time drug interaction databases.
* Improved Named Entity Recognition.
* Multilingual prescription processing.
* OCR-based handwritten prescription recognition.
* Explainable AI techniques.
* More comprehensive dosage validation.
* Integration with electronic health record systems.
* Continuous model retraining with validated data.

---

# 📚 Learning Outcomes

Through this project, the team will gain practical experience in:

* Natural Language Processing
* Text preprocessing
* Named Entity Recognition
* Feature engineering
* TF-IDF
* Word embeddings
* Machine Learning
* Deep Learning
* Model evaluation
* Flask web development
* Git & GitHub collaboration
* Responsible AI
* Healthcare NLP

---

# 📌 Project Status

🚧 **Status: In Development**

| Component             | Status         |
| --------------------- | -------------- |
| Project Planning      | ✅ Completed    |
| Dataset Selection     | ✅ Completed    |
| Data Preprocessing    | 🚧 In Progress |
| NER / Drug Extraction | 🚧 In Progress |
| SVM                   | 🚧 In Progress |
| Random Forest         | 🚧 In Progress |
| XGBoost               | 🚧 In Progress |
| LSTM                  | 🚧 In Progress |
| BiLSTM                | 🚧 In Progress |
| GRU                   | 🚧 In Progress |
| Rule-Based Validation | 🚧 In Progress |
| Web Application       | 🚧 In Progress |
| Model Comparison      | ⏳ Pending      |
| Final Testing         | ⏳ Pending      |
| Deployment            | ⏳ Pending      |

---

# 👨‍🎓 Academic Project

This project is developed as part of the **5th Semester Natural Language Processing (NLP) module**.

**Project:** Smart Prescription Error Detection Using NLP

**Team Members:**

* M.F.M. Afham
* M.L. Abdullah
* M.U. Sahy

---

## ⭐ Acknowledgements

We acknowledge the publicly available **MTSamples Medical Transcriptions Dataset** used as a source of medical text for this academic project.

---

## 📄 License

This project is intended primarily for **academic and educational purposes**.

Before redistributing the dataset or other third-party resources, review and comply with their respective licenses and terms of use.
