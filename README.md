<div align="center">

# 🛡️ FraudLens AI

### Explainable Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-006600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.46-blueviolet?style=for-the-badge)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*An end-to-end AI-powered fraud detection system with explainable predictions, batch analysis, and interactive analytics dashboard.*

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Tech Stack](#-tech-stack) • [ML Pipeline](#-machine-learning-pipeline)

</div>

---

## 📖 Overview

**FraudLens AI** is a production-ready, end-to-end machine learning application built to detect fraudulent credit card transactions with high precision and recall. The system goes beyond simple binary classification by incorporating **Explainable AI (XAI)** through SHAP values, giving analysts and stakeholders transparent insight into *why* a transaction is flagged as fraudulent.

The application features a multi-page **Streamlit dashboard** that supports single-transaction prediction with real-time SHAP explanations, batch processing of CSV files for bulk fraud scanning, and an interactive analytics dashboard with rich visualizations powered by Plotly. Whether you're a data scientist exploring model behavior or a fraud analyst reviewing flagged transactions, FraudLens AI provides the tools you need.

Built with a focus on handling **extreme class imbalance** (only 0.17% of transactions are fraudulent), the pipeline leverages SMOTE oversampling, stratified splitting, and careful model evaluation to deliver reliable predictions in real-world conditions.

---

## 🎯 Problem Statement

Credit card fraud is a rapidly growing threat in the global financial ecosystem. According to the **Nilson Report**, global card fraud losses are projected to exceed **$32 billion annually**, affecting millions of consumers and financial institutions worldwide. Traditional rule-based fraud detection systems are increasingly inadequate against sophisticated fraud schemes that evolve faster than static rules can adapt.

Machine learning offers a powerful alternative — but ML models are often treated as "black boxes," making it difficult for compliance teams, auditors, and regulators to trust or act on their outputs. **FraudLens AI** addresses both challenges simultaneously: it delivers **high-accuracy fraud detection** using ensemble learning while providing **full model transparency** through SHAP-based explainability. Every prediction comes with a clear breakdown of which features contributed most, empowering human decision-makers to validate and act on the model's recommendations with confidence.

---

## ✨ Features

- 🔍 **Single Transaction Prediction** — Input individual transaction features and receive instant fraud/legitimate classification with confidence scores
- 📊 **SHAP Explainability** — Every prediction is accompanied by a SHAP force plot showing exactly which features pushed the prediction toward fraud or legitimate
- 📁 **Batch Fraud Scanner** — Upload a CSV file of transactions for bulk analysis; download results with fraud probabilities and flags
- 📈 **Interactive Analytics Dashboard** — Explore dataset distributions, fraud patterns, and model performance metrics through rich Plotly visualizations
- ⚖️ **Imbalanced Data Handling** — SMOTE oversampling applied to training data ensures the model learns effectively from the rare fraud class
- 🚀 **Cloud-Ready Deployment** — Pre-configured for one-click deployment to Render with `render.yaml`; Streamlit Cloud compatible out of the box

---

## 🖥️ Demo

<!-- Screenshot placeholders -->
<!-- Add screenshots of your running application here -->

> **📸 Screenshots Coming Soon**
>
> To add screenshots, run the app locally and capture:
> 1. Home page with feature overview
> 2. Single prediction page with SHAP force plot
> 3. Batch scanner with uploaded results
> 4. Analytics dashboard with visualizations

```
🔗 Live Demo: [Coming Soon]
```

---

## 📊 Dataset Information

The project uses the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle, originally collected by the **Machine Learning Group at ULB (Université Libre de Bruxelles)**.

| Property | Value |
|---|---|
| **Total Transactions** | 284,807 |
| **Features** | 31 (Time, V1–V28, Amount, Class) |
| **Fraudulent Transactions** | 492 (0.1727%) |
| **Legitimate Transactions** | 284,315 (99.8273%) |
| **PCA-Transformed Features** | V1 – V28 (confidential original features) |
| **Non-Transformed Features** | Time, Amount |
| **Target Variable** | Class (0 = Legitimate, 1 = Fraud) |
| **File Size** | ~144 MB |

> [!NOTE]
> Features V1 through V28 are the result of a **PCA transformation** applied to the original transaction data for confidentiality. The original feature names and background context are not available. Only `Time` and `Amount` retain their original meaning.

---

## 🔬 Machine Learning Pipeline

The complete ML pipeline is documented in the Jupyter notebook at [`notebook/Fraud_Detection.ipynb`](notebook/Fraud_Detection.ipynb).

### 1️⃣ Data Loading & Exploratory Data Analysis

- Loaded 284,807 transactions from `creditcard.csv`
- Analyzed class distribution revealing extreme imbalance (99.83% legitimate vs 0.17% fraud)
- Explored feature distributions, correlations, and statistical summaries
- Identified key patterns in `Time`, `Amount`, and PCA-transformed features

### 2️⃣ Train-Test Split

- **80/20 stratified split** to preserve the fraud ratio in both training and test sets
- Ensures model evaluation reflects real-world class distribution
- Test set kept completely untouched during training

### 3️⃣ SMOTE Oversampling (Training Data Only)

- Applied **Synthetic Minority Oversampling Technique (SMOTE)** to the training set
- Generated synthetic fraud samples to balance the training class distribution
- SMOTE applied **only to training data** — test set remains untouched to ensure honest evaluation
- Used `imbalanced-learn` library for robust implementation

### 4️⃣ Model Training & Comparison

Three classification models were trained and evaluated:

- **Logistic Regression** — Baseline linear model
- **Random Forest** — Ensemble of decision trees with bagging
- **XGBoost** — Gradient-boosted decision trees with regularization

### 5️⃣ Evaluation & Model Selection

- Models evaluated using **Precision**, **Recall**, **F1 Score**, and **ROC-AUC**
- Emphasis on **F1 Score** as the primary metric due to class imbalance
- Random Forest selected as the production model based on best overall performance
- Final model serialized with `joblib` for deployment

---

## 📈 Model Comparison

Performance on the **test set** (unseen data, original class distribution):

| Model | Precision | Recall | F1 Score | Selected |
|---|---|---|---|---|
| Logistic Regression | 0.07 | 0.91 | 0.12 | ❌ |
| Random Forest | 0.83 | 0.83 | 0.83 | ✅ |
| XGBoost | 0.53 | 0.89 | 0.66 | ❌ |

> [!IMPORTANT]
> **Random Forest** was selected as the production model due to its superior **F1 Score of 0.83**, achieving the best balance between precision (minimizing false positives) and recall (catching actual fraud). Logistic Regression had high recall but extremely low precision, generating too many false alarms. XGBoost performed well but couldn't match Random Forest's precision.

---

## 🧠 Explainable AI (XAI)

FraudLens AI integrates **SHAP (SHapley Additive exPlanations)** to make every prediction transparent and interpretable.

### How It Works

- **TreeExplainer** is used for the Random Forest model, providing exact SHAP values with polynomial-time complexity
- Each prediction generates a **SHAP force plot** showing positive (pushing toward fraud) and negative (pushing toward legitimate) feature contributions
- The base value represents the model's average prediction, and each feature's SHAP value shows its marginal contribution

### Understanding the Features

Since features **V1 through V28** are PCA-transformed for confidentiality:

- Individual feature names (V1, V2, etc.) don't carry human-readable meaning
- However, SHAP values still reveal **which principal components** the model relies on most heavily
- The `Amount` feature retains its original meaning and is directly interpretable
- SHAP plots help identify if the model is making decisions based on sensible patterns

### Why Explainability Matters

- 🏦 **Regulatory Compliance** — Financial institutions must explain automated decisions
- 🔍 **Model Debugging** — Identify if the model relies on spurious correlations
- 🤝 **Stakeholder Trust** — Non-technical stakeholders can verify model reasoning
- 📋 **Audit Trail** — Every flagged transaction has a documented explanation

---

## 📁 Project Structure

```
FraudLens-AI/
│
├── 📄 app.py                          # Main Streamlit application
├── 📄 requirements.txt                # Python dependencies (pinned versions)
├── 📄 render.yaml                     # Render deployment configuration
├── 📄 .gitignore                      # Git ignore rules
│
├── 📁 .streamlit/
│   └── config.toml                    # Streamlit theme & server config
│
├── 📁 notebook/
│   └── Fraud_Detection.ipynb          # Full ML pipeline (EDA → Training → Evaluation)
│
├── 📁 sample_files/
│   └── transactions.csv              # Sample CSV for batch prediction testing
│
├── 🤖 fraud_detection_model.pkl       # Trained Random Forest model (serialized)
├── 📋 feature_names.pkl               # Feature name list (serialized)
└── 📊 creditcard.csv                  # Dataset (284,807 transactions, ~144 MB)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/FraudLens-AI.git
cd FraudLens-AI

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Download creditcard.csv from Kaggle:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place it in the project root directory

# 5. Run the application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> [!TIP]
> If you already have the trained model files (`fraud_detection_model.pkl` and `feature_names.pkl`), you can skip the notebook and run the app directly. To retrain, execute the Jupyter notebook in `notebook/Fraud_Detection.ipynb`.

---

## ☁️ Deployment

### Deploy to Render

FraudLens AI includes a pre-configured `render.yaml` for one-click deployment to [Render](https://render.com).

**Steps:**

1. Push your code to a GitHub repository
2. Log in to [Render Dashboard](https://dashboard.render.com)
3. Click **"New"** → **"Blueprint"**
4. Connect your GitHub repository
5. Render will auto-detect `render.yaml` and configure the service
6. Click **"Apply"** to deploy

```yaml
# render.yaml is pre-configured with:
# - Python 3.11 runtime
# - Auto-install dependencies
# - Streamlit server settings for production
```

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repository, branch, and `app.py` as the main file
4. Click **Deploy**

> [!WARNING]
> The `creditcard.csv` dataset is ~144 MB. For cloud deployments, consider using Git LFS or loading the dataset from an external source to stay within platform storage limits.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.9+ |
| **Web Framework** | Streamlit 1.45 |
| **ML Models** | Logistic Regression, Random Forest, XGBoost |
| **Explainability** | SHAP (TreeExplainer) |
| **Data Processing** | Pandas, NumPy |
| **Imbalanced Learning** | imbalanced-learn (SMOTE) |
| **Visualization** | Plotly, Matplotlib, SHAP Plots |
| **Model Serialization** | Joblib |
| **Deployment** | Render, Streamlit Cloud |
| **Version Control** | Git & GitHub |

---

## 🔮 Future Improvements

- 🌊 **Real-Time Streaming** — Integrate with Apache Kafka for real-time transaction monitoring and alerting
- 🧬 **Deep Learning Models** — Experiment with autoencoders and LSTMs for anomaly detection
- 🔌 **REST API Endpoint** — Build a FastAPI wrapper for programmatic access and microservice integration
- 🐳 **Docker Containerization** — Dockerize the application for portable, reproducible deployments
- 🧪 **A/B Testing Framework** — Implement model versioning and A/B testing for production model updates
- 📱 **Mobile-Responsive UI** — Optimize the dashboard layout for tablet and mobile access
- 🗄️ **Database Integration** — Store predictions and audit logs in PostgreSQL for compliance tracking

---

## 🏆 Resume-Worthy Achievements

This project demonstrates production-level ML engineering skills:

- 🔧 **Built an end-to-end ML pipeline** processing **284,807+ transactions** from raw data to deployed web application
- 📊 **Achieved 0.83 F1 Score** with Random Forest on a highly imbalanced dataset (0.17% fraud class) — a non-trivial result requiring careful sampling strategy
- 🧠 **Integrated SHAP for model explainability**, providing transparent, auditable predictions that meet financial industry compliance standards
- 🖥️ **Developed a multi-page Streamlit dashboard** with single prediction, batch processing, and analytics capabilities
- 📁 **Implemented batch processing pipeline** with CSV upload, bulk prediction, and downloadable results
- ⚖️ **Applied SMOTE oversampling** correctly (training data only) to handle extreme class imbalance without data leakage
- ☁️ **Configured cloud deployment** with Render blueprint for production-ready hosting
- 📈 **Compared 3 ML algorithms** (Logistic Regression, Random Forest, XGBoost) with rigorous evaluation methodology

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👤 Author

Sudulaguntla Usha Chowdary

- GitHub: [@yourusername](https://github.com/UshaSudulaguntla-11/)


---

<div align="center">

**⭐ If you found this project useful, please consider giving it a star!**

Made with ❤️ and Python

</div>
