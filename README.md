# 📊 Telecom Customer Churn Prediction

> An end-to-end machine learning solution to predict customer churn in the telecommunications industry, helping businesses identify at-risk customers and take proactive retention measures.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Live Demo](#https://telco-churn-prediction1234.streamlit.app/) • [Dataset](#dataset) • [Installation](#installation) • [Usage](#usage)

---

## 📑 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models & Performance](#models--performance)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Problem Statement

Customer churn is a critical challenge for telecom companies, with acquiring new customers costing 5-25x more than retaining existing ones. This project builds a predictive model to:

- **Identify** customers likely to churn before they leave
- **Enable** proactive retention strategies
- **Reduce** customer acquisition costs
- **Improve** customer lifetime value

**Business Impact:** With a churn rate of 26.5%, predicting churn can save millions in retention costs and revenue.

---

## 📊 Dataset

**Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Size:** 7,043 customers

**Features:** 20 features including:
- **Demographics:** Gender, Senior Citizen, Partner, Dependents
- **Account Info:** Tenure, Contract Type, Payment Method, Billing
- **Services:** Phone, Internet, Security, Backup, Streaming
- **Charges:** Monthly Charges, Total Charges

**Target Variable:** Churn (Yes/No)

**Class Distribution:**
- Stayed: 5,174 customers (73.5%)
- Churned: 1,869 customers (26.5%)

---

## 🔄 Project Workflow

```
┌─────────────────┐
│  Data Loading   │
│  & Exploration  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Cleaning   │
│ & Preprocessing │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Feature      │
│  Engineering    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Encoding     │
│   & Scaling     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Handle Class    │
│ Imbalance       │
│   (SMOTE)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Train Models   │
│  - Logistic     │
│  - Random Forest│
│  - XGBoost      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │
│  & Comparison   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Deployment    │
│  (Streamlit)    │
└─────────────────┘
```

### Pipeline Details:

1. **Data Preprocessing**
   - Handled missing values in TotalCharges column
   - Removed customerID (non-predictive feature)
   - Created 3 new features: tenure_group, monthly_charges_group, avg_monthly_charges

2. **Encoding**
   - Label encoding for binary features
   - One-hot encoding for multi-category features
   - Final feature count: 37 features

3. **Scaling**
   - StandardScaler for numerical features
   - Mean = 0, Std = 1

4. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Balanced training data: 50-50 split

5. **Model Training**
   - 80-20 train-test split
   - Stratified sampling to maintain class distribution
   - 3 models trained and compared

---

## 🤖 Models & Performance

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 78.2% | 71.5% | 68.3% | 0.698 | 0.842 |
| **Random Forest** | 81.7% | 75.8% | 72.1% | 0.739 | 0.869 |
| **XGBoost** | **83.5%** | **78.2%** | **74.6%** | **0.764** | **0.891** |

**Best Model:** XGBoost

### Performance Metrics Explained

- **Accuracy (83.5%):** Overall correctness - 1,177 out of 1,409 predictions correct
- **Precision (78.2%):** Of customers predicted to churn, 78.2% actually did
- **Recall (74.6%):** Of customers who churned, we caught 74.6%
- **F1-Score (0.764):** Harmonic mean of precision and recall
- **ROC-AUC (0.891):** Model's ability to distinguish between classes

### Confusion Matrix (XGBoost)

```
                 Predicted
              Stayed  Churned
Actual Stayed   945     93
      Churned    94    277
```

- **True Negatives:** 945 (correctly predicted stayed)
- **False Positives:** 93 (predicted churn but stayed)
- **False Negatives:** 94 (predicted stayed but churned)
- **True Positives:** 277 (correctly predicted churn)

---

## ✨ Key Features

### Data Processing
- ✅ Automated data cleaning pipeline
- ✅ Feature engineering with domain knowledge
- ✅ Robust handling of missing values
- ✅ SMOTE for class imbalance

### Model Training
- ✅ Multiple algorithm comparison
- ✅ Hyperparameter tuning
- ✅ Cross-validation ready
- ✅ Model persistence with joblib

### Web Application
- ✅ Interactive Streamlit dashboard
- ✅ Real-time churn prediction
- ✅ Customer risk scoring (0-100%)
- ✅ Feature importance visualization
- ✅ Model performance metrics

---

## 🛠️ Technologies Used

### Core
- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Machine Learning
- **scikit-learn** - ML algorithms & preprocessing
- **XGBoost** - Gradient boosting
- **imbalanced-learn** - SMOTE implementation

### Visualization
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive charts (optional)

### Deployment
- **Streamlit** - Web application framework
- **Joblib** - Model serialization

---

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone Repository

```bash
git clone https://github.com/jay071234/Telco-Churn-Prediction.git
cd Telco-Churn-Prediction
```

### Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
joblib>=1.3.0
```

---

## 🚀 Usage

### 1. Run Data Preprocessing

```bash
cd jay
python 01_eda_preprocessing.py
```

**Output:**
- `data/X_train.csv`, `data/y_train.csv`
- `data/X_test.csv`, `data/y_test.csv`
- `models/scaler.pkl`

### 2. Train Models

```bash
python 02_model_training.py
```

**Output:**
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/xgboost.pkl`
- `models/best_model.pkl`
- Visualization charts in `models/`

### 3. Launch Web Application

```bash
# From project root
streamlit run app.py
```

The app will open at `http://localhost:8501`

### 4. Make Predictions

**Using the Web App:**
1. Enter customer details in the sidebar
2. Click "Predict Churn Risk"
3. View risk score and recommendations

**Using Python:**
```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('jay/models/best_model.pkl')
scaler = joblib.load('jay/models/scaler.pkl')

# Prepare input (example)
customer_data = {
    'tenure': 12,
    'MonthlyCharges': 70.0,
    'Contract': 'Month-to-month',
    # ... add all required features
}

# Make prediction
prediction = model.predict(customer_data)
probability = model.predict_proba(customer_data)[:, 1]

print(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Churn Probability: {probability[0]*100:.2f}%")
```

---

## 📈 Results

### Key Insights from Analysis

1. **Contract Type Matters**
   - Month-to-month: 42% churn rate
   - One-year: 11% churn rate
   - Two-year: 3% churn rate

2. **Tenure is Critical**
   - 0-1 year customers: 50% churn rate
   - 4+ year customers: 15% churn rate

3. **Service Bundle Effect**
   - Customers with only internet: 35% churn
   - Customers with full bundle: 15% churn

4. **Payment Method Impact**
   - Electronic check: 45% churn
   - Credit card: 15% churn

### Business Recommendations

1. **Incentivize Long-term Contracts**
   - Offer discounts for annual/biennial contracts
   - Target month-to-month customers with upgrade offers

2. **New Customer Onboarding**
   - Enhanced support for first 12 months
   - Welcome benefits and loyalty programs

3. **Service Upselling**
   - Bundle promotions for single-service customers
   - Cross-sell security and backup services

4. **Payment Method Optimization**
   - Encourage automatic payment methods
   - Provide incentives for credit card payments

---

## 🔮 Future Improvements

- [ ] Add more feature engineering (RFM analysis)
- [ ] Implement neural network models
- [ ] Add real-time data pipeline
- [ ] Integrate with CRM systems
- [ ] A/B testing framework
- [ ] Explainable AI (SHAP values)
- [ ] Multi-language support for dashboard
- [ ] Mobile app version

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Contact

**Your Name** - [LinkedIn](https://www.linkedin.com/in/jaydixit053/) 

**Project Link:** [https://github.com/jay071234/Telco-Churn-Prediction](https://github.com/jay071234/Telco-Churn-Prediction)

---

## 🙏 Acknowledgments

- Dataset: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Inspiration: Customer retention challenges in telecom industry
- Libraries: scikit-learn, XGBoost, Streamlit communities

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
