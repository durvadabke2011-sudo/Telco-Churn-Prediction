# 📊 Telco Customer Churn Prediction

A complete end-to-end machine learning project that predicts whether a telecom customer will churn, built with **scikit-learn** and deployed via a **Streamlit** web app.

---

## 📁 Project Structure

```
telco_churn_project/
│
├── app.py                          # Streamlit web application
├── Telco_Customer_Churn.ipynb      # Jupyter notebook (EDA + model training)
├── Telco-Customer-Churn.csv        # Dataset
├── churn_model.pkl                 # Pre-trained Random Forest model
├── scaler.pkl                      # Fitted StandardScaler
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Clone / unzip the project
```bash
unzip telco_churn_project.zip
cd telco_churn_project
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

---

## 🧠 Model Details

| Item | Detail |
|------|--------|
| Algorithm | Random Forest Classifier |
| Test Accuracy | ~73% |
| Features | 30 (after one-hot encoding) |
| Target | `Churn` (1 = Churns, 0 = Stays) |

### Features used
- **Demographic:** Gender, Senior Citizen, Partner, Dependents  
- **Account:** Tenure, Contract type, Paperless Billing, Payment Method  
- **Services:** Phone, Multiple Lines, Internet, Online Security, Backup, Device Protection, Tech Support, Streaming TV/Movies  
- **Charges:** Monthly Charges, Total Charges  

---

## 📓 Notebook Workflow

1. **Data Loading & Cleaning** — handle missing `TotalCharges`, drop `customerID`
2. **EDA** — distribution plots, churn rate by category, correlation heatmap
3. **Preprocessing** — binary encoding, outlier capping (IQR), `pd.get_dummies`, `StandardScaler`
4. **Modelling** — Logistic Regression, Decision Tree, Random Forest
5. **Evaluation** — Accuracy, Confusion Matrix, Classification Report, 5-fold CV
6. **Export** — saves `churn_model.pkl` and `scaler.pkl`

> ⚠️ If you retrain the model in the notebook, it will regenerate both `.pkl` files. The app will automatically use the updated files.

---

## 🌐 App Usage

1. Fill in all customer details in the form.
2. Click **🔍 Predict**.
3. The app will display:
   - ✅ Customer will **Stay** or ❌ Customer will **Churn**
   - Churn probability percentage
   - A visual risk progress bar

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.0.0
joblib>=1.2.0
streamlit>=1.25.0
```

---

## 📌 Notes

- The dataset (`Telco-Customer-Churn.csv`) is a sample of 1,000 rows representative of the original IBM Telco Churn dataset.
- The original full dataset (7,043 rows) is available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- Replace the CSV with the full dataset and re-run the notebook for better model performance.

---

## 🤝 Credits

- Dataset: IBM Sample Data / Kaggle  
- Framework: [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org)
