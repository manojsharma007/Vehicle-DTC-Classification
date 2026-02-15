# 🚗 Vehicle Diagnostic Trouble Code (DTC) Classification using Machine Learning

An end-to-end machine learning project for **predicting vehicle Diagnostic Trouble Codes (DTCs)** using OBD-II sensor data, with an interactive **Streamlit web application** for model evaluation and visualization.

---

## 📌 Problem Statement

Modern vehicles generate large volumes of sensor data through the On-Board Diagnostics (OBD-II) system.  
Manual interpretation of this data for fault diagnosis is time-consuming and requires expert knowledge.

The objective of this project is to **design, implement, and evaluate multiple machine learning classification models** that can automatically predict **vehicle Diagnostic Trouble Codes (DTCs)** based on OBD-II sensor parameters, and deploy the evaluation through an interactive Streamlit application.

---

## 📊 Dataset Description

The dataset used in this project consists of **OBD-II vehicle sensor readings**, capturing engine, throttle, load, and environmental parameters.

**Dataset details:**
- **Source:** Public OBD-II dataset (Kaggle)
- **Total records:** ~27,000+
- **Number of features:** 14 numerical sensor features
- **Target variable:** `dtc_code` (multi-class classification)

### Selected Input Features
- Barometric Pressure  
- Divers Demand Engine Percent Torque  
- Relative Throttle Position  
- Accelerator Pedal Position (D & E)  
- Consumption Rate  
- Load  
- Mass Air Flow  
- Speed  
- Ambient Air Temperature  
- RPM  
- Actual Engine Percent Torque  
- Fuel Level  
- Throttle Position  

### Target Classes (DTC Codes)
- `NORMAL`
- `P0115` – Engine coolant temperature sensor fault  
- `P0120` – Throttle position sensor fault  
- `P0171` – System too lean condition  
- `P0300` – Random/multiple cylinder misfire  

---

## 🧠 Machine Learning Models Implemented

The following classification models were trained and evaluated on the same dataset:

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

---

## 📈 Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.5504 | 0.8681 | 0.3239 | 0.7344 | 0.3525 | 0.1858 |
| Decision Tree | 0.9798 | 0.9591 | 0.8379 | 0.9334 | 0.8770 | 0.8671 |
| KNN | 0.9676 | 0.9431 | 0.6934 | 0.4540 | 0.5084 | 0.7495 |
| Naive Bayes | 0.8167 | 0.8926 | 0.3648 | 0.6444 | 0.4270 | 0.3351 |
| Random Forest | 0.9860 | 0.9983 | 0.8755 | 0.9035 | 0.8836 | 0.9007 |
| **XGBoost** | **0.9899** | **0.9992** | **0.9311** | **0.9241** | **0.9257** | **0.9284** |

---

## 🔍 Model-wise Observations

| Model | Observation |
|------|-------------|
| Logistic Regression | Struggled with complex non-linear relationships in sensor data. |
| Decision Tree | Achieved high accuracy but can overfit without proper constraints. |
| KNN | Sensitive to data distribution; lower recall for minority fault classes. |
| Naive Bayes | Fast and simple but limited by feature independence assumptions. |
| Random Forest | Strong and stable performance due to ensemble averaging. |
| XGBoost | Best overall performer across all evaluation metrics. |

---

## 🚀 Streamlit Web Application

A **Streamlit-based interactive web application** was developed and deployed using **Streamlit Community Cloud**.

### Key Features
- Upload labeled test dataset (CSV)
- Select machine learning model
- Display evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - MCC
- Visualize confusion matrix
- Clean and user-friendly interface

---

## 📁 Project Structure
vehicle-dtc-classification/
│── app.py
│── requirements.txt
│── README.md
│── model/
│ ├── logistic_regression.pkl
│ ├── decision_tree.pkl
│ ├── knn.pkl
│ ├── naive_bayes.pkl
│ ├── random_forest.pkl
│ ├── xgboost.pkl
│ ├── scaler.pkl
│ └── label_encoder.pkl

---

## ✅ Conclusion

This project demonstrates the effectiveness of ensemble machine learning techniques, particularly **XGBoost**, in diagnosing vehicle faults from OBD-II sensor data.  
The deployed Streamlit application enables interactive evaluation of trained models on labeled test datasets, making the system suitable for practical diagnostic analysis.

---

## 🔒 Academic Integrity Statement

All code, preprocessing steps, model training, and application development were performed independently.  
No external templates or plagiarized code were used in this project.

---


