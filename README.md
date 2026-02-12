# 📊 Retail Demand Forecasting System

[![CI Pipeline](https://github.com/tejonish/retail-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/tejonish/retail-demand-forecasting/actions)

End-to-end **Retail Demand Forecasting System** built using **XGBoost**, deployed via **Streamlit Cloud**, and integrated with **CI/CD pipelines using GitHub Actions**.

🔗 Live Demo: https://tejonish-demand-forecasting.streamlit.app/

---

## 🚀 Project Overview

This project implements a production-ready **time-series demand forecasting pipeline** for retail sales prediction.  

The system:

- Performs feature engineering using **lag features** and **rolling statistical windows**
- Prevents **data leakage** using time-aware validation
- Uses **XGBoost (Gradient Boosting)** for structured time-series modeling
- Implements **recursive multi-step forecasting (7-day horizon)**
- Serves predictions through an interactive **Streamlit dashboard**
- Includes **CI/CD automation and Docker build validation**

---

## 🧠 Key Machine Learning Concepts Implemented

- Time-series forecasting
- Autocorrelation handling
- Lag features (Lag_1, Lag_7)
- Rolling statistics (7, 14, 30-day means)
- Recursive multi-step prediction
- Time-aware train/validation split
- MAPE, MAE, RMSE evaluation metrics
- Data leakage detection and prevention

---

## 📈 Model Performance

Validation Performance:

- **MAPE:** ~10%
- **MAE:** ~633
- **RMSE:** ~909

The model demonstrates stable generalization without leakage and preserves weekly seasonality patterns.

---
## 🏗️ System Architecture

Raw Data → Feature Engineering → XGBoost Model Training
→ Model Serialization (joblib)
→ Streamlit Inference Layer
→ Recursive 7-Day Forecast Engine
→ Public Deployment


---

## 🌍 Deployment

The application is publicly deployed using:

- **Streamlit Cloud** (Continuous Deployment)
- GitHub repository integration

Every push to the `main` branch automatically triggers redeployment.

---

## 🔁 CI/CD Pipeline

Implemented using **GitHub Actions**:

### Continuous Integration (CI)
- Dependency validation
- Environment setup
- Import checks

### Continuous Deployment (CD)
- Automated Docker image build
- Production container validation
- Automatic redeployment on push

---

## 🐳 Docker Support

The project includes a `Dockerfile` for containerized deployment.

Docker image builds automatically via GitHub Actions to ensure production compatibility.

---

## 🛠️ Tech Stack

- Python
- XGBoost
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Git & GitHub
- GitHub Actions (CI/CD)
- Docker

---

## 📊 Features

- Interactive Store Selection
- Custom Forecast Start Date
- Recursive 7-Day Forecast
- Historical vs Forecast Visualization
- Clean Daily Date Axis
- Production-ready inference pipeline

---

## 💼 Business Impact

This system enables:

- Short-term inventory planning
- Demand spike detection
- Weekly seasonality modeling
- Retail store-level decision support

---

## 📌 Future Improvements

- Confidence intervals using quantile regression
- Hyperparameter optimization with Optuna
- Model registry integration
- Cloud storage for artifacts
- Batch forecasting for multiple stores

---

## 👤 Author

Nishanth B (Nish)

Machine Learning Engineer | Applied AI | Production ML Systems

---


## 🏗️ System Architecture

