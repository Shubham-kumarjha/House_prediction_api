# 🏠 California House Price Prediction API

A production-ready **FastAPI** web service that predicts house prices using a trained Machine Learning model. The API supports both single-data predictions via JSON payloads and bulk file-based batch predictions with downloadable CSV outputs.

---

## ✨ Features

- **Single Prediction Endpoint (`/predict`)**: Input single housing data features and receive real-time price estimations and confidence ranges.
- **Batch CSV Prediction (`/predict_file`)**: Upload a structured CSV file and receive an updated CSV stream containing predicted prices.
- **Data Validation**: Uses **Pydantic** schemas for robust request validation and error handling.
- **Health Check (`/health`)**: Monitor API operational status, model configuration, and average error margins.
- **Interactive Swagger Documentation**: Auto-generated interactive API docs available at `/docs`.

---

## 🛠️ Tech Stack & Tools

- **Backend Framework**: FastAPI
- **Data Processing**: Pandas
- **Machine Learning**: Scikit-Learn / Joblib
- **Validation**: Pydantic
- **Server**: Uvicorn

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your local machine.

### 2. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Shubham-kumarjha/House_prediction_api.git](https://github.com/Shubham-kumarjha/House_prediction_api.git)
   cd House_prediction_api