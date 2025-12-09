# 🚗 Car Price Predictor

A machine learning-powered web application that accurately predicts the market value of used cars based on vehicle specifications and current market trends. This project combines advanced ensemble modeling techniques with a user-friendly Flask web interface.

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Team](#-team)

## 📖 Overview
The **Car Price Predictor** aims to bring transparency to the used car market. By analyzing key factors such as brand, vehicle age, mileage, transmission type, and ownership history, the system provides a fair price estimation. The core prediction logic distinguishes between "Luxury," "Premium," and "Budget" brands to adjust valuations dynamically.

## 🚀 Key Features
* **Advanced ML Pipeline:** Utilizes a **Stacking Regressor** ensemble (Random Forest, Gradient Boosting, XGBoost) for high-accuracy predictions.
* **Smart Feature Engineering:** Handles outliers with `RobustScaler` and implements custom logic for luxury vs. premium brand segmentation.
* **Interactive Web Interface:** A responsive web app built with **Flask**, allowing users to input car details and get instant price estimates in INR (₹).
* **Robust Preprocessing:** Automated data cleaning pipeline including log transformation for target variables and One-Hot Encoding for categorical features.

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Web Framework:** Flask
* **Machine Learning:** Scikit-learn, XGBoost
* **Data Processing:** Pandas, NumPy
* **Model Persistence:** Joblib
* **Visualization:** Matplotlib, Seaborn (for analysis)
* **Frontend:** HTML/CSS (Embedded in Flask)

## ⚙️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/sisi-2002/Car-Price-Predictor.git](https://github.com/sisi-2002/Car-Price-Predictor.git)
    cd Car-Price-Predictor
    ```

2.  **Create a virtual environment (Optional but recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install pandas numpy scikit-learn xgboost flask joblib matplotlib
    ```

## 🖥️ Usage

### 1. Train the Model
Before running the application, you need to train the model and generate the necessary artifacts (`.joblib` files).
```bash
python train_and_predict.py
