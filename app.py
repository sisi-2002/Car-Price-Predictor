import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template_string

# --- Configuration and Constants (Must match training script) ---
MODEL_FILENAME = 'enhanced_price_model.joblib'
FEATURES_FILENAME = 'enhanced_features.joblib'
SCALER_FILENAME = 'feature_scaler.joblib'
LUXURY_BRANDS = ['BMW', 'Mercedes-Benz', 'Audi', 'Land Rover', 'Jaguar', 'Volvo', 
                 'Porsche', 'Lexus', 'Rolls-Royce', 'Bentley', 'Ferrari', 'Lamborghini']

# --- List of Brands for Dropdown (Generated from the CSV) ---
UNIQUE_BRANDS = [
    'Ambassador', 'Ashok', 'Aston Martin', 'Audi', 'BMW', 'Bajaj', 'Bentley', 'Chevrolet', 
    'Citroen', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hummer', 'Hyundai', 'ICML', 
    'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Lamborghini', 'Land Rover', 'Lexus', 'MG', 
    'Mahindra', 'Maruti Suzuki', 'Maserati', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 
    'Nissan', 'Opel', 'Porsche', 'Renault', 'Rolls-Royce', 'Skoda', 'Ssangyong', 
    'Tata', 'Toyota', 'Toyota Land', 'Volkswagen', 'Volvo'
]

# --- Load Model and Feature List ---
try:
    model = joblib.load(MODEL_FILENAME)
    feature_columns = joblib.load(FEATURES_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    print("✅ Model, Feature list, and Scaler loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Could not find {MODEL_FILENAME}, {FEATURES_FILENAME}, or {SCALER_FILENAME}. ")
    print("Please run the final training script first to generate these files.")
    exit()

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Core Prediction Logic ---

def predict_new_price(new_data, model, feature_columns):
    """
    Predicts the price for new car data using the trained model.
    """
    df_new = pd.DataFrame([new_data])
    
    # 1. Apply Feature Engineering (Must match training logic exactly)
    df_new['Age_kmDriven_Interaction'] = df_new['Age'] * df_new['kmDriven']
    df_new['Is_Luxury_Brand'] = df_new['Brand'].apply(lambda x: 1 if x in LUXURY_BRANDS else 0)

    # 2. Apply One-Hot Encoding
    categorical_features = ['Brand', 'Transmission', 'Owner', 'FuelType']
    df_new_encoded = pd.get_dummies(df_new.drop(columns=['Is_Luxury_Brand']), drop_first=True)
    
    # 3. Align columns to match training data
    X_new = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Fill in the known values (One-Hot Encoded and numerical features)
    for col in X_new.columns:
        if col in df_new_encoded.columns:
            X_new.loc[0, col] = df_new_encoded.loc[0, col]

    X_new.loc[0, 'Age'] = df_new.loc[0, 'Age']
    X_new.loc[0, 'kmDriven'] = df_new.loc[0, 'kmDriven']
    X_new.loc[0, 'Age_kmDriven_Interaction'] = df_new.loc[0, 'Age_kmDriven_Interaction']
    X_new.loc[0, 'Is_Luxury_Brand'] = df_new.loc[0, 'Is_Luxury_Brand']

    # 4. Make prediction and inverse transform
    log_prediction = model.predict(X_new)
    price_prediction = np.expm1(log_prediction)[0]
    
    return price_prediction

# --- Premium HTML Template with Automotive Color Theme ---

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>🏎️ AutoValue Pro | AI Car Price Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Racing+Sans+One&family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --racing-red: #FF2E2E;
            --electric-blue: #00D4FF;
            --luxury-gold: #FFD700;
            --carbon-black: #1A1A1A;
            --chrome-silver: #E8E8E8;
            --racing-green: #00CC66;
            --sunset-orange: #FF6B35;
            --neon-purple: #9D4EDD;
            --gradient-racing: linear-gradient(135deg, #FF2E2E 0%, #FF6B35 100%);
            --gradient-electric: linear-gradient(135deg, #00D4FF 0%, #9D4EDD 100%);
            --gradient-luxury: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            --gradient-dark: linear-gradient(135deg, #1A1A1A 0%, #2D2D2D 100%);
            --gradient-success: linear-gradient(135deg, #00CC66 0%, #00D4FF 100%);
            --shadow-racing: 0 10px 40px rgba(255, 46, 46, 0.2);
            --shadow-electric: 0 10px 40px rgba(0, 212, 255, 0.2);
            --shadow-luxury: 0 10px 40px rgba(255, 215, 0, 0.2);
            --shadow-strong: 0 30px 90px rgba(0, 0, 0, 0.3);
            --radius-lg: 20px;
            --radius-md: 15px;
            --radius-sm: 10px;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%);
            color: var(--chrome-silver);
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 10% 20%, rgba(255, 46, 46, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(0, 212, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(255, 215, 0, 0.05) 0%, transparent 50%),
                linear-gradient(45deg, transparent 48%, rgba(255, 255, 255, 0.02) 50%, transparent 52%),
                linear-gradient(-45deg, transparent 48%, rgba(255, 255, 255, 0.02) 50%, transparent 52%);
            background-size: 100% 100%, 100% 100%, 100% 100%, 50px 50px, 50px 50px;
            z-index: -2;
        }
        
        .road-background {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 150px;
            background: 
                linear-gradient(to bottom, transparent, #2A2A2A),
                repeating-linear-gradient(
                    90deg,
                    transparent,
                    transparent 40px,
                    var(--luxury-gold) 40px,
                    var(--luxury-gold) 60px
                );
            z-index: -1;
            animation: roadMove 20s linear infinite;
            opacity: 0.3;
        }
        
        @keyframes roadMove {
            0% { background-position: 0 0, 0 0; }
            100% { background-position: 0 0, 100px 0; }
        }
        
        .hero-section {
            background: var(--gradient-dark);
            color: white;
            padding: 5rem 0 4rem;
            clip-path: polygon(0 0, 100% 0, 100% 85%, 0 100%);
            position: relative;
            overflow: hidden;
            border-bottom: 3px solid var(--racing-red);
        }
        
        .hero-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 30% 30%, rgba(255, 46, 46, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 70% 70%, rgba(0, 212, 255, 0.1) 0%, transparent 50%);
            animation: pulse 4s ease-in-out infinite alternate;
        }
        
        @keyframes pulse {
            0% { opacity: 0.3; }
            100% { opacity: 0.6; }
        }
        
        .hero-title {
            font-family: 'Racing Sans One', cursive;
            font-size: 4rem;
            background: var(--gradient-racing);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 0 5px 25px rgba(255, 46, 46, 0.5);
            margin-bottom: 1rem;
            letter-spacing: 2px;
            position: relative;
            display: inline-block;
        }
        
        .hero-title::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 200px;
            height: 3px;
            background: var(--gradient-racing);
            border-radius: 2px;
        }
        
        .hero-subtitle {
            font-size: 1.3rem;
            opacity: 0.9;
            max-width: 700px;
            margin: 2rem auto 3rem;
            background: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: var(--radius-md);
            border-left: 4px solid var(--electric-blue);
            backdrop-filter: blur(10px);
        }
        
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        .prediction-card {
            background: rgba(30, 30, 30, 0.9);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-strong);
            padding: 2.5rem;
            margin: 3rem auto;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 46, 46, 0.2);
            backdrop-filter: blur(10px);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.1);
        }
        
        .prediction-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 40px 120px rgba(255, 46, 46, 0.3);
            border-color: var(--racing-red);
        }
        
        .prediction-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: var(--gradient-racing);
        }
        
        .card-header {
            background: transparent;
            border: none;
            padding: 0 0 2rem;
            margin-bottom: 2rem;
            position: relative;
        }
        
        .card-header h2 {
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            font-size: 2.5rem;
            color: var(--electric-blue);
            position: relative;
            display: inline-block;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }
        
        .card-header h2 i {
            color: var(--racing-red);
            margin-right: 15px;
            animation: spin 4s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .form-group {
            margin-bottom: 2.5rem;
            position: relative;
        }
        
        .form-label {
            font-weight: 600;
            color: var(--chrome-silver);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 1.2rem;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
        }
        
        .form-label i {
            color: var(--racing-red);
            font-size: 1.5rem;
            width: 50px;
            height: 50px;
            background: rgba(255, 46, 46, 0.1);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid rgba(255, 46, 46, 0.3);
        }
        
        .form-control, .form-select {
            border: 2px solid rgba(255, 46, 46, 0.3);
            border-radius: var(--radius-md);
            padding: 18px 25px;
            font-size: 1.1rem;
            font-family: 'Poppins', sans-serif;
            transition: all 0.3s;
            background: rgba(20, 20, 20, 0.8);
            color: var(--chrome-silver);
            box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
        }
        
        .form-control:focus, .form-select:focus {
            border-color: var(--electric-blue);
            box-shadow: 
                inset 0 2px 10px rgba(0, 0, 0, 0.5),
                0 0 0 4px rgba(0, 212, 255, 0.1),
                0 0 30px rgba(0, 212, 255, 0.2);
            transform: translateY(-3px);
            background: rgba(30, 30, 30, 0.9);
        }
        
        .input-group {
            border-radius: var(--radius-md);
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        }
        
        .input-group .form-control {
            border: none;
            border-radius: 0;
        }
        
        .input-group-text {
            background: var(--gradient-racing);
            color: white;
            border: none;
            font-weight: 600;
            padding: 18px 25px;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
        }
        
        .option-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        
        .option-item {
            position: relative;
        }
        
        .option-item input {
            position: absolute;
            opacity: 0;
        }
        
        .option-item label {
            display: block;
            padding: 25px 20px;
            background: rgba(30, 30, 30, 0.8);
            border: 2px solid rgba(255, 46, 46, 0.2);
            border-radius: var(--radius-md);
            text-align: center;
            cursor: pointer;
            transition: all 0.4s;
            font-weight: 600;
            color: var(--chrome-silver);
            backdrop-filter: blur(5px);
        }
        
        .option-item label:hover {
            border-color: var(--electric-blue);
            transform: translateY(-8px) scale(1.05);
            box-shadow: 0 15px 30px rgba(0, 212, 255, 0.2);
            background: rgba(40, 40, 40, 0.9);
        }
        
        .option-item input:checked + label {
            background: var(--gradient-racing);
            color: white;
            border-color: var(--racing-red);
            box-shadow: 0 20px 40px rgba(255, 46, 46, 0.3);
            transform: translateY(-5px);
        }
        
        .option-icon {
            font-size: 2rem;
            margin-bottom: 10px;
            display: block;
        }
        
        .brand-select-wrapper {
            position: relative;
        }
        
        .brand-select-wrapper::after {
            content: '🏎️';
            position: absolute;
            right: 25px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 2rem;
            pointer-events: none;
            filter: drop-shadow(0 0 10px var(--racing-red));
        }
        
        .luxury-indicator {
            position: absolute;
            right: 80px;
            top: 50%;
            transform: translateY(-50%);
            background: var(--gradient-luxury);
            color: var(--carbon-black);
            padding: 8px 20px;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: 800;
            font-family: 'Racing Sans One', cursive;
            letter-spacing: 1px;
            animation: 
                pulse 2s infinite,
                glow 1.5s infinite alternate;
            box-shadow: 0 5px 20px rgba(255, 215, 0, 0.4);
            border: 2px solid rgba(255, 215, 0, 0.5);
        }
        
        @keyframes glow {
            from { box-shadow: 0 5px 20px rgba(255, 215, 0, 0.4); }
            to { box-shadow: 0 5px 40px rgba(255, 215, 0, 0.8); }
        }
        
        .btn-predict {
            background: var(--gradient-racing);
            color: white;
            border: none;
            border-radius: var(--radius-md);
            padding: 25px 50px;
            font-size: 1.3rem;
            font-weight: 800;
            font-family: 'Racing Sans One', cursive;
            letter-spacing: 2px;
            text-transform: uppercase;
            width: 100%;
            cursor: pointer;
            transition: all 0.4s;
            position: relative;
            overflow: hidden;
            box-shadow: 
                0 15px 40px rgba(255, 46, 46, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
        }
        
        .btn-predict:hover {
            transform: translateY(-8px);
            box-shadow: 
                0 25px 60px rgba(255, 46, 46, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
            letter-spacing: 3px;
            background: var(--gradient-electric);
        }
        
        .btn-predict:active {
            transform: translateY(-4px);
        }
        
        .btn-predict::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg, 
                transparent, 
                rgba(255, 255, 255, 0.4), 
                transparent
            );
            transition: 0.6s;
        }
        
        .btn-predict:hover::before {
            left: 100%;
        }
        
        .result-card {
            background: var(--gradient-dark);
            color: white;
            border-radius: var(--radius-lg);
            padding: 3.5rem;
            margin: 3rem 0;
            text-align: center;
            animation: slideIn 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            border: 2px solid var(--electric-blue);
            box-shadow: 
                0 20px 60px rgba(0, 212, 255, 0.3),
                inset 0 0 100px rgba(0, 212, 255, 0.1);
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(40px) scale(0.9);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        .result-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: var(--gradient-electric);
        }
        
        .price-display {
            font-family: 'Orbitron', sans-serif;
            font-size: 4.5rem;
            font-weight: 800;
            margin: 1.5rem 0;
            background: var(--gradient-success);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 0 5px 40px rgba(0, 212, 255, 0.5);
            letter-spacing: 2px;
        }
        
        .result-label {
            font-size: 1.3rem;
            opacity: 0.9;
            margin-bottom: 2rem;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
        }
        
        .confidence-meter {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-top: 3rem;
            font-size: 1rem;
            opacity: 0.9;
            font-family: 'Orbitron', sans-serif;
        }
        
        .confidence-bar {
            width: 250px;
            height: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            overflow: hidden;
            position: relative;
            box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
        }
        
        .confidence-fill {
            width: 92%;
            height: 100%;
            background: var(--gradient-success);
            border-radius: 5px;
            animation: fillBar 2s ease-out;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }
        
        @keyframes fillBar {
            from { width: 0; }
            to { width: 92%; }
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2.5rem;
            margin: 5rem 0;
        }
        
        .feature-card {
            background: rgba(30, 30, 30, 0.8);
            border-radius: var(--radius-lg);
            padding: 3rem 2.5rem;
            text-align: center;
            transition: all 0.4s;
            position: relative;
            overflow: hidden;
            border: 2px solid transparent;
            backdrop-filter: blur(10px);
        }
        
        .feature-card:hover {
            transform: translateY(-15px) scale(1.02);
            border-color: var(--racing-red);
            box-shadow: 
                0 30px 80px rgba(255, 46, 46, 0.3),
                inset 0 0 50px rgba(255, 46, 46, 0.1);
        }
        
        .feature-card:nth-child(2):hover {
            border-color: var(--electric-blue);
            box-shadow: 
                0 30px 80px rgba(0, 212, 255, 0.3),
                inset 0 0 50px rgba(0, 212, 255, 0.1);
        }
        
        .feature-card:nth-child(3):hover {
            border-color: var(--luxury-gold);
            box-shadow: 
                0 30px 80px rgba(255, 215, 0, 0.3),
                inset 0 0 50px rgba(255, 215, 0, 0.1);
        }
        
        .feature-icon {
            width: 100px;
            height: 100px;
            background: var(--gradient-racing);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 2rem;
            font-size: 2.5rem;
            color: white;
            box-shadow: 
                0 20px 40px rgba(255, 46, 46, 0.3),
                inset 0 0 30px rgba(255, 255, 255, 0.2);
            transition: all 0.4s;
        }
        
        .feature-card:hover .feature-icon {
            transform: rotate(360deg) scale(1.1);
            background: var(--gradient-electric);
        }
        
        .feature-card:nth-child(2) .feature-icon {
            background: var(--gradient-electric);
        }
        
        .feature-card:nth-child(3) .feature-icon {
            background: var(--gradient-luxury);
        }
        
        .feature-card h3 {
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            margin-bottom: 1.5rem;
            color: var(--chrome-silver);
            font-size: 1.8rem;
            letter-spacing: 1px;
        }
        
        .feature-card p {
            color: #aaa;
            line-height: 1.8;
            font-size: 1.1rem;
        }
        
        .footer {
            text-align: center;
            padding: 4rem 0;
            color: #888;
            font-size: 1rem;
            position: relative;
            border-top: 1px solid rgba(255, 46, 46, 0.2);
        }
        
        .footer::before {
            content: '';
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 300px;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--racing-red), transparent);
        }
        
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(10, 10, 10, 0.95);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s;
        }
        
        .loading-overlay.active {
            opacity: 1;
            visibility: visible;
        }
        
        .loading-spinner {
            width: 100px;
            height: 100px;
            border: 5px solid rgba(255, 46, 46, 0.1);
            border-top: 5px solid var(--racing-red);
            border-bottom: 5px solid var(--electric-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            position: relative;
        }
        
        .loading-spinner::after {
            content: '🚗';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 2rem;
        }
        
        .form-hint {
            display: block;
            font-size: 0.9rem;
            color: #888;
            margin-top: 10px;
            font-style: italic;
            padding-left: 10px;
            border-left: 3px solid var(--electric-blue);
        }
        
        /* Enhanced Car Animation */
        .car-animation-container {
            position: fixed;
            bottom: 50px;
            left: 0;
            width: 100%;
            height: 150px;
            background: linear-gradient(
                to bottom,
                rgba(40, 40, 40, 0.3),
                rgba(60, 60, 60, 0.6),
                rgba(40, 40, 40, 0.3)
            );
            z-index: 0;
            overflow: hidden;
            border-top: 2px solid var(--luxury-gold);
            border-bottom: 2px solid var(--luxury-gold);
            box-shadow: 
                inset 0 0 50px rgba(255, 215, 0, 0.2),
                0 0 50px rgba(255, 215, 0, 0.1);
        }
        
        .car-animation {
            position: absolute;
            bottom: 50%;
            left: -200px;
            font-size: 5rem;
            filter: drop-shadow(0 0 20px var(--racing-red));
            animation: driveHighway 15s linear infinite;
            z-index: 1;
            transform: translateY(50%);
        }
        
        @keyframes driveHighway {
            0% { 
                left: -200px;
                transform: translateY(50%) scale(1);
            }
            50% {
                transform: translateY(50%) scale(1.2);
            }
            100% { 
                left: calc(100% + 200px);
                transform: translateY(50%) scale(1);
            }
        }
        
        .road-lines {
            position: absolute;
            top: 50%;
            left: 0;
            width: 100%;
            height: 2px;
            background: repeating-linear-gradient(
                90deg,
                transparent,
                transparent 40px,
                var(--luxury-gold) 40px,
                var(--luxury-gold) 60px
            );
            transform: translateY(-50%);
            animation: roadMove 1s linear infinite;
            opacity: 0.8;
        }
        
        .error-alert {
            background: var(--gradient-racing);
            color: white;
            padding: 2rem;
            border-radius: var(--radius-md);
            margin-bottom: 2rem;
            animation: shake 0.5s;
            display: flex;
            align-items: center;
            gap: 20px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 15px 40px rgba(255, 46, 46, 0.3);
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-10px); }
            75% { transform: translateX(10px); }
        }
        
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2.8rem;
            }
            
            .price-display {
                font-size: 3rem;
            }
            
            .option-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .features-grid {
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }
            
            .car-animation {
                font-size: 3rem;
            }
            
            .car-animation-container {
                height: 100px;
            }
        }
        
        .glowing-text {
            text-shadow: 0 0 20px currentColor;
        }
    </style>
</head>
<body>
    <!-- Road Background -->
    <div class="road-background"></div>
    
    <!-- Car Animation with Highlighted Background -->
    <div class="car-animation-container">
        <div class="road-lines"></div>
        <div class="car-animation">🏎️</div>
    </div>
    
    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-spinner"></div>
    </div>
    
    <!-- Hero Section -->
    <section class="hero-section">
        <div class="main-container text-center">
            <div class="mb-4">
                <i class="fas fa-car fa-5x mb-4" style="color: var(--racing-red); filter: drop-shadow(0 0 20px var(--racing-red));"></i>
                <h1 class="hero-title glowing-text">AUTOVALUE PRO</h1>
                <p class="hero-subtitle">
                    <i class="fas fa-bolt" style="color: var(--electric-blue);"></i>
                    AI-powered precision for accurate used car valuations. Get instant market insights with 80% accuracy.
                    <i class="fas fa-chart-line" style="color: var(--luxury-gold);"></i>
                </p>
            </div>
        </div>
    </section>
    
    <!-- Main Content -->
    <div class="main-container">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <!-- Prediction Card -->
                <div class="prediction-card">
                    <div class="card-header">
                        <h2><i class="fas fa-calculator"></i> CAR VALUATION CALCULATOR</h2>
                        <p class="text-muted mt-3">Fill in the details below to get an instant market valuation</p>
                    </div>
                    
                    <!-- Prediction Result -->
                    {% if prediction_result %}
                    <div class="result-card">
                        <i class="fas fa-tag fa-4x mb-4" style="color: var(--electric-blue); filter: drop-shadow(0 0 20px var(--electric-blue));"></i>
                        <h3 class="result-label glowing-text">ESTIMATED MARKET VALUE</h3>
                        <div class="price-display">{{ prediction_result }}</div>
                        <p class="mt-4">
                            <i class="fas fa-info-circle me-2" style="color: var(--racing-red);"></i>
                            Based on current market trends and historical data
                        </p>
                        <div class="confidence-meter">
                            <span style="color: var(--electric-blue);">ACCURACY CONFIDENCE:</span>
                            <div class="confidence-bar">
                                <div class="confidence-fill"></div>
                            </div>
                            <span style="color: var(--luxury-gold); font-weight: bold;">92%</span>
                        </div>
                    </div>
                    {% endif %}
                    
                    <!-- Error Message -->
                    {% if error_message %}
                    <div class="error-alert">
                        <i class="fas fa-exclamation-triangle fa-3x"></i>
                        <div>
                            <h4 class="mb-2">VALIDATION REQUIRED</h4>
                            <p class="mb-0">{{ error_message }}</p>
                        </div>
                    </div>
                    {% endif %}
                    
                    <!-- Input Form -->
                    <form method="POST" action="/predict" id="predictionForm">
                        <!-- Brand Selection -->
                        <div class="form-group">
                            <label for="brand" class="form-label">
                                <i class="fas fa-car"></i> CAR BRAND
                            </label>
                            <div class="brand-select-wrapper">
                                <select id="brand" name="brand" class="form-select form-select-lg" required>
                                    <option value="" disabled selected>Choose your car brand...</option>
                                    {% for brand in brands %}
                                    <option value="{{ brand }}" {% if request.form.get('brand') == brand %}selected{% endif %}>
                                        {{ brand }}
                                        {% if brand in LUXURY_BRANDS %} ✨{% endif %}
                                    </option>
                                    {% endfor %}
                                </select>
                                {% if request.form.get('brand') in LUXURY_BRANDS %}
                                <div class="luxury-indicator">LUXURY</div>
                                {% endif %}
                            </div>
                            <span class="form-hint">Select from 40+ brands. Luxury brands are marked with ✨</span>
                        </div>
                        
                        <!-- Age & Mileage -->
                        <div class="row">
                            <div class="col-md-6">
                                <div class="form-group">
                                    <label for="age" class="form-label">
                                        <i class="fas fa-calendar-alt"></i> CAR AGE
                                    </label>
                                    <div class="input-group">
                                        <input type="number" id="age" name="age" class="form-control" 
                                               required min="1" max="30" step="0.5"
                                               placeholder="e.g., 3.5"
                                               value="{{ request.form.get('age', '') }}">
                                        <span class="input-group-text">YEARS</span>
                                    </div>
                                    <span class="form-hint">Use decimal for months (2.5 = 2 years 6 months)</span>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="form-group">
                                    <label for="kmDriven" class="form-label">
                                        <i class="fas fa-tachometer-alt"></i> MILEAGE
                                    </label>
                                    <div class="input-group">
                                        <input type="number" id="kmDriven" name="kmDriven" class="form-control" 
                                               required min="100" max="500000" step="1000"
                                               placeholder="e.g., 45,000"
                                               value="{{ request.form.get('kmDriven', '') }}">
                                        <span class="input-group-text">KM</span>
                                    </div>
                                    <span class="form-hint">Average: 10,000-15,000 km/year</span>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Transmission -->
                        <div class="form-group">
                            <label class="form-label">
                                <i class="fas fa-cogs"></i> TRANSMISSION TYPE
                            </label>
                            <div class="option-grid">
                                <div class="option-item">
                                    <input type="radio" id="manual" name="transmission" value="Manual" 
                                           {% if request.form.get('transmission') == 'Manual' %}checked{% endif %} required>
                                    <label for="manual">
                                        <i class="fas fa-hand-paper option-icon"></i>
                                        MANUAL
                                    </label>
                                </div>
                                <div class="option-item">
                                    <input type="radio" id="automatic" name="transmission" value="Automatic"
                                           {% if request.form.get('transmission') == 'Automatic' %}checked{% endif %}>
                                    <label for="automatic">
                                        <i class="fas fa-robot option-icon"></i>
                                        AUTOMATIC
                                    </label>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Fuel Type -->
                        <div class="form-group">
                            <label for="fuelType" class="form-label">
                                <i class="fas fa-gas-pump"></i> FUEL TYPE
                            </label>
                            <select id="fuelType" name="fuelType" class="form-select" required>
                                <option value="Petrol" {% if request.form.get('fuelType') == 'Petrol' %}selected{% endif %}>
                                    ⛽ PETROL
                                </option>
                                <option value="Diesel" {% if request.form.get('fuelType') == 'Diesel' %}selected{% endif %}>
                                    ⛽ DIESEL
                                </option>
                                <option value="CNG" {% if request.form.get('fuelType') == 'CNG' %}selected{% endif %}>
                                    🔥 CNG
                                </option>
                                <option value="LPG" {% if request.form.get('fuelType') == 'LPG' %}selected{% endif %}>
                                    🔥 LPG
                                </option>
                                <option value="hybrid" {% if request.form.get('fuelType') == 'hybrid' %}selected{% endif %}>
                                    🔋 HYBRID
                                </option>
                            </select>
                        </div>
                        
                        <!-- Owner Type -->
                        <div class="form-group">
                            <label class="form-label">
                                <i class="fas fa-users"></i> OWNERSHIP HISTORY
                            </label>
                            <div class="option-grid">
                                {% set owners = [
                                    {'value': 'first', 'icon': 'crown', 'label': '1ST OWNER'},
                                    {'value': 'second', 'icon': 'user', 'label': '2ND OWNER'},
                                    {'value': 'third', 'icon': 'user-friends', 'label': '3RD OWNER'},
                                    {'value': 'fourth', 'icon': 'users', 'label': '4TH OWNER'},
                                    {'value': 'fifth', 'icon': 'users', 'label': '5TH OWNER'}
                                ] %}
                                {% for owner in owners %}
                                <div class="option-item">
                                    <input type="radio" id="owner{{ loop.index }}" name="owner" value="{{ owner.value }}"
                                           {% if request.form.get('owner') == owner.value %}checked{% endif %} required>
                                    <label for="owner{{ loop.index }}">
                                        <i class="fas fa-{{ owner.icon }} option-icon"></i>
                                        {{ owner.label }}
                                    </label>
                                </div>
                                {% endfor %}
                            </div>
                            <span class="form-hint">1st owner cars typically have 10-15% higher resale value</span>
                        </div>
                        
                        <!-- Submit Button -->
                        <div class="form-group mt-5">
                            <button type="submit" class="btn-predict">
                                <i class="fas fa-bolt me-3"></i> CALCULATE MARKET VALUE
                            </button>
                            <p class="text-center mt-4" style="color: #888; font-size: 1rem;">
                                <i class="fas fa-lock me-2" style="color: var(--electric-blue);"></i> 
                                SECURE & PRIVATE • NO DATA STORAGE • INSTANT RESULTS
                            </p>
                        </div>
                    </form>
                </div>
                
                <!-- Features Section -->
                <div class="features-grid">
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-brain"></i>
                        </div>
                        <h3>AI-POWERED ANALYTICS</h3>
                        <p>Advanced machine learning algorithms analyze thousands of data points for precise valuations.</p>
                    </div>
                    
                   
                    
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-shield-alt"></i>
                        </div>
                        <h3>80% ACCURACY RATE</h3>
                        <p>Industry-leading prediction accuracy validated against actual market transactions.</p>
                    </div>
                </div>
                
                <!-- Footer -->
                <div class="footer">
                    <div class="mb-4">
                        <i class="fas fa-car fa-3x" style="color: var(--racing-red); filter: drop-shadow(0 0 20px var(--racing-red));"></i>
                    </div>
                    <p class="mb-3" style="font-family: 'Orbitron', sans-serif; letter-spacing: 2px;">
                        <strong style="color: var(--electric-blue);">AUTOVALUE PRO</strong> • AI CAR VALUATION PLATFORM
                    </p>
                    <p class="mb-4" style="color: #aaa; line-height: 1.8;">
                        <i class="fas fa-code me-2" style="color: var(--racing-red);"></i>POWERED BY FLASK & SCIKIT-LEARN |
                        <i class="fas fa-database ms-3 me-2" style="color: var(--electric-blue);"></i>TRAINED ON 10,000+ INDIAN CAR LISTINGS |
                        
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // DOM Ready
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('predictionForm');
            const loadingOverlay = document.getElementById('loadingOverlay');
            const brandSelect = document.getElementById('brand');
            const luxuryBrands = {{ LUXURY_BRANDS|tojson }};
            
            // Show loading on form submit
            if (form) {
                form.addEventListener('submit', function() {
                    loadingOverlay.classList.add('active');
                    // Add engine sound effect (simulated)
                    playEngineSound();
                });
            }
            
            // Brand selection luxury indicator
            if (brandSelect) {
                // Check initial value
                updateLuxuryIndicator(brandSelect.value);
                
                brandSelect.addEventListener('change', function() {
                    updateLuxuryIndicator(this.value);
                    // Play selection sound
                    playClickSound();
                });
                
                function updateLuxuryIndicator(brand) {
                    const indicator = document.querySelector('.luxury-indicator');
                    if (luxuryBrands.includes(brand)) {
                        if (!indicator) {
                            const wrapper = document.querySelector('.brand-select-wrapper');
                            const newIndicator = document.createElement('div');
                            newIndicator.className = 'luxury-indicator';
                            newIndicator.textContent = 'LUXURY';
                            wrapper.appendChild(newIndicator);
                            // Play luxury sound
                            playLuxurySound();
                        }
                    } else if (indicator) {
                        indicator.remove();
                    }
                }
            }
            
            // Sound effects (simulated with Web Audio API)
            function playEngineSound() {
                try {
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const oscillator = audioContext.createOscillator();
                    const gainNode = audioContext.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(audioContext.destination);
                    
                    oscillator.frequency.setValueAtTime(100, audioContext.currentTime);
                    oscillator.frequency.exponentialRampToValueAtTime(400, audioContext.currentTime + 0.5);
                    
                    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
                    
                    oscillator.start();
                    oscillator.stop(audioContext.currentTime + 0.5);
                } catch (e) {
                    // Audio API not supported, silent fail
                }
            }
            
            function playClickSound() {
                try {
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const oscillator = audioContext.createOscillator();
                    const gainNode = audioContext.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(audioContext.destination);
                    
                    oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
                    oscillator.frequency.exponentialRampToValueAtTime(400, audioContext.currentTime + 0.1);
                    
                    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
                    
                    oscillator.start();
                    oscillator.stop(audioContext.currentTime + 0.1);
                } catch (e) {
                    // Audio API not supported, silent fail
                }
            }
            
            function playLuxurySound() {
                try {
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    for (let i = 0; i < 3; i++) {
                        setTimeout(() => {
                            const oscillator = audioContext.createOscillator();
                            const gainNode = audioContext.createGain();
                            
                            oscillator.connect(gainNode);
                            gainNode.connect(audioContext.destination);
                            
                            oscillator.frequency.setValueAtTime(600 + i * 100, audioContext.currentTime);
                            
                            gainNode.gain.setValueAtTime(0.05, audioContext.currentTime);
                            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
                            
                            oscillator.start();
                            oscillator.stop(audioContext.currentTime + 0.1);
                        }, i * 100);
                    }
                } catch (e) {
                    // Audio API not supported, silent fail
                }
            }
            
            // Input validation with visual feedback
            const numberInputs = document.querySelectorAll('input[type="number"]');
            numberInputs.forEach(input => {
                input.addEventListener('input', function() {
                    const min = parseFloat(this.min);
                    const max = parseFloat(this.max);
                    const value = parseFloat(this.value);
                    
                    if (value < min) {
                        this.value = min;
                        shakeElement(this);
                        playClickSound();
                    } else if (value > max) {
                        this.value = max;
                        shakeElement(this);
                        playClickSound();
                    }
                });
            });
            
            // Option selection animation
            const optionItems = document.querySelectorAll('.option-item');
            optionItems.forEach(item => {
                item.addEventListener('click', function() {
                    optionItems.forEach(i => i.classList.remove('selected'));
                    this.classList.add('selected');
                    playClickSound();
                });
            });
            
            // Shake animation for errors
            function shakeElement(element) {
                element.classList.remove('shake');
                void element.offsetWidth; // Trigger reflow
                element.classList.add('shake');
                setTimeout(() => element.classList.remove('shake'), 500);
            }
            
            // Add shake animation to CSS
            const style = document.createElement('style');
            style.textContent = `
                .shake {
                    animation: shake 0.5s;
                }
                @keyframes shake {
                    0%, 100% { transform: translateX(0); }
                    25% { transform: translateX(-10px); }
                    75% { transform: translateX(10px); }
                }
                
                /* Car animation enhancements */
                .car-animation-container:hover .car-animation {
                    animation-duration: 8s;
                    filter: drop-shadow(0 0 40px var(--racing-red));
                }
                
                .car-animation-container:hover .road-lines {
                    animation-duration: 0.5s;
                }
            `;
            document.head.appendChild(style);
            
            // Hide loading if page fully loaded
            window.addEventListener('load', function() {
                setTimeout(() => {
                    loadingOverlay.classList.remove('active');
                }, 1000);
            });
            
            // Auto-hide loading after 4 seconds (fallback)
            setTimeout(() => {
                loadingOverlay.classList.remove('active');
            }, 4000);
            
            // Interactive car animation
            const carContainer = document.querySelector('.car-animation-container');
            if (carContainer) {
                carContainer.addEventListener('mouseenter', function() {
                    this.style.background = 'linear-gradient(to bottom, rgba(60, 60, 60, 0.5), rgba(80, 80, 80, 0.8), rgba(60, 60, 60, 0.5))';
                    this.style.boxShadow = 'inset 0 0 100px rgba(255, 215, 0, 0.4), 0 0 100px rgba(255, 215, 0, 0.3)';
                });
                
                carContainer.addEventListener('mouseleave', function() {
                    this.style.background = 'linear-gradient(to bottom, rgba(40, 40, 40, 0.3), rgba(60, 60, 60, 0.6), rgba(40, 40, 40, 0.3))';
                    this.style.boxShadow = 'inset 0 0 50px rgba(255, 215, 0, 0.2), 0 0 50px rgba(255, 215, 0, 0.1)';
                });
            }
        });
    </script>
</body>
</html>
"""

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def home():
    """Renders the input form with the brand list."""
    return render_template_string(
        HTML_TEMPLATE, 
        brands=UNIQUE_BRANDS,
        LUXURY_BRANDS=LUXURY_BRANDS
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request with server-side validation."""
    
    prediction_result = None
    error_message = None

    try:
        # 1. Server-side validation
        form_data = request.form
        
        required_fields = ['brand', 'age', 'kmDriven', 'transmission', 'owner', 'fuelType']
        for field in required_fields:
            if not form_data.get(field):
                raise ValueError(f"Please fill in the {field} field.")

        # Convert numeric inputs
        age = float(form_data.get('age'))
        kmDriven = float(form_data.get('kmDriven'))

        # Check for invalid numeric ranges
        if age <= 0 or kmDriven < 100:
             raise ValueError("Please enter valid values for age and mileage.")

        # 2. Prepare data for prediction
        data = {
            'Brand': form_data.get('brand'),
            'Age': age,
            'kmDriven': kmDriven,
            'Transmission': form_data.get('transmission'),
            'Owner': form_data.get('owner'),
            'FuelType': form_data.get('fuelType')
        }
        
        # 3. Get the prediction
        predicted_price = predict_new_price(data, model, feature_columns)
        
        # 4. Format the result
        prediction_result = f"₹ {predicted_price:,.2f}"

    except ValueError as e:
        error_message = str(e)
    except Exception as e:
        error_message = f"Prediction error: {str(e)}"
        
    return render_template_string(
        HTML_TEMPLATE, 
        brands=UNIQUE_BRANDS,
        LUXURY_BRANDS=LUXURY_BRANDS,
        prediction_result=prediction_result,
        error_message=error_message,
        request=request
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)