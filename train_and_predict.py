# train_model.py
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
DATA_FILE = 'used_cars_dataset_v2.csv'
PIPELINE_FILE = 'best_pipeline.joblib'
RANDOM_STATE = 42

LUXURY_BRANDS = ['BMW', 'Mercedes-Benz', 'Audi', 'Land Rover', 'Jaguar', 'Volvo',
                 'Porsche', 'Lexus', 'Rolls-Royce', 'Bentley', 'Ferrari', 'Lamborghini',
                 'Maserati', 'Aston Martin', 'McLaren']

PREMIUM_BRANDS = ['Honda', 'Toyota', 'Hyundai', 'Volkswagen', 'Skoda', 'Nissan', 'Ford']

# -------------------------
# Helpers
# -------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    eps = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def detect_outliers_iqr(df, column, multiplier=3.0):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# -------------------------
# Feature engineering transformer (pipeline friendly)
# -------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, luxury_brands=None, premium_brands=None):
        self.luxury_brands = luxury_brands or []
        self.premium_brands = premium_brands or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Clean AskPrice & kmDriven if present
        if 'AskPrice' in df.columns:
            df['AskPrice'] = (df['AskPrice'].astype(str)
                              .str.replace('₹', '', regex=False)
                              .str.replace(',', '', regex=False)
                              .astype(float))
        if 'kmDriven' in df.columns:
            df['kmDriven'] = (df['kmDriven'].astype(str)
                              .str.replace(' km', '', regex=False)
                              .str.replace(',', '', regex=False)
                              .astype(float))
        # Ensure Age numeric
        if 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            df['Age'] = df['Age'].fillna(df['Age'].median())

        # Basic engineered features
        df['Age_Squared'] = df['Age'] ** 2
        df['Age_Cubed'] = df['Age'] ** 3
        df['kmDriven_Log'] = np.log1p(df['kmDriven'])
        df['kmDriven_Squared'] = df['kmDriven'] ** 2
        df['Avg_km_per_year'] = df['kmDriven'] / (df['Age'] + 1)
        df['Age_kmDriven_Interaction'] = df['Age'] * df['kmDriven']
        df['Age_km_Ratio'] = df['Age'] / (df['kmDriven'] + 1)
        df['Is_Luxury_Brand'] = df['Brand'].apply(lambda x: 1 if x in (self.luxury_brands or []) else 0)
        df['Is_Premium_Brand'] = df['Brand'].apply(lambda x: 1 if x in (self.premium_brands or []) else 0)

        if 'Owner' in df.columns:
            owner_map = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
            df['Owner_Numeric'] = df['Owner'].map(owner_map).fillna(5)

        if 'Transmission' in df.columns:
            df['Is_Manual'] = (df['Transmission'] == 'Manual').astype(int)
        if 'FuelType' in df.columns:
            df['Is_Diesel'] = (df['FuelType'] == 'Diesel').astype(int)
            df['Is_Petrol'] = (df['FuelType'] == 'Petrol').astype(int)

        df['Depreciation_Factor'] = df['Age'] * (1 + df['Is_Luxury_Brand'])
        df['Value_Score'] = df['kmDriven'] / (df['Age'] + 1)

        return df

# -------------------------
# Smoothed target encoder for Brand (no external dependency)
# -------------------------
class BrandTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10.0, min_samples_leaf=1):
        """
        smoothing: higher -> more global mean weight
        """
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.mapping_ = {}
        self.global_mean_ = None
        self.feature_name_in_ = 'Brand_TgtEnc'

    def fit(self, X, y):
        # X is a DataFrame or array with Brand column
        df = X.copy()
        # Expect y to be log-price (transformed)
        if isinstance(y, (pd.Series, np.ndarray)):
            y_series = pd.Series(y).reset_index(drop=True)
        else:
            y_series = pd.Series(y)

        # handle missing Brand
        brands = df['Brand'].fillna('__missing__')
        tmp = pd.DataFrame({'brand': brands, 'y': y_series})
        agg = tmp.groupby('brand')['y'].agg(['mean', 'count']).reset_index().rename(columns={'mean': 'brand_mean', 'count': 'brand_count'})
        self.global_mean_ = y_series.mean()
        # smoothing = 1 / (1 + exp(-(n - min_samples_leaf) / smoothing))
        # We'll implement the common smoothing formula:
        # enc = (count * mean + smoothing * global_mean) / (count + smoothing)
        agg['smooth'] = (agg['brand_count'] * agg['brand_mean'] + self.smoothing * self.global_mean_) / (agg['brand_count'] + self.smoothing)
        self.mapping_ = dict(zip(agg['brand'], agg['smooth']))
        # Unknown brands will map to global_mean_
        return self

    def transform(self, X):
        df = X.copy()
        brands = df['Brand'].fillna('__missing__')
        enc = brands.map(self.mapping_).fillna(self.global_mean_)
        df[self.feature_name_in_] = enc.values
        # we may want to drop original Brand afterwards in pipeline's ColumnTransformer remainder='drop'
        return df

# -------------------------
# Main
# -------------------------
def main():
    print("="*60)
    print("  START: TRAINING ENHANCED PIPELINE")
    print("="*60)

    # Load raw data
    df_raw = pd.read_csv(DATA_FILE)
    print(f"Initial dataset size: {len(df_raw)} rows")

    # Clean numeric columns
    df = df_raw.copy()
    df['AskPrice'] = (df['AskPrice'].astype(str)
                      .str.replace('₹', '', regex=False)
                      .str.replace(',', '', regex=False)
                      .astype(float))
    df['kmDriven'] = (df['kmDriven'].astype(str)
                      .str.replace(' km', '', regex=False)
                      .str.replace(',', '', regex=False)
                      .astype(float))
    df.dropna(subset=['AskPrice', 'kmDriven', 'Age'], inplace=True)

    # Drop columns we don't want
    for c in ['Year', 'PostedDate', 'AdditionInfo', 'model']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # Remove extreme outliers (safe)
    df = detect_outliers_iqr(df, 'AskPrice', multiplier=3.0)
    df = detect_outliers_iqr(df, 'kmDriven', multiplier=3.0)
    print(f"Cleaned dataset size: {len(df)} rows (removed {len(df_raw) - len(df)} rows)")

    # Target transform (log)
    df['LogAskPrice'] = np.log1p(df['AskPrice'])

    # Feature engineering (standalone to compute Brand encoding)
    fe = FeatureEngineer(luxury_brands=LUXURY_BRANDS, premium_brands=PREMIUM_BRANDS)
    df_fe = fe.transform(df)

    # Brand target encoding (fit on full training set below)
    # We'll split first (to avoid target leakage when encoding)
    price_bins = pd.qcut(df_fe['AskPrice'], q=10, duplicates='drop', labels=False)
    X_full = df_fe.drop(columns=['AskPrice', 'LogAskPrice'])
    y_full = df_fe['LogAskPrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.20, random_state=RANDOM_STATE, stratify=price_bins
    )

    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

    # Fit BrandTargetEncoder on X_train & y_train
    bte = BrandTargetEncoder(smoothing=15.0)
    bte.fit(X_train, y_train)

    # Apply bte to train and test
    X_train_b = bte.transform(X_train)
    X_test_b = bte.transform(X_test)

    # Choose features for numeric and categorical pipelines
    # Numeric features: all numeric columns except label
    numeric_cols = X_train_b.select_dtypes(include=[np.number]).columns.tolist()
    # Remove the label-like columns if present
    for forbidden in ['LogAskPrice', 'AskPrice']:
        if forbidden in numeric_cols:
            numeric_cols.remove(forbidden)
    # We will use BrandTargetEncoder's feature name instead of Brand string
    # Categorical features to one-hot encode: Transmission, Owner, FuelType (if present)
    categorical_candidates = ['Transmission', 'Owner', 'FuelType']
    categorical_cols = [c for c in categorical_candidates if c in X_train_b.columns]

    # Remove original 'Brand' from numeric cols if present
    if 'Brand' in numeric_cols:
        numeric_cols.remove('Brand')

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")

    # Column transformer: numerical scaling + one-hot for categorical
    numeric_pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])

    # OneHotEncoder: note sparse_output (scikit-learn >=1.2)
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    categorical_pipeline = Pipeline([
        ('onehot', onehot)
    ])

    # Build ColumnTransformer. We include the Brand encoded column (Brand_TgtEnc) as numeric
    # Ensure Brand_TgtEnc is present in numeric_cols; add if missing
    if 'Brand_TgtEnc' not in numeric_cols and 'Brand_TgtEnc' in X_train_b.columns:
        numeric_cols.append('Brand_TgtEnc')

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ], remainder='drop', sparse_threshold=0, verbose=False)

    # Build regressors: well-tuned-ish defaults
    rf = RandomForestRegressor(
        n_estimators=350, max_depth=18, min_samples_split=4,
        min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE
    )

    xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.7, tree_method='hist', random_state=RANDOM_STATE, n_jobs= -1, verbosity=0
    )

    final_gbdt = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE)

    # Stacking ensemble
    estimators = [
        ('rf', rf),
        ('xgb', xgb)
    ]

    stack = StackingRegressor(estimators=estimators, final_estimator=final_gbdt, n_jobs=-1, passthrough=False)

    # Full pipeline: FeatureEngineer (for new rows) -> BrandTargetEncoder applied separately earlier? We'll store mapping for inference.
    # For training, we'll construct the final training arrays from X_train_b (which already contains Brand_TgtEnc).
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', stack)
    ])

    # Fit pipeline on processed arrays
    # Prepare X_train_processed and X_test_processed (DataFrames already have Brand_TgtEnc)
    X_train_proc = X_train_b.copy()
    X_test_proc = X_test_b.copy()

    # For safety, keep only columns included in ColumnTransformer (numeric_cols + categorical_cols)
    # ColumnTransformer expects DataFrame columns reachable; it will select by name in lists.
    # Fit pipeline
    print("\n--- Training stacking ensemble pipeline (this may take a few minutes) ---")
    pipeline.fit(X_train_proc, y_train)

    # Save pipeline and Brand Target Encoding mapping and feature lists for inference
    artifacts = {
        'pipeline': pipeline,
        'brand_mapping': bte.mapping_,
        'brand_global_mean': bte.global_mean_,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'feature_order_info': {
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols
        }
    }
    joblib.dump(artifacts, PIPELINE_FILE, compress=3)
    print(f"\n✅ Pipeline + artifacts saved to '{PIPELINE_FILE}'")

    # Evaluate on test set
    print("\n--- Evaluating on test set ---")
    y_pred_log = pipeline.predict(X_test_proc)
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred_log)

    r2 = r2_score(y_test, y_pred_log)
    rmse = mean_squared_error(y_test_original, y_pred_original, squared=False)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
    accuracy = 100 - mape

    print("\n" + "="*60)
    print("      FINAL PIPELINE PERFORMANCE")
    print("="*60)
    print(f"✅ R² Score (log target):               {r2:.4f}")
    print(f"✅ RMSE (Original scale):               ₹ {rmse:,.2f}")
    print(f"✅ MAE (Original scale):                ₹ {mae:,.2f}")
    print(f"✅ MAPE (Original scale):               {mape:.2f}%")
    print(f"🎯 APPROX ACCURACY (100 - MAPE):         {accuracy:.2f}%")
    print("="*60)

    # Save evaluation plots
    try:
        plt.figure(figsize=(8,6))
        plt.scatter(y_test_original, y_pred_original, alpha=0.4, s=20)
        mx = max(y_test_original.max(), y_pred_original.max())
        plt.plot([0,mx],[0,mx], 'r--', linewidth=1)
        plt.xlabel('Actual Price (₹)')
        plt.ylabel('Predicted Price (₹)')
        plt.title(f'Actual vs Predicted (Accuracy ≈ {accuracy:.2f}%)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('final_pipeline_actual_vs_predicted.png', dpi=300)
        plt.close()
        print("✅ Saved plot 'final_pipeline_actual_vs_predicted.png'")
    except Exception as e:
        print("Warning: could not save plot:", e)

    print("\nTraining complete.")

if __name__ == '__main__':
    main()
