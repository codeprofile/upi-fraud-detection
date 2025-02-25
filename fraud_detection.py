import pandas as pd
import numpy as np
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import uvicorn

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()


# Define input schema for API requests
class TransactionInput(BaseModel):
    timestamp: str
    amount: float
    merchant_name: int
    mcc: int


# Step 1: Load Data
def load_data():
    """Load transaction and MCC data."""
    try:
        transactions = pd.read_parquet("transactions.parquet")
        mcc_data = pd.read_csv("mcc_codes.csv")
        logging.info("Data loaded successfully.")
        return transactions, mcc_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


# Step 2: Merge Data
def merge_data(transactions, mcc_data):
    """Merge transaction data with MCC descriptions."""
    try:
        # Merge on MCC code
        merged_data = pd.merge(transactions, mcc_data, left_on="MCC", right_on="mcc", how="left")
        logging.info("Data merged successfully.")
        return merged_data
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        raise


# Step 3: Data Preprocessing
def preprocess_data(df):
    """Clean and transform transaction data."""
    try:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(
            df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str) + ' ' + df['Time'])
        df['transaction_hour'] = df['timestamp'].dt.hour
        df['transaction_day'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['transaction_day'].apply(lambda x: 1 if x >= 5 else 0)

        # Convert amount to numeric and handle missing values
        df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True).astype(float)
        df['Amount'].fillna(0, inplace=True)

        # Fill missing ZIP and MCC descriptions
        df['Zip'].fillna(0, inplace=True)
        df['combined_description'].fillna("Unknown", inplace=True)

        return df
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise


# Step 4: Feature Engineering
def feature_engineering(df):
    """Create meaningful features and encode categorical variables."""
    try:
        label_enc = LabelEncoder()
        df['merchant_encoded'] = label_enc.fit_transform(df['Merchant Name'].astype(str))
        df['mcc_encoded'] = label_enc.fit_transform(df['MCC'].astype(str))

        # Create features
        features = ['Amount', 'transaction_hour', 'is_weekend', 'merchant_encoded', 'mcc_encoded']

        # Encode 'Is Fraud?' column
        fraud_mapping = {'Yes': 1, 'No': 0}
        if 'Is Fraud?' in df.columns:
            df['Is Fraud?'] = df['Is Fraud?'].map(fraud_mapping).fillna(0).astype(int)
            target = df['Is Fraud?']
        else:
            target = pd.Series([0] * len(df))  # Default to non-fraudulent

        return df[features], target
    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        raise


# Step 5: Handle Imbalanced Data
def balance_data(X, y):
    """Balance the dataset using SMOTE."""
    try:
        smote = SMOTE(sampling_strategy=0.2, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logging.info("Data balanced using SMOTE.")
        return X_resampled, y_resampled
    except Exception as e:
        logging.error(f"Error in data balancing: {e}")
        raise


# Step 6: Train Models
def train_models(X_train, y_train):
    """Train fraud detection models and save them."""
    try:
        xgb = XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        iso_forest = IsolationForest(contamination=0.02, random_state=42)
        iso_forest.fit(X_train)

        # Save trained models
        joblib.dump(xgb, "xgb_model.pkl")
        joblib.dump(iso_forest, "isolation_forest.pkl")
        logging.info("Models trained and saved successfully.")
        return xgb, iso_forest
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise


# Load models at startup for API
try:
    xgb_model = joblib.load("xgb_model.pkl")
    iso_forest_model = joblib.load("isolation_forest.pkl")
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    xgb_model, iso_forest_model = None, None


# Step 7: API for Real-Time Fraud Detection
@app.post("/predict")
def predict(transaction: TransactionInput):
    """Predict if a transaction is fraudulent using trained models."""
    try:
        df = pd.DataFrame([transaction.dict()])
        df = preprocess_data(df)
        X, _ = feature_engineering(df)

        xgb_pred = xgb_model.predict(X)[0] if xgb_model else None
        iso_pred = iso_forest_model.predict(X)[0] if iso_forest_model else None

        return {"fraud_xgb": bool(xgb_pred), "fraud_isolation_forest": bool(iso_pred)}
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction error.")


# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    if not xgb_model and not iso_forest_model:
        transactions, mcc_data = load_data()
        merged_data = merge_data(transactions, mcc_data)
        processed_data = preprocess_data(merged_data)

        # Feature engineering and balancing
        X, y = feature_engineering(processed_data)
        X_resampled, y_resampled = balance_data(X, y)

        # Train-test split and model training
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        train_models(X_train, y_train)

    # # Run FastAPI server
    uvicorn.run(app, host="localhost", port=8000)
