import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from train_model import train_playstore_model

# Config
RAW_FILE = "play_store_reviews.csv"
CLEANED_FILE = "cleaned_playstore_reviews.csv"
MODEL_FILE = "playstore_review_model.joblib"
PREDICTIONS_FILE = "predicted_engagement.csv"

# Load dataset
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df

# Clean and feature engineer data
def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates(subset=["reviewId"], keep="last").reset_index(drop=True)
    df["content"] = df["content"].fillna("").astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["thumbsUpCount"] = pd.to_numeric(df["thumbsUpCount"], errors="coerce").fillna(0)
    df["reviewCreatedVersion"] = df["reviewCreatedVersion"].fillna("unknown")

    # Derived features
    df["review_length"] = df["content"].apply(len)
    df["has_reply"] = df["replyContent"].notnull().astype(int)
    df["engagement_score"] = df["thumbsUpCount"] + (df["has_reply"] * 5)
    return df

# Prepare features and labels
def prepare_features(df):
    X = df[["score", "thumbsUpCount", "review_length", "has_reply", "engagement_score"]]
    y = (df["score"] >= 4).astype(int)  # 1 = positive, 0 = neutral/negative
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def main():
    df_raw = load_data(RAW_FILE)
    df_cleaned = clean_data(df_raw)
    df_cleaned.to_csv(CLEANED_FILE, index=False)
    print(f"Saved cleaned dataset to {CLEANED_FILE}")

    X, y, scaler = prepare_features(df_cleaned)
    model = train_playstore_model(X, y)
    joblib.dump(model, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

    # Generate predictions
    y_pred = model.predict(X)
    df_cleaned["Predicted_Positive"] = y_pred
    df_cleaned.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Saved predictions to {PREDICTIONS_FILE}")

if __name__ == "__main__":
    main()
