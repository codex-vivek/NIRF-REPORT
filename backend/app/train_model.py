import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import shap

# Paths
DATA_PATH = "backend/data/nirf_synthetic_data.csv"
MODEL_PATH = "backend/models/nirf_rank_predictor.pkl"
EXPLAINER_PATH = "backend/models/shap_explainer.pkl"

def train_model():
    if not os.path.exists(DATA_PATH):
        print("Data not found. Please generate data first.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Features and Target
    # We predict Final Score first, as it is more intrinsic to the institution than Rank (which is relative)
    # However, the user wants "Rank Prediction". 
    # Strategy: Predict Score -> Map to Rank Range.
    
    X = df[['TLR', 'RPC', 'GO', 'OI', 'PR']]
    y = df['final_score']
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model: Random Forest Regressor (Good for tabular, captures non-linearities, interpretable)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Trained. MAE: {mae:.2f}, R2: {r2:.2f}")
    
    # Explainability (SHAP) - Train explainer on a subset
    explainer = shap.Explainer(model, X_train)
    # We won't save the full explainer object if it's too huge, but for this size it's fine.
    # Actually, TreeExplainer is better for RF
    # explainer = shap.TreeExplainer(model) 
    
    # Save Model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    # joblib.dump(explainer, EXPLAINER_PATH) # Optional, can re-instantiate
    
    # We also need a way to map Score -> Rank.
    # We can save the sorted scores of the training set (or full set) as a reference.
    reference_scores = df[['final_score', 'Rank']].sort_values(by='final_score', ascending=False)
    joblib.dump(reference_scores, "backend/models/reference_ranks.pkl")
    
    print(f"Model and reference data saved to backend/models/")

if __name__ == "__main__":
    train_model()
