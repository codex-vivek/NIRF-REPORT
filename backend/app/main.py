from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import shap 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NIRF Rank Predictor AI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "nirf_rank_predictor.pkl")
REF_PATH = os.path.join(BASE_DIR, "models", "reference_ranks.pkl")

# Global variables for model
model = None
reference_ranks = None
explainer = None

class NIRFInput(BaseModel):
    TLR: float
    RPC: float
    GO: float
    OI: float
    PR: float

class PredictionOutput(BaseModel):
    predicted_score: float
    rank_range_min: int
    rank_range_max: int
    shap_values: dict
    recommendations: list

def load_models():
    global model, reference_ranks, explainer
    if os.path.exists(MODEL_PATH) and os.path.exists(REF_PATH):
        model = joblib.load(MODEL_PATH)
        reference_ranks = joblib.load(REF_PATH)
        # Initialize SHAP explainer
        # For efficiency, we might just use the model's feature_importances_ for global, 
        # but for local (instance) explanation, TreeExplainer is best.
        explainer = shap.TreeExplainer(model)
        print("Models loaded successfully.")
    else:
        print("Models not found. Please train them first.")

@app.on_event("startup")
def startup_event():
    load_models()

@app.post("/predict", response_model=PredictionOutput)
def predict_rank(data: NIRFInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_df = pd.DataFrame([data.dict()])
    
    # 1. Predict Score
    pred_score = model.predict(input_df)[0]
    
    # 2. Map to Rank Range
    preds = reference_ranks['final_score'].values
    ranks = reference_ranks['Rank'].values
    
    # Find approx rank
    idx = np.searchsorted(-preds, -pred_score) 
    estimated_rank = idx + 1
    
    rank_margin = max(5, int(estimated_rank * 0.15))
    rank_min = max(1, estimated_rank - rank_margin)
    rank_max = estimated_rank + rank_margin
    
    # 3. Explainability (SHAP)
    shap_vals = explainer.shap_values(input_df)
    shap_dict = dict(zip(input_df.columns, shap_vals[0]))
    
    # 4. Data-Driven Recommendations
    recommendations = []
    
    # Logic A: Identify features pulling the score down (negative SHAP)
    sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1])
    for feat, val in sorted_shap:
        if val < 0:
            recommendations.append(f"AI Observation: {feat} is currently below the benchmark for your target range, detracting {abs(val):.2f} pts from your score.")
    
    # Logic B: Simulated Impact (Purely ML based)
    impacts = {}
    for feat in input_df.columns:
        simulated_input = input_df.copy()
        simulated_input[feat] = min(100, simulated_input[feat].values[0] * 1.1)
        new_score = model.predict(simulated_input)[0]
        impacts[feat] = new_score - pred_score
        
    best_feat = max(impacts, key=impacts.get)
    if impacts[best_feat] > 0.1:
        recommendations.append(f"ML Recommendation: Increasing {best_feat} by 10% would yield the highest score boost (+{impacts[best_feat]:.2f} pts) based on learned patterns.")

    # Generic high level advice
    top_importance = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[0][0]
    recommendations.append(f"Critical Area: {top_importance} shows the highest volatility in your profile. Prioritize data quality and investment here.")

    return {
        "predicted_score": pred_score,
        "rank_range_min": rank_min,
        "rank_range_max": rank_max,
        "shap_values": shap_dict,
        "recommendations": recommendations[:5]
    }

@app.get("/")
def health_check():
    return {
        "status": "ok", 
        "service": "NIRF AI Predictor",
       
    }
