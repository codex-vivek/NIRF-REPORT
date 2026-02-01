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
    # Find where this score would fit in the reference distribution
    # We look for rows with score close to pred_score
    # Logic: Find ranks of scores within +/- 2 points (approx confidence)
    
    # Simple logic: Find the index in sorted reference
    # Check bounds
    preds = reference_ranks['final_score'].values
    ranks = reference_ranks['Rank'].values
    
    # Ideally, we find the index where pred_score would be inserted
    idx = np.searchsorted(-preds, -pred_score) # - for descending order
    # idx is the 0-based rank index. So Rank is approx idx + 1.
    estimated_rank = idx + 1
    
    # Define range (e.g., +/- 5 ranks or based on score density)
    # Heuristic: +/- 5% of the rank
    rank_margin = max(5, int(estimated_rank * 0.1))
    rank_min = max(1, estimated_rank - rank_margin)
    rank_max = estimated_rank + rank_margin
    
    # 3. Explainability (SHAP)
    shap_values = explainer.shap_values(input_df)
    # shap_values is a list or array. For regression, it's (1, num_features)
    shap_dict = dict(zip(input_df.columns, shap_values[0]))
    
    # 4. Recommendations
    # Identify weakest parameter (relative to max possible or contribution)
    # Simple logic: "Increase [Weakest Feature] to improve."
    # Advanced: Simulate +X% increase and see impact.
    
    recommendations = []
    
    # Check for low values
    params = data.dict()
    for key, val in params.items():
        if val < 50: # Threshold
            recommendations.append(f"Focus on improving {key} (Current: {val}). It contributes significantly to the gap.")
            
    # Impact Analysis based on Feature Importance
    # Logic: Which feature has the highest potential for growth?
    sorted_importances = sorted(shap_dict.items(), key=lambda x: x[1])
    # Features dragging the score down will have negative SHAP (relative to mean) or lower positive.
    # Actually SHAP matches deviation from mean.
    
    # Let's suggest improving the metrics where SHAP is lowest (pulling down) OR 
    # where the absolute value is low?
    # Better: Identify the param with the most "headroom" (Max - Current) * Weight
    
    max_vals = {'TLR': 100, 'RPC': 100, 'GO': 100, 'OI': 100, 'PR': 100}
    weights = {'TLR': 0.3, 'RPC': 0.3, 'GO': 0.2, 'OI': 0.1, 'PR': 0.1}
    
    potential_gain = {}
    for k in params:
        headroom = max_vals[k] - params[k]
        gain = headroom * weights[k]
        potential_gain[k] = gain
        
    best_roi = max(potential_gain, key=potential_gain.get)
    if potential_gain[best_roi] > 1:
        recommendations.append(f"High Impact Action: Improving '{best_roi}' offers the best ROI for rank improvement due to its weight and current gap.")

    return {
        "predicted_score": pred_score,
        "rank_range_min": rank_min,
        "rank_range_max": rank_max,
        "shap_values": shap_dict,
        "recommendations": recommendations
    }

@app.get("/")
def health_check():
    return {"status": "ok", "service": "NIRF AI Predictor"}
