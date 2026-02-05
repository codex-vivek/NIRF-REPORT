import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def generate_synthetic_data(n_samples=500):
    np.random.seed(42)
    
    # NIRF parameters usually range from 0 to 100
    # TLR: Teaching, Learning & Resources (Weight ~0.3)
    # RPC: Research and Professional Practice (Weight ~0.3)
    # GO: Graduation Outcomes (Weight ~0.2)
    # OI: Outreach and Inclusivity (Weight ~0.1)
    # PR: Peer Perception (Weight ~0.1)
    
    tlr = np.random.uniform(30, 95, n_samples)
    rpc = np.random.uniform(5, 90, n_samples)
    go = np.random.uniform(40, 98, n_samples)
    oi = np.random.uniform(30, 85, n_samples)
    pr = np.random.uniform(5, 95, n_samples)
    
    # Simple linear combination with some noise to simulate a complex system
    # In reality, NIRF ranking is a weighted sum, but we train a model to "learn" this
    score = (0.35 * tlr + 0.30 * rpc + 0.15 * go + 0.10 * oi + 0.10 * pr) 
    # Add noise
    score += np.random.normal(0, 2, n_samples)
    score = np.clip(score, 0, 100)
    
    df = pd.DataFrame({
        'TLR': tlr,
        'RPC': rpc,
        'GO': go,
        'OI': oi,
        'PR': pr,
        'final_score': score
    })
    
    # Sort by score to assign ranks
    df = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    
    return df

def train():
    print("Generating synthetic NIRF data...")
    df = generate_synthetic_data(1000)
    
    X = df[['TLR', 'RPC', 'GO', 'OI', 'PR']]
    y = df['final_score']
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model
    model_dir = os.path.join("app", "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "nirf_rank_predictor.pkl")
    ref_path = os.path.join(model_dir, "reference_ranks.pkl")
    
    joblib.dump(model, model_path)
    # Save the reference dataframe (scores and ranks) for range estimation
    joblib.dump(df[['final_score', 'Rank']], ref_path)
    
    print(f"Model saved to {model_path}")
    print(f"Reference data saved to {ref_path}")

if __name__ == "__main__":
    train()
