import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=1000, output_path="backend/data/nirf_synthetic_data.csv"):
    np.random.seed(42)
    
    # NIRF Parameters (approximate max values based on typical scoring, though theoretically normalized to 100 total)
    # real NIRF uses sub-parameters, but we will use the 5 main heads for the high-level model as requested.
    # TLR (100), RPC (100), GO (100), OI (100), PR (100) -> Then weighted.
    # Weights: TLR (0.3), RPC (0.3), GO (0.2), OI (0.1), PR (0.1)
    
    data = {
        'TLR': np.random.uniform(30, 90, num_samples), # Teaching, Learning & Resources
        'RPC': np.random.uniform(10, 80, num_samples), # Research and Professional Practice (usually harder)
        'GO': np.random.uniform(40, 95, num_samples),  # Graduation Outcomes
        'OI': np.random.uniform(30.0, 85.0, num_samples), # Outreach and Inclusivity
        'PR': np.random.uniform(10, 90, num_samples),  # Peer Perception
    }
    
    df = pd.DataFrame(data)
    
    # Calculate Weighted Score (The Formula)
    # In reality, this is known, but we want the ML to "learn" the pattern and also handle non-linearities or noise
    # simulation of "hidden" complexity.
    # We add some noise to make it a machine learning problem not just a calculator.
    
    df['calculated_score'] = (
        df['TLR'] * 0.3 +
        df['RPC'] * 0.3 +
        df['GO'] * 0.2 +
        df['OI'] * 0.1 +
        df['PR'] * 0.1
    )
    
    # Add noise
    df['noise'] = np.random.normal(0, 1.5, num_samples)
    df['final_score'] = df['calculated_score'] + df['noise']
    df['final_score'] = df['final_score'].clip(0, 100)
    
    # Assign Ranks based on final_score (Descending)
    df = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data generated at {output_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
