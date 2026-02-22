export interface NIRFInputs {
  TLR: number;
  RPC: number;
  GO: number;
  OI: number;
  PR: number;
}

export interface PredictionResult {
  predicted_score: number;
  rank_range_min: number;
  rank_range_max: number;
  shap_values: Record<string, number>;
  recommendations: string[];
}

const WEIGHTS: Record<keyof NIRFInputs, number> = {
  TLR: 0.35,
  RPC: 0.30,
  GO: 0.15,
  OI: 0.10,
  PR: 0.10,
};

const BASELINES: Record<keyof NIRFInputs, number> = {
  TLR: 62.5, // Average of [30, 95]
  RPC: 47.5, // Average of [5, 90]
  GO: 69.0,  // Average of [40, 98]
  OI: 57.5,  // Average of [30, 85]
  PR: 50.0,  // Average of [5, 95]
};

export async function predictStatic(inputs: NIRFInputs): Promise<PredictionResult> {
  // 1. Calculate Predicted Score
  let score = 0;
  for (const key in WEIGHTS) {
    const k = key as keyof NIRFInputs;
    score += inputs[k] * WEIGHTS[k];
  }
  
  // Add some simulated base bias similar to the noise in training
  score = Math.min(100, Math.max(0, score + 2.5));

  // 2. Map to Rank Range (Simulated based on 1000 institution samples)
  // Scores typically range from ~20 to ~95
  const minExpected = 20;
  const maxExpected = 95;
  const totalSamples = 1000;
  
  // Normalized score (0 to 1) where 1 is best
  const normalized = (score - minExpected) / (maxExpected - minExpected);
  const estimatedRank = Math.max(1, Math.round(totalSamples * (1 - normalized)));
  
  const rankMargin = Math.max(5, Math.floor(estimatedRank * 0.12));
  const rankMin = Math.max(1, estimatedRank - rankMargin);
  const rankMax = estimatedRank + rankMargin;

  // 3. SHAP Impact Analysis (Simplified linear contribution)
  const shapValues: Record<string, number> = {};
  for (const key in WEIGHTS) {
    const k = key as keyof NIRFInputs;
    shapValues[k] = (inputs[k] - BASELINES[k]) * WEIGHTS[k];
  }

  // 4. Data-Driven Recommendations
  const recommendations: string[] = [];
  
  // Logic A: Negative impacts
  const sortedShap = Object.entries(shapValues).sort((a, b) => a[1] - b[1]);
  for (const [feat, val] of sortedShap) {
    if (val < 0) {
      recommendations.push(`AI Observation: ${feat} is currently below the benchmark, detracting ${Math.abs(val).toFixed(2)} pts from your score.`);
    }
  }

  // Logic B: Highest ROI
  const impacts: Record<string, number> = {};
  for (const key in WEIGHTS) {
    const k = key as keyof NIRFInputs;
    // Simulate what happens if we increase this feat by 10 units
    impacts[k] = 10 * WEIGHTS[k];
  }
  const bestFeat = Object.entries(impacts).sort((a, b) => b[1] - a[1])[0][0];
  recommendations.push(`ML Recommendation: Improvements in ${bestFeat} will yield the highest score boost based on model weights.`);

  // Logic C: Critical Area
  const topImportance = Object.entries(shapValues).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0][0];
  recommendations.push(`Critical Area: ${topImportance} currently has the most significant impact on your profile variance.`);

  return {
    predicted_score: score,
    rank_range_min: rankMin,
    rank_range_max: rankMax,
    shap_values: shapValues,
    recommendations: recommendations.slice(0, 5)
  };
}
