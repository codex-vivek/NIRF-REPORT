import React, { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, Cell, ReferenceLine 
} from 'recharts';
import { 
  AlertTriangle, TrendingUp, ArrowLeft, BrainCircuit, 
  ShieldCheck, Activity, BarChart3, Info
} from 'lucide-react';

const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
  ? "http://localhost:8000" 
  : "/api";

interface PredictionResult {
  predicted_score: number;
  rank_range_min: number;
  rank_range_max: number;
  shap_values: Record<string, number>;
  recommendations: string[];
}

function App() {
  const [view, setView] = useState<'entry' | 'report'>('entry');
  const [institutionName, setInstitutionName] = useState('');
  const [category, setCategory] = useState('University');
  
  const [inputs, setInputs] = useState({
    TLR: 60,
    RPC: 40,
    GO: 70,
    OI: 55,
    PR: 20
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const { data } = await axios.post(`${API_URL}/predict`, inputs);
      setResult(data);
      setView('report');
    } catch (err) {
      console.error(err);
      alert("Failed to fetch prediction. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const performanceData = result ? Object.keys(inputs).map(key => ({
    name: key,
    value: inputs[key as keyof typeof inputs],
  })) : [];

  const shapData = result ? Object.entries(result.shap_values).map(([name, impact]) => ({
    name,
    impact: Number(impact.toFixed(2))
  })).sort((a, b) => b.impact - a.impact) : [];

  return (
    <div className="container">
      <AnimatePresence mode="wait">
        {view === 'entry' ? (
          <motion.div
            key="entry"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          >
            <div className="flex flex-col items-center mb-4">
              <div className="ai-badge">
                <BrainCircuit size={14} />
                POWERED BY SUPERVISED MACHINE LEARNING
              </div>
              <div className="text-[10px] text-blue-400 font-bold tracking-[0.3em] mt-2 opacity-80 uppercase">
                CODE BY VIVEK AI
              </div>
            </div>
            <h1>NIRF Rank AI Predictor</h1>
            <p className="subtitle">
              Advanced predictive analytics for first-time applicants. Get data-driven insights based on historical institutional metrics.
            </p>

            <form onSubmit={handlePredict} className="glass-card">
              <h2><ShieldCheck size={20} className="text-blue-400" /> Basic Profile</h2>
              <div className="form-grid">
                <div className="input-group">
                  <label>Institution Name</label>
                  <input 
                    type="text" 
                    placeholder="e.g. Graphic Era Global University" 
                    value={institutionName}
                    onChange={(e) => setInstitutionName(e.target.value)}
                    required
                  />
                </div>
                <div className="input-group">
                  <label>Institutional Category</label>
                  <select value={category} onChange={(e) => setCategory(e.target.value)}>
                    <option>University</option>
                    <option>Engineering</option>
                    <option>Management</option>
                    <option>Pharmacy</option>
                    <option>Medical</option>
                  </select>
                </div>
              </div>

              <h2><Activity size={20} className="text-emerald-400" /> Metric Assessment (0-100)</h2>
              <div className="params-grid">
                {Object.entries(inputs).map(([key, val]) => (
                  <div key={key} className="input-group">
                    <label title={key}>
                      {key === 'TLR' && 'Teaching (TLR)'}
                      {key === 'RPC' && 'Research (RPC)'}
                      {key === 'GO' && 'Graduation (GO)'}
                      {key === 'OI' && 'Inclusivity (OI)'}
                      {key === 'PR' && 'Perception (PR)'}
                    </label>
                    <input 
                      type="number" 
                      min="0" 
                      max="100" 
                      value={val}
                      onChange={(e) => setInputs(prev => ({ ...prev, [key]: Number(e.target.value) }))}
                      required
                    />
                  </div>
                ))}
              </div>

              <button type="submit" className="btn-submit" disabled={loading}>
                {loading ? 'Analyzing Historical Data...' : 'Generate AI Rank Prediction Report'}
              </button>
            </form>

            <div className="mt-12 p-8 rounded-3xl bg-white/5 border border-white/10 text-center max-w-2xl mx-auto">
              <h3 className="text-gray-200 text-sm font-bold uppercase mb-3 tracking-[0.2em] flex items-center justify-center gap-2">
                <BarChart3 size={16} className="text-blue-400" /> ML Methodologies
              </h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                Our system utilizes supervised regression models trained on thousands of data points from previous NIRF cycles. 
                Instead of fixed rules, it identifies non-linear correlations between metrics to estimate your standing.
                Explainability is provided via SHAP (SHapley Additive exPlanations) values to identify specific performance drivers.
              </p>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="report"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            transition={{ duration: 0.5 }}
          >
            <button 
              onClick={() => setView('entry')} 
              className="mb-8 flex items-center gap-2 text-gray-400 hover:text-white transition-all border border-white/10 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 cursor-pointer"
            >
              <ArrowLeft size={18} /> Modify Parameters
            </button>
            
            <Header result={result} name={institutionName} />

            <div className="report-grid">
              <div className="flex flex-col gap-8">
                {/* Performance Chart */}
                <div className="report-card">
                  <div className="flex items-center justify-between mb-6">
                    <h2><Activity size={20} className="text-emerald-400" /> Metric Performance</h2>
                  </div>
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={performanceData}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis 
                          dataKey="name" 
                          stroke="#64748b" 
                          fontSize={11}
                          tickLine={false}
                          axisLine={false}
                        />
                        <YAxis 
                          stroke="#64748b" 
                          fontSize={11}
                          tickLine={false}
                          axisLine={false}
                          domain={[0, 100]}
                        />
                        <Tooltip 
                          contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', fontSize: '12px' }}
                          cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                        />
                        <Bar dataKey="value" barSize={40} radius={[6, 6, 0, 0]}>
                          {performanceData.map((entry, index) => (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={entry.value < 40 ? 'var(--accent-red)' : 'var(--accent-green)'} 
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="chart-footer">
                    Benchmarks are calculated relative to <span>historical NIRF percentiles</span>
                  </div>
                </div>

                {/* SHAP Impact Analysis */}
                <div className="report-card">
                  <div className="flex items-center justify-between mb-2">
                    <h2><BrainCircuit size={20} className="text-purple-400" /> AI Impact Analysis (SHAP)</h2>
                    <div title="This shows how much each parameter pushed the predicted score up or down relative to the average." className="text-gray-500 cursor-help">
                      <Info size={16} />
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 mb-6">Contribution of each metric to the final score prediction</p>
                  
                  <div className="chart-container" style={{ height: 200 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart 
                        layout="vertical" 
                        data={shapData}
                        margin={{ left: 20, right: 40 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis type="number" hide />
                        <YAxis 
                          type="category" 
                          dataKey="name" 
                          stroke="#64748b" 
                          fontSize={11}
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip 
                          contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', fontSize: '12px' }}
                          formatter={(value) => [`${value} pts impact`, 'Impact']}
                        />
                        <ReferenceLine x={0} stroke="#475569" strokeWidth={2} />
                        <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                          {shapData.map((entry, index) => (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={entry.impact > 0 ? '#3b82f6' : '#f43f5e'} 
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="report-card">
                <h2><TrendingUp size={20} className="text-blue-400" /> AI Recommendations</h2>
                <div style={{ marginTop: 10 }}>
                  {result?.recommendations.map((rec, i) => {
                    const isMLRec = rec.includes('ML Recommendation');
                    const isFocus = rec.includes('Critical');
                    return (
                      <div key={i} className={`recommendation-box ${isMLRec ? 'active' : ''}`}>
                        <div className="strategy-title">
                          {isMLRec ? <TrendingUp size={16} /> : isFocus ? <ShieldCheck size={16} /> : <AlertTriangle size={16} />}
                          {isMLRec ? "Predicted Highest ROI" : isFocus ? "Strategic Focus" : `Observation ${i + 1}`}
                        </div>
                        <p className="recommendation-text">{rec}</p>
                      </div>
                    );
                  })}
                  {(!result?.recommendations || result.recommendations.length === 0) && (
                    <p className="recommendation-text">Maintain current performance across all parameters.</p>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Header({ result, name }: { result: PredictionResult | null, name: string }) {
  return (
    <div className="report-header-card">
      <div className="text-[10px] text-blue-400 font-bold tracking-[0.4em] mb-4 opacity-70">
        CODE BY VIVEK AI
      </div>
      <div className="rank-title">Probable NIRF Rank Range</div>
      {result && (
        <>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rank-value"
          >
            #{result.rank_range_min} - #{result.rank_range_max}
          </motion.div>
          <div className="score-text">Estimated Overall Score: <span className="text-white font-bold">{result.predicted_score.toFixed(2)}</span></div>
          <div className="mt-4 text-xs text-gray-500 max-w-sm mx-auto">
            Predicted for <strong>{name || 'Selected Institution'}</strong> based on current supervised learning weights.
          </div>
        </>
      )}
    </div>
  );
}

export default App;
