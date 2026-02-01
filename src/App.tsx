import React, { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { AlertTriangle, TrendingUp, ArrowLeft } from 'lucide-react';

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

  const chartData = result ? Object.keys(inputs).map(key => ({
    name: key,
    value: inputs[key as keyof typeof inputs],
  })) : [];

  return (
    <div className="container">
      <AnimatePresence mode="wait">
        {view === 'entry' ? (
          <motion.div
            key="entry"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <h1>Institution Data Entry</h1>
            <p className="subtitle">Enter your institution's estimated parameters to get an AI rank prediction.</p>

            <form onSubmit={handlePredict} className="form-card">
              <h2>Basic Information</h2>
              <div className="form-grid">
                <div className="input-group">
                  <label>Institution Name</label>
                  <input 
                    type="text" 
                    placeholder="e.g. Graphic Era University" 
                    value={institutionName}
                    onChange={(e) => setInstitutionName(e.target.value)}
                    required
                  />
                </div>
                <div className="input-group">
                  <label>Category</label>
                  <select value={category} onChange={(e) => setCategory(e.target.value)}>
                    <option>University</option>
                    <option>Engineering</option>
                    <option>Management</option>
                    <option>Pharmacy</option>
                  </select>
                </div>
              </div>

              <h2>NIRF Parameters (0-100)</h2>
              <div className="params-grid">
                {Object.entries(inputs).map(([key, val]) => (
                  <div key={key} className="input-group">
                    <label>
                      {key === 'TLR' && 'Teaching, Learning & Resources (TLR)'}
                      {key === 'RPC' && 'Research & Professional Practice (RPC)'}
                      {key === 'GO' && 'Graduation Outcomes (GO)'}
                      {key === 'OI' && 'Outreach & Inclusivity (OI)'}
                      {key === 'PR' && 'Peer Perception (PR)'}
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
                {loading ? 'Processing...' : 'Predict Rank & Generate Report'}
              </button>
            </form>

            <div className="mt-8 p-6 bg-blue-900/10 border border-blue-500/20 rounded-xl">
              <h3 className="text-blue-400 text-sm font-bold uppercase mb-2 tracking-widest">AI & ML Architecture</h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                This is a fully AI and machine learning–based project developed using Python and supervised learning models. 
                It is trained on historical NIRF data to predict rank ranges and uses feature importance techniques 
                to identify weak performance areas. The project is deployable on cloud platforms and can be accessed from any system.
              </p>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="report"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
          >
            <button 
              onClick={() => setView('entry')} 
              className="mb-6 flex items-center gap-2 text-gray-400 hover:text-white transition-colors border-none bg-transparent cursor-pointer"
            >
              <ArrowLeft size={18} /> Back to Data Entry
            </button>
            
            <h1 className="mb-2">{institutionName || 'Institution'} - NIRF Report</h1>
            <p className="subtitle">AI-Generated Performance Analysis</p>

            {result && (
              <div className="report-header-card">
                <div className="rank-title">Predicted Rank Range</div>
                <div className="rank-value">{result.rank_range_min} - {result.rank_range_max}</div>
                <div className="score-text">Overall Score: {result.predicted_score.toFixed(1)} / 100</div>
              </div>
            )}

            <div className="report-grid">
              <div className="report-card">
                <h2>Parameter Performance</h2>
                <div style={{ height: 250, width: '100%', marginTop: 20 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#374151" />
                      <XAxis 
                        dataKey="name" 
                        stroke="#9ca3af" 
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis 
                        stroke="#9ca3af" 
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                        domain={[0, 100]}
                      />
                      <Tooltip 
                        contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                      />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {chartData.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.value < 40 ? '#ef4444' : '#22c55e'} 
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="chart-footer">
                  Red bars indicate <span>critical weak areas</span> requiring immediate action
                </div>
              </div>

              <div className="report-card">
                <h2>AI Recommendations</h2>
                <div style={{ marginTop: 10 }}>
                  {result?.recommendations.map((rec, i) => {
                    const isHighImpact = rec.includes('High Impact');
                    return (
                      <div key={i} className={`recommendation-box ${!isHighImpact ? 'alert' : ''}`}>
                        <div className="strategy-title">
                          {isHighImpact ? <TrendingUp size={16} /> : <AlertTriangle size={16} />}
                          Strategy #{i + 1}
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

export default App;
