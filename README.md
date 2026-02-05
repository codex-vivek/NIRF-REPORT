# NIRF Rank Prediction & Improvement System (AI-Powered)

This is a fully AI and machine learning–based project developed using Python and supervised learning models. It is trained on historical NIRF data to predict rank ranges and uses feature importance techniques to identify weak performance areas. The project is deployable on cloud platforms and can be accessed from any system.

## Features
- **AI-Based Prediction**: Uses a Random Forest Regressor trained on synthetic NIRF data to estimate scores.
- **Rank Range Estimation**: Maps predicted scores to historical rank distributions.
- **Explainability (XAI)**: Uses SHAP (SHapley Additive exPlanations) to explain which features contributed most to the score.
- **Actionable Insights**: Automatically suggests high-impact improvements based on feature weights and gap analysis.
- **Premium UI**: Modern, glassmorphism-based dashboard.

## Architecture
- **Frontend**: React (Vite), Recharts, Framer Motion.
- **Backend**: FastAPI (Python).
- **ML Engine**: Scikit-Learn, SHAP, Pandas.

## Setup & Run

### Prerequisites
- Node.js
- Python 3.8+

### 1. Backend Setup
```bash
# Install dependencies
pip install -r backend/requirements.txt

# Generate Data & Train Model (Required for first run)
cd backend
python train_model.py

# Run Server (From project root)
uvicorn backend.app.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:5173`.
