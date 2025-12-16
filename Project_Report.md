Final Report: Credit Risk Probability Model for Alternative Data (BNPL)
1. Project Overview
This project builds an end-to-end credit risk probability model for a Buy-Now-Pay-Later (BNPL) context using alternative transaction data. The goal is to create a solution that is:

Predictive enough to be useful for risk decisioning
Interpretable and governable (aligned with Basel II style expectations)
Reproducible (consistent pipeline + tracking)
Deployable (API + containerization + CI)
The delivered system includes:

Data exploration and insight generation (EDA)
A proxy target engineering approach to label “high risk” customers where true defaults are not available
A model training pipeline using classic, interpretable baselines (Logistic Regression, Random Forest)
MLflow tracking + model registry integration
A FastAPI service that loads the production model from MLflow and exposes prediction endpoints
Docker + Docker Compose for deployment
GitHub Actions CI for linting and tests
2. Business Understanding & Objectives (Why we built it)
In regulated credit decisioning, success is not just “high accuracy”—it’s also auditable, explainable, and stable.

2.1 Basel II / Governance Motivation
The project follows the principle that a credit risk model must be:

Transparent: explain what drives risk up/down
Auditable: reproducible training steps and decisions
Governable: clear documentation, change control, monitoring
2.2 Why a Proxy Target is Necessary
In many alternative-data environments, we don’t observe contractual default (e.g., 90+ DPD). Therefore:

We engineered a proxy label to represent risk based on observed behavior signals.
This was explicitly treated as a business risk: proxy misalignment can cause bias or poor decisions, so it must be documented and monitored.
2.3 Model Trade-offs
The project leans toward models that are easier to govern:

Logistic Regression (preferred interpretability)
Optional mention of WoE/scorecard style alignment (traditional, explainable credit modeling)
Random Forest included as a stronger nonlinear baseline (but less transparent)
3. Data Understanding
3.1 Dataset Used
The project uses a transaction-level dataset located under:

data/raw/data.csv
Loaded shape observed in EDA:

95,662 rows
16 columns
Key fields include:

IDs: TransactionId, BatchId, CustomerId, AccountId, SubscriptionId
Transaction attributes: Amount, Value, ProductCategory, ProviderId, ChannelId, PricingStrategy
Time: TransactionStartTime
Original label: FraudResult (highly imbalanced)
4. Exploratory Data Analysis (EDA) – What we found
Notebook:

notebooks/eda.ipynb
4.1 Key Findings / Insights
Severe class imbalance (FraudResult)
Mean of FraudResult ≈ 0.0020 (about 0.2% positives)
Implication: accuracy is misleading; need PR-AUC / F1 / recall/precision trade-offs, class weighting, etc.
Feature redundancy: Amount vs Value
Correlation between Amount and Value ≈ 0.98–0.99
Implication: keep one / engineer robust transformations to avoid multicollinearity and simplify models.
Outliers are common in transaction values
IQR outlier fraction: Amount ≈ 25%, Value ≈ 9.6%
Implication: consider log scaling, robust preprocessing, or outlier flags.
“Constant” columns
CountryCode and CurrencyCode effectively constant in this dataset
Implication: drop in modeling (no predictive signal)
High-cardinality identifiers
TransactionId, BatchId are near-unique identifiers
Implication: not usable directly; use CustomerId for aggregation/behavioral feature engineering
5. Proxy Target Engineering (Customer Risk Label Creation)
Notebook:

notebooks/04_proxy_target_variable_engineering.ipynb
Because a clean default label is unavailable, the project creates a customer-level proxy risk label using RFM + clustering:

5.1 RFM Features Built
Aggregated by CustomerId:

Recency: days since last transaction (relative to snapshot date)
Frequency: number of transactions
Monetary: total Value sum of transactions
5.2 Clustering to Create Risk Groups
Standardization via StandardScaler
Clustering via KMeans(n_clusters=3, random_state=42, n_init=10)
5.3 Proxy Label Definition
Identify “high risk cluster” as the cluster with the highest average recency (customers who haven’t transacted recently)
Create binary label:
is_high_risk = 1 if in high-risk cluster, else 0
Class distribution created:

is_high_risk = 1: 1426
is_high_risk = 0: 2316
5.4 Output Artifact Produced
Saved processed dataset:

data/processed/customer_risk_profiles.csv
This file becomes the training dataset for modeling.

6. Model Training & Experiment Tracking (MLflow)
Notebook:

notebooks/05_model_training_and_tracking.ipynb
6.1 Training Setup
Features: Recency, Frequency, Monetary
Target: is_high_risk
Split: train_test_split(..., test_size=0.2, stratify=y, random_state=42)
Observed split sizes:

Train: 2993 rows
Test: 749 rows
6.2 Models Trained (Baseline + Comparator)
Logistic Regression
Random Forest Classifier
Uses typical classification metrics:
accuracy, precision, recall, f1, roc_auc
6.3 MLflow Usage
MLflow is used to:
Track experiments (params, metrics)
Store artifacts/models
Register a best model to the Model Registry
A key integration detail used later in the API:

Model registered under:
MODEL_NAME = "CreditRiskModel"
loaded from stage:
MODEL_STAGE = "production"
7. Serving the Model (FastAPI)
Files:

src/api/main.py
src/api/pydantic_models.py
7.1 API Features
The FastAPI app exposes:

GET /health
Returns:
status: "ok"
model_loaded: true/false
Used for readiness checks
POST /predict
Input schema (PredictionRequest):
Recency: float
Frequency: float
Monetary: float
Output schema (PredictionResponse):
risk_probability: float
7.2 Model Loading Strategy (Production-Ready)
The API loads the model directly from MLflow registry:

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.sklearn.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
If the model fails to load:

API returns HTTP 503 on predict:
"Model not available. Please check MLflow setup."
This is important operationally: the service degrades safely instead of returning wrong predictions.

8. Containerization & Deployment
Files:

Dockerfile
docker-compose.yml
8.1 Dockerfile
Base: python:3.11-slim
Installs dependencies from requirements.txt
Copies src/
Runs:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Exposes port 8000
8.2 Docker Compose
Service: api
Builds from project root
Maps:
"8000:8000"
This allows easy local deployment with one command (compose up) and aligns with typical production container workflows.

9. CI/CD (Quality & Automation)
File:

.github/workflows/ci.yml
The pipeline runs on:

Push to main
Pull requests targeting main
Steps:

Checkout code
Setup Python 3.11
Install dependencies
Lint with flake8
Test with pytest
This ensures:

Code quality checks are automated
Tests run before merge/deploy
Regression risk is reduced
10. Dependencies / Environment
File:

requirements.txt
Key libraries involved:

pandas, numpy, scikit-learn for data and ML
mlflow for tracking and model registry
fastapi, uvicorn for serving
pytest for testing
flake8 and black included for code quality tooling
xverse==1.0.5 used (WoE tooling support; also aligns with credit scorecard approaches)
11. What We Achieved (Deliverables & Outcomes)
11.1 Technical Deliverables
EDA pipeline with documented insights and implications
Customer-level modeling dataset
customer_risk_profiles.csv with RFM + proxy target
Proxy label engineering method
RFM + KMeans to generate is_high_risk
Model training and evaluation workflow
Train/test split, metrics, and experiment tracking
MLflow model registry integration
A production stage model (CreditRiskModel, stage production)
Production-style model serving API
Typed request/response, health endpoint, error handling
Deployment readiness
Docker + Compose
CI/CD
GitHub Actions lint + tests
11.2 Business Outcomes
A workable approach for risk scoring using alternative data
A process aligned with model governance expectations:
documentation, reproducibility, traceability (MLflow), operational endpoints, CI checks
12. Limitations / Risks (Important for a real-world report)
Proxy label risk:
is_high_risk is behavior-derived, not contractual default.
Requires monitoring and validation against real repayment/default outcomes when available.
Feature simplicity:
Current API expects only Recency/Frequency/Monetary.
Real BNPL scoring usually benefits from richer behavioral/merchant/network/device features.
Imbalance in original fraud label:
FraudResult is extremely imbalanced and may not reflect default risk.
Operational dependency:
API availability depends on MLflow tracking server being reachable and model staged correctly.
13. Recommended Next Steps (If extending the project)
Improve proxy target definition:
Try alternative clustering, monotonic segmentation, or expert-defined rules
Validate stability over time
Add model monitoring:
data drift, score drift, calibration monitoring
Expand features:
rolling-window metrics, customer tenure, average basket size, channel mix, merchant/product patterns
Add tests for the API layer (contract tests) and pipeline components
Add a simple model card / governance checklist (reason codes, stability metrics, fairness checks)
Status / Completion Summary
Completed: A full end-to-end credit risk modeling project with proxy target creation, model training & tracking, production-style API serving, containerization, and CI/CD.
Output: This document is the requested Final Report describing what was done and what was achieved.
Feedback submitted