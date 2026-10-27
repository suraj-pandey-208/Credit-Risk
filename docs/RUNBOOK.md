# Runbook

### Local
`pip install -r requirements.txt`
`streamlit run streamlit_app.py`

### Retrain
`python src/train.py`

### Tests
`pytest -q`

### Docker
`docker compose up --build`

### Artifacts
- `champion_xgboost.joblib`: trained model
- `shap_explainer.joblib`: TreeSHAP explainer
- `model_metadata.json`: training metadata
- `model_comparison.csv`: model benchmark
- `business_metrics.json`: threshold backtest
