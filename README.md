# Credit Risk AI — Full End-to-End Project

**Required stack:** Python 3.11, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit, PostgreSQL, Evidently, MLflow, Docker, AWS ECS, Jupyter Notebook.

## Architecture

```text
                         ┌──────────────────────┐
                         │ Streamlit Frontend   │
                         │ Dashboard + Scoring  │
                         └──────────┬───────────┘
                                    │ HTTP/JSON
                                    ▼
                         ┌──────────────────────┐
                         │ FastAPI Backend      │
                         │ /predict /health     │
                         └───────┬───────┬──────┘
                                 │       │
                         ┌───────▼───┐   ▼
                         │ XGBoost   │ PostgreSQL
                         │ + SHAP    │ prediction logs
                         └───────────┘
                                 │
                       ┌─────────▼─────────┐
                       │ MLflow + Evidently│
                       │ tracking/monitor  │
                       └────────────────────┘
```

## Dataset
`data/raw/cs-training.csv` is the uploaded real dataset: 150,000 rows, target `SeriousDlqin2yrs`. No synthetic fallback.

## Actual model result
The supplied artifacts were trained on this dataset. See `FINAL_RESULTS.md` for exact holdout metrics. The reported champion is XGBoost with ROC-AUC 0.8685 and the validation-selected business threshold produced 31.73% NPA reduction on the untouched test set.

## Run with Docker — recommended

```bash
docker compose up --build
```

Open:
- Streamlit: http://localhost:8501
- FastAPI Swagger: http://localhost:8000/docs
- FastAPI health: http://localhost:8000/health
- MLflow: http://localhost:5000
- PostgreSQL: localhost:5432

Stop:
```bash
docker compose down
```

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Second terminal:
```bash
.venv\\Scripts\\activate
set API_URL=http://localhost:8000
streamlit run app/streamlit_app.py
```

## Jupyter Notebook

Open `notebooks/Credit_Risk_AI_End_to_End.ipynb`. It covers EDA, preprocessing/feature engineering, model evaluation, threshold analysis and SHAP.

## MLflow

Docker starts an MLflow server. To log the existing experiment:
```bash
python scripts/run_mlflow_tracking.py
```
Then view http://localhost:5000.

## Evidently / monitoring

```bash
python scripts/run_monitoring.py
```
The monitoring module attempts an Evidently data-drift report and provides a transparent PSI fallback if the installed Evidently API differs.

## Tests
```bash
pytest -q
```

## AWS ECS
See `deploy/aws/README.md` and the ECS task definition templates. Use ECR + ECS Fargate + RDS PostgreSQL + Secrets Manager + CloudWatch for deployment.

## Important governance note
The model metrics are historical holdout backtests. They are not guarantees for future lending performance. Before real lending use, validate calibration, fairness, stability, drift, adverse-action requirements, security and human oversight.
