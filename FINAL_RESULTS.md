# FINAL RESULTS

## Dataset
- File: `cs-training.csv`
- Rows: 150,000
- Target: `SeriousDlqin2yrs`
- Positive/default rate: 6.68%
- Synthetic fallback: **No**

## Model performance — untouched test set

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| XGBoost Champion | 0.8685 | 0.4070 |\n| Random Forest | 0.8633 | 0.3865 |\n| Logistic Regression | 0.8623 | 0.3490 |\n
## Business threshold backtest

Threshold was selected using the validation split, not the test set.

- Selected threshold: **0.355**
- Approval rate: **95.55%**
- Baseline NPA: **6.68%**
- Approved portfolio NPA: **4.56%**
- NPA reduction: **31.73%**
- 85% ROC-AUC target: **PASS**
- 30% NPA-reduction target: **PASS**

## Engineering
The feature engineering adapts the earlier AntiGravity ideas to this dataset:
- utilization and debt-ratio transforms
- income/dependent ratios
- delinquency totals and weighted severity
- 60+/90+ delinquency flags
- high-utilization/high-debt flags
- credit-line/age ratios
- missingness indicators

## Important
These metrics are historical holdout backtests. They are not guarantees of future lending performance and should not be used as an automatic credit decision system without proper model validation, calibration, fairness testing, governance and regulatory review.
