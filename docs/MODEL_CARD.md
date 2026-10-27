# Model Card

## Intended use
Educational/portfolio demonstration of binary credit-risk prediction.

## Model
XGBoost classifier with engineered application-time features.

## Evaluation
A stratified 80/20 holdout is used, with an additional validation split inside training for threshold selection. The final reported business metrics are calculated on the untouched test set.

## Limitations
- Historical dataset; not evidence of current lender performance.
- Dataset may contain sampling/selection effects.
- AUC does not establish calibration or fairness.
- SHAP is explanatory, not causal.
- Real deployment requires calibration, subgroup/fairness testing, drift monitoring, governance, security and regulatory review.
