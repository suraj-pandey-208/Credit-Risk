# import os
# import json
# import requests
# import pandas as pd
# import streamlit as st
# import plotly.express as px



# API_URL = os.getenv(
#     "API_URL",
#     "http://localhost:8000"
# )

# ROOT = os.path.dirname(
#     os.path.dirname(
#         os.path.abspath(__file__)
#     )
# )

# ART = os.path.join(
#     ROOT,
#     "artifacts"
# )



# st.set_page_config(
#     page_title="Credit Risk AI",
#     page_icon="💳",
#     layout="wide"
# )




# st.markdown(
#     """
#     <style>

#     .block-container {
#         max-width: 1450px;
#         padding-top: 1.5rem;
#     }

#     .hero {
#         padding: 1.5rem 1.7rem;
#         border-radius: 20px;
#         background: linear-gradient(
#             120deg,
#             #111827,
#             #334155
#         );
#         color: white;
#         margin-bottom: 1.2rem;
#     }

#     .hero h1 {
#         margin-bottom: 0.3rem;
#     }

#     .hero p {
#         margin-bottom: 0;
#         color: #d1d5db;
#     }

#     .info-box {
#         padding: 1rem 1.2rem;
#         border-radius: 12px;
#         background: rgba(59,130,246,0.08);
#         border: 1px solid rgba(59,130,246,0.2);
#         margin: 0.8rem 0;
#     }

#     .success-box {
#         padding: 1rem 1.2rem;
#         border-radius: 12px;
#         background: rgba(34,197,94,0.08);
#         border: 1px solid rgba(34,197,94,0.2);
#         margin: 0.8rem 0;
#     }

#     .warning-box {
#         padding: 1rem 1.2rem;
#         border-radius: 12px;
#         background: rgba(245,158,11,0.08);
#         border: 1px solid rgba(245,158,11,0.2);
#         margin: 0.8rem 0;
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # ============================================================
# # HEADER
# # ============================================================

# st.markdown(
#     """
#     <div class="hero">
#         <h1>💳 Credit Risk AI</h1>
#         <p>
#             AI-powered credit default prediction using
#             XGBoost, SHAP, FastAPI and Streamlit
#         </p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # ============================================================
# # CONNECT TO FASTAPI
# # ============================================================

# try:

#     response = requests.get(
#         f"{API_URL}/model-info",
#         timeout=5
#     )

#     response.raise_for_status()

#     info = response.json()

#     meta = info["metadata"]

#     business = info["business_metrics"]

# except Exception as e:

#     st.error(
#         f"""
#         FastAPI backend unavailable.

#         API URL: {API_URL}

#         Error: {e}

#         Start the backend using:

#         python -m uvicorn api.main:app --reload --port 8000
#         """
#     )

#     st.stop()


# # ============================================================
# # HEALTH CHECK
# # ============================================================

# try:

#     health_response = requests.get(
#         f"{API_URL}/health",
#         timeout=5
#     )

#     health_response.raise_for_status()

#     health = health_response.json()

#     st.caption(
#         f"API: {health.get('status', 'unknown')}  •  "
#         f"Database: {health.get('database', 'unknown')}  •  "
#         f"Model: {meta.get('champion', 'XGBoost')}"
#     )

# except Exception as e:

#     st.warning(
#         f"Health check failed: {e}"
#     )


# # ============================================================
# # TABS
# # ============================================================

# tabs = st.tabs(
#     [
#         "📊 Dashboard",
#         "🔎 Applicant Scoring",
#         "🧠 Explainability",
#         "🗃️ Prediction History"
#     ]
# )


# # ============================================================
# # TAB 1 — DASHBOARD
# # ============================================================

# with tabs[0]:

#     st.subheader(
#         "Credit Risk Model Dashboard"
#     )

#     # --------------------------------------------------------
#     # LOAD METRICS
#     # --------------------------------------------------------

#     metrics_path = os.path.join(
#         ART,
#         "metrics",
#         "champion_metrics.json"
#     )

#     try:

#         with open(
#             metrics_path,
#             "r"
#         ) as f:

#             metrics = json.load(f)

#         roc_auc = metrics.get(
#             "roc_auc",
#             0
#         )

#         pr_auc = metrics.get(
#             "pr_auc",
#             0
#         )

#     except Exception:

#         roc_auc = 0
#         pr_auc = 0

#     npa_reduction = business.get(
#         "npa_reduction",
#         0
#     )

#     approval_rate = business.get(
#         "approval_rate",
#         0
#     )

#     threshold = business.get(
#         "threshold",
#         0
#     )

#     # --------------------------------------------------------
#     # METRIC CARDS
#     # --------------------------------------------------------

#     c1, c2, c3, c4 = st.columns(4)

#     c1.metric(
#         "Champion ROC-AUC",
#         f"{roc_auc:.4f}"
#     )

#     c2.metric(
#         "PR-AUC",
#         f"{pr_auc:.4f}"
#     )

#     c3.metric(
#         "NPA Reduction",
#         f"{npa_reduction * 100:.2f}%"
#     )

#     c4.metric(
#         "Approval Rate",
#         f"{approval_rate * 100:.2f}%"
#     )

#     # --------------------------------------------------------
#     # THRESHOLD
#     # --------------------------------------------------------

#     st.info(
#         f"""
#         **Policy threshold: {threshold:.3f}**

#         The model produces a default-risk probability.
#         The policy threshold is the cutoff used by the
#         lending decision policy to determine whether the
#         applicant falls inside or outside the approval
#         threshold.

#         The threshold was selected using validation data
#         and then evaluated on an untouched test set.
#         """
#     )

#     # --------------------------------------------------------
#     # MODEL PERFORMANCE SUMMARY
#     # --------------------------------------------------------

#     st.markdown(
#         "### 📈 What do these metrics mean?"
#     )

#     with st.expander(
#         "Understand ROC-AUC, PR-AUC and business metrics",
#         expanded=True
#     ):

#         st.markdown(
#             f"""
#             **ROC-AUC = {roc_auc:.4f}**

#             ROC-AUC measures how well the model distinguishes
#             between applicants who default and applicants who
#             do not default across different classification
#             thresholds.

#             - **1.0** → perfect separation
#             - **0.5** → approximately random ranking
#             - Higher values indicate better separation

#             Your champion model has a ROC-AUC of
#             **{roc_auc:.4f}**.

#             ---

#             **PR-AUC = {pr_auc:.4f}**

#             Precision-Recall AUC focuses on the model's ability
#             to identify the positive/default class.

#             - **Precision:** Of the applicants predicted as
#               risky, how many were actually risky?
#             - **Recall:** Of the applicants who actually
#               defaulted, how many did the model identify?

#             PR-AUC is particularly useful when the default
#             class is relatively uncommon.

#             ---

#             **NPA Reduction = {npa_reduction * 100:.2f}%**

#             This is a business backtest metric showing the
#             reduction in the observed non-performing/default
#             rate under the evaluated approval policy compared
#             with the baseline used in the project.

#             ---

#             **Approval Rate = {approval_rate * 100:.2f}%**

#             This represents the proportion of applicants that
#             remain within the evaluated approval policy.
#             """
#         )


# # ============================================================
# # MODEL COMPARISON
# # ============================================================

#     st.markdown(
#         "### 🔬 Model Comparison"
#     )

#     comparison_path = os.path.join(
#         ART,
#         "metrics",
#         "model_comparison.csv"
#     )

#     if os.path.exists(
#         comparison_path
#     ):

#         try:

#             comp = pd.read_csv(
#                 comparison_path
#             )

#             fig = px.bar(
#                 comp,
#                 x="model",
#                 y="roc_auc",
#                 text_auto=".4f",
#                 title="Test ROC-AUC Comparison"
#             )

#             st.plotly_chart(
#                 fig,
#                 use_container_width=True
#             )

#             st.caption(
#                 "This chart compares the evaluated models "
#                 "using ROC-AUC on the test set."
#             )

#         except Exception as e:

#             st.warning(
#                 f"Could not load model comparison: {e}"
#             )


# # ============================================================
# # BUSINESS CURVE
# # ============================================================

#     threshold_curve = os.path.join(
#         ART,
#         "plots",
#         "threshold_business_curve.png"
#     )

#     if os.path.exists(
#         threshold_curve
#     ):

#         st.markdown(
#             "### 💼 Policy Threshold & Business Trade-off"
#         )

#         st.image(
#             threshold_curve,
#             caption=(
#                 "Validation-selected policy threshold "
#                 "and business trade-off"
#             ),
#             use_container_width=True
#         )

#         st.caption(
#             """
#             This plot shows how changing the decision
#             threshold affects the business trade-off.
#             The selected threshold is then evaluated on
#             the untouched test set.
#             """
#         )


# # ============================================================
# # TAB 2 — APPLICANT SCORING
# # ============================================================

# with tabs[1]:

#     st.subheader(
#         "🔎 Real-time Applicant Scoring"
#     )

#     st.markdown(
#         """
#         Enter applicant information below. The data is sent
#         to the FastAPI backend, where the trained XGBoost
#         model generates a default-risk probability and
#         SHAP provides an explanation for the prediction.
#         """
#     )

#     # --------------------------------------------------------
#     # FORM
#     # --------------------------------------------------------

#     with st.form(
#         "score"
#     ):

#         left, right = st.columns(2)

#         # ====================================================
#         # LEFT
#         # ====================================================

#         with left:

#             applicant_id = st.text_input(
#                 "Applicant ID",
#                 "APP-001"
#             )

#             util = st.number_input(
#                 "Revolving utilization",
#                 min_value=0.0,
#                 max_value=20.0,
#                 value=0.35,
#                 step=0.01
#             )

#             age = st.number_input(
#                 "Age",
#                 min_value=18,
#                 max_value=120,
#                 value=35
#             )

#             d30 = st.number_input(
#                 "30–59 days past due",
#                 min_value=0,
#                 max_value=50,
#                 value=0
#             )

#             debt = st.number_input(
#                 "Debt ratio",
#                 min_value=0.0,
#                 max_value=50.0,
#                 value=0.35,
#                 step=0.01
#             )

#             income = st.number_input(
#                 "Monthly income",
#                 min_value=0.0,
#                 max_value=1_000_000.0,
#                 value=5000.0,
#                 step=100.0
#             )

#         # ====================================================
#         # RIGHT
#         # ====================================================

#         with right:

#             open_lines = st.number_input(
#                 "Open credit lines + loans",
#                 min_value=0,
#                 max_value=200,
#                 value=8
#             )

#             d90 = st.number_input(
#                 "90+ days late",
#                 min_value=0,
#                 max_value=50,
#                 value=0
#             )

#             real = st.number_input(
#                 "Real-estate loans / lines",
#                 min_value=0,
#                 max_value=100,
#                 value=1
#             )

#             d60 = st.number_input(
#                 "60–89 days past due",
#                 min_value=0,
#                 max_value=50,
#                 value=0
#             )

#             deps = st.number_input(
#                 "Dependents",
#                 min_value=0.0,
#                 max_value=30.0,
#                 value=1.0,
#                 step=1.0
#             )

#         submitted = st.form_submit_button(
#             "🚀 Run Risk Assessment",
#             type="primary"
#         )


#     # ========================================================
#     # RUN PREDICTION
#     # ========================================================

#     if submitted:

#         payload = {

#             "applicant_id":
#                 applicant_id,

#             "RevolvingUtilizationOfUnsecuredLines":
#                 util,

#             "age":
#                 age,

#             "NumberOfTime30_59DaysPastDueNotWorse":
#                 d30,

#             "DebtRatio":
#                 debt,

#             "MonthlyIncome":
#                 income,

#             "NumberOfOpenCreditLinesAndLoans":
#                 open_lines,

#             "NumberOfTimes90DaysLate":
#                 d90,

#             "NumberRealEstateLoansOrLines":
#                 real,

#             "NumberOfTime60_89DaysPastDueNotWorse":
#                 d60,

#             "NumberOfDependents":
#                 deps
#         }

#         try:

#             r = requests.post(
#                 f"{API_URL}/predict",
#                 json=payload,
#                 timeout=30
#             )

#         except Exception as e:

#             st.error(
#                 f"Could not connect to FastAPI: {e}"
#             )

#             st.stop()


#         # ====================================================
#         # SUCCESS
#         # ====================================================

#         if r.ok:

#             result = r.json()

#             risk_probability = result.get(
#                 "risk_probability",
#                 0
#             )

#             risk_band = result.get(
#                 "risk_band",
#                 "Unknown"
#             )

#             policy_threshold = result.get(
#                 "policy_threshold",
#                 0
#             )

#             decision = result.get(
#                 "decision",
#                 "Unknown"
#             )


#             # ------------------------------------------------
#             # RESULT
#             # ------------------------------------------------

#             st.markdown(
#                 "### 📋 Prediction Result"
#             )

#             a, b, c = st.columns(3)

#             a.metric(
#                 "Default Risk",
#                 f"{risk_probability * 100:.2f}%"
#             )

#             b.metric(
#                 "Risk Band",
#                 risk_band
#             )

#             c.metric(
#                 "Policy Threshold",
#                 f"{policy_threshold:.3f}"
#             )


#             st.progress(
#                 min(
#                     max(
#                         risk_probability,
#                         0
#                     ),
#                     1
#                 ),
#                 text=(
#                     f"Risk score: "
#                     f"{risk_probability:.4f}"
#                 )
#             )


#             # ------------------------------------------------
#             # DECISION
#             # ------------------------------------------------

#             if risk_probability < policy_threshold:

#                 st.success(
#                     f"Decision: {decision}"
#                 )

#             else:

#                 st.warning(
#                     f"Decision: {decision}"
#                 )


#             # =================================================
#             # APPLICANT SHAP EXPLANATION
#             # =================================================

#             st.divider()

#             st.markdown(
#                 "## 🧠 Why did the model make this prediction?"
#             )

#             st.markdown(
#                 f"""
#                 The model estimated a default probability of
#                 **{risk_probability * 100:.2f}%** for this applicant.

#                 SHAP explains the individual prediction by
#                 showing how different features pushed the
#                 model's output **toward higher risk or lower
#                 risk**.

#                 **Important:** A SHAP value is a model
#                 contribution. It is **not a percentage change
#                 in default probability**.
#                 """
#             )


#             # ------------------------------------------------
#             # SHAP DRIVER DATA
#             # ------------------------------------------------

#             drivers = result.get(
#                 "top_model_drivers",
#                 []
#             )


#             if drivers:

#                 applicant_values = {

#                     "RevolvingUtilizationOfUnsecuredLines":
#                         util,

#                     "age":
#                         age,

#                     "NumberOfTime30_59DaysPastDueNotWorse":
#                         d30,

#                     "DebtRatio":
#                         debt,

#                     "MonthlyIncome":
#                         income,

#                     "NumberOfOpenCreditLinesAndLoans":
#                         open_lines,

#                     "NumberOfTimes90DaysLate":
#                         d90,

#                     "NumberRealEstateLoansOrLines":
#                         real,

#                     "NumberOfTime60_89DaysPastDueNotWorse":
#                         d60,

#                     "NumberOfDependents":
#                         deps
#                 }


#                 driver_rows = []


#                 for driver in drivers:

#                     feature = driver.get(
#                         "feature",
#                         "Unknown"
#                     )

#                     shap_value = float(
#                         driver.get(
#                             "shap_value",
#                             0
#                         )
#                     )

#                     actual_value = applicant_values.get(
#                         feature,
#                         "Engineered feature"
#                     )


#                     if shap_value > 0:

#                         direction = (
#                             "Increases default-risk direction"
#                         )

#                         explanation = (
#                             "This feature pushed the "
#                             "model prediction toward "
#                             "higher default risk."
#                         )

#                     elif shap_value < 0:

#                         direction = (
#                             "Decreases default-risk direction"
#                         )

#                         explanation = (
#                             "This feature pushed the "
#                             "model prediction toward "
#                             "lower default risk."
#                         )

#                     else:

#                         direction = "Neutral"

#                         explanation = (
#                             "This feature had little "
#                             "directional contribution."
#                         )


#                     driver_rows.append(
#                         {
#                             "Feature":
#                                 feature,

#                             "Applicant Value":
#                                 actual_value,

#                             "SHAP Value":
#                                 shap_value,

#                             "Direction":
#                                 direction,

#                             "Explanation":
#                                 explanation
#                         }
#                     )


#                 driver_df = pd.DataFrame(
#                     driver_rows
#                 )


#                 # =================================================
#                 # LOWER RISK
#                 # =================================================

#                 st.markdown(
#                     "### 🟢 Factors pushing risk lower"
#                 )

#                 negative_df = driver_df[
#                     driver_df["SHAP Value"] < 0
#                 ].copy()


#                 if negative_df.empty:

#                     st.info(
#                         "No negative SHAP contributions "
#                         "were returned."
#                     )

#                 else:

#                     negative_df = negative_df.sort_values(
#                         "SHAP Value"
#                     )


#                     for _, row in negative_df.iterrows():

#                         st.markdown(
#                             f"""
#                             **{row['Feature']}**

#                             Applicant value:
#                             `{row['Applicant Value']}`

#                             SHAP contribution:
#                             `{row['SHAP Value']:.4f}`

#                             **What it means:**  
#                             {row['Explanation']}
#                             """
#                         )

#                         st.divider()


#                 # =================================================
#                 # HIGHER RISK
#                 # =================================================

#                 st.markdown(
#                     "### 🔴 Factors pushing risk higher"
#                 )

#                 positive_df = driver_df[
#                     driver_df["SHAP Value"] > 0
#                 ].copy()


#                 if positive_df.empty:

#                     st.info(
#                         "No positive SHAP contributions "
#                         "were returned."
#                     )

#                 else:

#                     positive_df = positive_df.sort_values(
#                         "SHAP Value",
#                         ascending=False
#                     )


#                     for _, row in positive_df.iterrows():

#                         st.markdown(
#                             f"""
#                             **{row['Feature']}**

#                             Applicant value:
#                             `{row['Applicant Value']}`

#                             SHAP contribution:
#                             `+{row['SHAP Value']:.4f}`

#                             **What it means:**  
#                             {row['Explanation']}
#                             """
#                         )

#                         st.divider()


#                 # =================================================
#                 # SHAP TABLE
#                 # =================================================

#                 st.markdown(
#                     "### 📊 Detailed Applicant-level SHAP"
#                 )

#                 st.dataframe(
#                     driver_df,
#                     use_container_width=True,
#                     hide_index=True
#                 )


#                 # =================================================
#                 # SHAP CHART
#                 # =================================================

#                 chart_df = driver_df.copy()

#                 chart_df["absolute_shap"] = (
#                     chart_df["SHAP Value"].abs()
#                 )

#                 chart_df = chart_df.sort_values(
#                     "absolute_shap",
#                     ascending=True
#                 )


#                 fig = px.bar(
#                     chart_df,
#                     x="SHAP Value",
#                     y="Feature",
#                     orientation="h",
#                     text="SHAP Value",
#                     title=(
#                         "Applicant-level SHAP Contributions"
#                     )
#                 )

#                 fig.add_vline(
#                     x=0,
#                     line_width=1
#                 )

#                 st.plotly_chart(
#                     fig,
#                     use_container_width=True
#                 )


#                 # =================================================
#                 # SIMPLE OVERALL EXPLANATION
#                 # =================================================

#                 st.markdown(
#                     "### 📌 Simple explanation"
#                 )


#                 strongest_negative = None
#                 strongest_positive = None


#                 if not negative_df.empty:

#                     strongest_negative = (
#                         negative_df
#                         .iloc[
#                             negative_df[
#                                 "SHAP Value"
#                             ]
#                             .abs()
#                             .argmax()
#                         ]
#                     )


#                 if not positive_df.empty:

#                     strongest_positive = (
#                         positive_df
#                         .iloc[
#                             positive_df[
#                                 "SHAP Value"
#                             ]
#                             .abs()
#                             .argmax()
#                         ]
#                     )


#                 explanation_parts = []


#                 explanation_parts.append(
#                     f"The model predicted a "
#                     f"**{risk_probability * 100:.2f}%** "
#                     f"default probability."
#                 )


#                 if strongest_negative is not None:

#                     explanation_parts.append(
#                         f"The strongest lower-risk "
#                         f"contribution among the displayed "
#                         f"features was **"
#                         f"{strongest_negative['Feature']}** "
#                         f"with a SHAP value of "
#                         f"**{strongest_negative['SHAP Value']:.4f}**."
#                     )


#                 if strongest_positive is not None:

#                     explanation_parts.append(
#                         f"The strongest higher-risk "
#                         f"contribution among the displayed "
#                         f"features was **"
#                         f"{strongest_positive['Feature']}** "
#                         f"with a SHAP value of "
#                         f"**{strongest_positive['SHAP Value']:.4f}**."
#                     )


#                 st.info(
#                     " ".join(
#                         explanation_parts
#                     )
#                 )


#             else:

#                 st.info(
#                     "The API did not return applicant-level "
#                     "SHAP drivers."
#                 )


#             # ------------------------------------------------
#             # SHAP DISCLAIMER
#             # ------------------------------------------------

#             st.caption(
#                 """
#                 SHAP explains the relationship between the
#                 trained model's inputs and its prediction.
#                 It does not establish causation. Derived
#                 features may appear in the explanation when
#                 they are part of the model's feature pipeline.
#                 """
#             )


#         # ====================================================
#         # API ERROR
#         # ====================================================

#         else:

#             st.error(
#                 f"""
#                 Prediction API returned an error.

#                 Status code: {r.status_code}

#                 Response:

#                 {r.text}
#                 """
#             )


# # ============================================================
# # TAB 3 — MODEL EXPLAINABILITY
# # ============================================================

# with tabs[2]:

#     st.subheader(
#         "🧠 Model Explainability"
#     )


#     # ========================================================
#     # INTRODUCTION
#     # ========================================================

#     st.markdown(
#         """
#         This section explains **how the credit-risk model
#         should be interpreted**.

#         There are three different levels of explanation:

#         1. **ROC-AUC** → How well the model separates
#            default and non-default applicants overall.

#         2. **Precision-Recall** → How well the model identifies
#            the default class, especially when defaults are
#            less common.

#         3. **SHAP** → Which features generally influence the
#            model and, for an individual applicant, which
#            features pushed that particular prediction higher
#            or lower.
#         """
#     )


#     # ========================================================
#     # ROC CURVE
#     # ========================================================

#     st.markdown(
#         "## 1️⃣ ROC Curve"
#     )


#     st.markdown(
#         f"""
#         **ROC-AUC = {roc_auc:.4f}**

#         The ROC curve evaluates the model's ability to
#         distinguish applicants who default from applicants
#         who do not default while the classification threshold
#         changes.

#         **X-axis — False Positive Rate**

#         The proportion of non-default applicants incorrectly
#         classified as positive/risky.

#         **Y-axis — True Positive Rate**

#         The proportion of actual default applicants correctly
#         identified by the model.

#         **Dashed diagonal**

#         Represents approximately random ranking.

#         **Blue curve**

#         Represents the performance of the trained champion
#         model at different thresholds.

#         The AUC summarizes the ranking performance across
#         these thresholds.
#         """
#     )


#     roc_path = os.path.join(
#         ART,
#         "plots",
#         "roc_curve.png"
#     )


#     if os.path.exists(
#         roc_path
#     ):

#         st.image(
#             roc_path,
#             caption=(
#                 f"Champion ROC Curve — "
#                 f"AUC = {roc_auc:.3f}"
#             ),
#             use_container_width=True
#         )


#     # ========================================================
#     # ROC INTERPRETATION
#     # ========================================================

#     with st.expander(
#         "💡 How should I read this ROC curve?"
#     ):

#         st.markdown(
#             f"""
#             Imagine the model gives every applicant a
#             risk score.

#             If we change the cutoff used to classify someone
#             as risky, the number of correctly identified
#             defaults and incorrectly flagged safe applicants
#             also changes.

#             The ROC curve shows this trade-off.

#             The model's **ROC-AUC is {roc_auc:.4f}**.

#             AUC is a ranking/discrimination metric. It does
#             not mean that an individual applicant has an
#             {roc_auc * 100:.1f}% probability of default.
#             """
#         )


#     # ========================================================
#     # PR CURVE
#     # ========================================================

#     st.markdown(
#         "## 2️⃣ Precision-Recall Curve"
#     )


#     st.markdown(
#         f"""
#         **PR-AUC = {pr_auc:.4f}**

#         Credit default datasets often contain fewer default
#         cases than non-default cases. Precision-Recall analysis
#         therefore gives another useful view of performance.

#         **Precision**

#         Among applicants predicted as risky, how many were
#         actually defaults?

#         **Recall**

#         Among all applicants who actually defaulted, how many
#         did the model identify?

#         The Precision-Recall curve shows the trade-off between
#         these two quantities as the decision threshold changes.
#         """
#     )


#     pr_path = os.path.join(
#         ART,
#         "plots",
#         "pr_curve.png"
#     )


#     if os.path.exists(
#         pr_path
#     ):

#         st.image(
#             pr_path,
#             caption=(
#                 f"Champion Precision-Recall Curve — "
#                 f"PR-AUC = {pr_auc:.3f}"
#             ),
#             use_container_width=True
#         )


#     with st.expander(
#         "💡 Why is Precision-Recall important for credit risk?"
#     ):

#         st.markdown(
#             """
#             In a credit-risk problem, the default class may
#             be much smaller than the non-default class.

#             Accuracy alone can therefore be misleading.

#             Precision tells us how reliable the model's risky
#             predictions are.

#             Recall tells us how many of the actual risky
#             applicants the model manages to identify.

#             PR-AUC summarizes this precision-recall behaviour
#             across thresholds.
#             """
#         )


#     # ========================================================
#     # GLOBAL SHAP
#     # ========================================================

#     st.markdown(
#         "## 3️⃣ Global SHAP Feature Importance"
#     )


#     st.markdown(
#         """
#         Global SHAP explains **which features generally
#         influence the model the most across the evaluated
#         applicants**.

#         This is different from applicant-level SHAP.

#         - **Global SHAP:** What features generally matter?
#         - **Applicant SHAP:** Why did the model make this
#           particular prediction?
#         """
#     )


#     shap_path = os.path.join(
#         ART,
#         "metrics",
#         "shap_global_importance.csv"
#     )


#     if os.path.exists(
#         shap_path
#     ):

#         try:

#             imp = pd.read_csv(
#                 shap_path
#             )


#             imp = (
#                 imp
#                 .head(15)
#                 .sort_values(
#                     "mean_abs_shap"
#                 )
#             )


#             fig = px.bar(
#                 imp,
#                 x="mean_abs_shap",
#                 y="feature",
#                 orientation="h",
#                 title=(
#                     "Top Features by Mean Absolute SHAP Value"
#                 )
#             )


#             st.plotly_chart(
#                 fig,
#                 use_container_width=True
#             )


#             with st.expander(
#                 "💡 How should I read Global SHAP?"
#             ):

#                 st.markdown(
#                     """
#                     Mean absolute SHAP measures the average
#                     magnitude of a feature's contribution.

#                     A larger value means the feature generally
#                     has a larger influence on the model's output.

#                     Important:

#                     **Global SHAP importance does NOT tell us
#                     whether a feature increases or decreases
#                     risk.**

#                     It only tells us how strongly the feature
#                     influences predictions on average.

#                     The direction for one specific applicant
#                     is obtained from applicant-level SHAP.
#                     """
#                 )


#         except Exception as e:

#             st.warning(
#                 f"Could not load SHAP importance: {e}"
#             )

#     else:

#         st.warning(
#             "Global SHAP importance file not found."
#         )


#     # ========================================================
#     # FINAL EXPLANATION
#     # ========================================================

#     st.markdown(
#         "## 🔍 In simple words"
#     )


#     st.info(
#         """
#         **ROC-AUC answers:**
#         "How well does the model separate risky and
#         non-risky applicants overall?"

#         **PR-AUC answers:**
#         "How well does the model identify the relatively
#         uncommon default class?"

#         **Global SHAP answers:**
#         "Which features generally influence the model?"

#         **Applicant-level SHAP answers:**
#         "For this particular applicant, which features pushed
#         the prediction toward higher or lower risk?"
#         """
#     )


#     st.caption(
#         """
#         SHAP is a model-explanation method. It describes how
#         features contribute to the model output; it does not
#         by itself establish causation or constitute a complete
#         regulatory adverse-action explanation.
#         """
#     )


# # ============================================================
# # TAB 4 — PREDICTION HISTORY
# # ============================================================

# with tabs[3]:

#     st.subheader(
#         "🗃️ Recent PostgreSQL Prediction Events"
#     )


#     try:

#         r = requests.get(
#             f"{API_URL}/predictions?limit=100",
#             timeout=10
#         )


#         if r.ok:

#             items = r.json().get(
#                 "items",
#                 []
#             )


#             if items:

#                 st.dataframe(
#                     pd.DataFrame(items),
#                     use_container_width=True,
#                     hide_index=True
#                 )

#             else:

#                 st.info(
#                     "No prediction history available."
#                 )


#         else:

#             st.error(
#                 f"""
#                 Prediction history API error:

#                 {r.text}
#                 """
#             )


#     except Exception as e:

#         st.error(
#             f"""
#             Could not fetch prediction history:

#             {e}
#             """
#         )



import os
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv("API_URL", "http://localhost:8000")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")

st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #111827, #334155);
        color: white;
        margin-bottom: 1.2rem;
    }

    .hero p {
        color: #d1d5db;
        margin-bottom: 0;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# IMPORTANT: No HTML tags in the header itself
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div style="font-size: 2rem; font-weight: 700;">
            💳 Credit Risk AI
        </div>
        <p>
            AI-powered credit default prediction using
            XGBoost, SHAP, FastAPI and Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONNECT TO FASTAPI
# ============================================================

try:
    response = requests.get(f"{API_URL}/model-info", timeout=10)
    response.raise_for_status()

    info = response.json()
    meta = info.get("metadata", {})
    business = info.get("business_metrics", {})

except requests.RequestException as e:
    st.error("FastAPI backend is unavailable.")
    st.code(f"API URL: {API_URL}\nError: {e}")
    st.info(
        "Check that the API container is running and "
        "the API_URL environment variable is correct."
    )
    st.stop()

except (ValueError, AttributeError) as e:
    st.error(f"Could not read model information: {e}")
    st.stop()

# ============================================================
# HEALTH CHECK
# ============================================================

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    health_response.raise_for_status()
    health = health_response.json()

    api_status = health.get("status", "unknown")
    database_status = health.get("database", "unknown")
    model_name = meta.get("champion", "XGBoost")

    st.caption(
        f"API: {api_status}  •  "
        f"Database: {database_status}  •  "
        f"Model: {model_name}"
    )

except requests.RequestException as e:
    st.warning(f"Health check failed: {e}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(path, default=None):
    """Safely load a JSON file."""
    if default is None:
        default = {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return default


def load_csv(path):
    """Safely load a CSV file."""
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load {os.path.basename(path)}: {e}")

    return pd.DataFrame()


def display_image(path, caption):
    """Display an image only if it exists."""
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)


def fetch_predictions(limit=100):
    """Fetch prediction history from the backend."""
    try:
        response = requests.get(
            f"{API_URL}/predictions",
            params={"limit": limit},
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("items", [])
    except (requests.RequestException, ValueError) as e:
        st.error(f"Could not fetch prediction history: {e}")
        return None


# ============================================================
# TABS
# ============================================================

dashboard_tab, scoring_tab, explainability_tab = st.tabs(
    [
        "📊 Dashboard",
        "🔎 Applicant Scoring",
        "🧠 Explainability",
    ]
)

# ============================================================
# TAB 1: DASHBOARD
# ============================================================

with dashboard_tab:
    st.subheader("Credit Risk Model Dashboard")

    metrics_path = os.path.join(
        ART, "metrics", "champion_metrics.json"
    )
    metrics = load_json(metrics_path)

    roc_auc = float(metrics.get("roc_auc", 0) or 0)
    pr_auc = float(metrics.get("pr_auc", 0) or 0)

    npa_reduction = float(
        business.get("npa_reduction", 0) or 0
    )
    approval_rate = float(
        business.get("approval_rate", 0) or 0
    )
    threshold = float(
        business.get("threshold", 0) or 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Champion ROC-AUC", f"{roc_auc:.4f}")
    c2.metric("PR-AUC", f"{pr_auc:.4f}")
    c3.metric("NPA Reduction", f"{npa_reduction * 100:.2f}%")
    c4.metric("Approval Rate", f"{approval_rate * 100:.2f}%")

    st.info(
        f"**Policy threshold: {threshold:.3f}**\n\n"
        "The threshold is the decision cutoff used by the "
        "evaluated lending policy. Review the validation and "
        "test metrics before using this policy in practice."
    )

    with st.expander("Understand the model metrics", expanded=True):
        st.markdown(
            f"""
            **ROC-AUC: {roc_auc:.4f}**

            Measures how well the model ranks default and
            non-default applicants across classification thresholds.
            A value closer to 1 generally indicates better separation.

            **PR-AUC: {pr_auc:.4f}**

            Summarizes the precision-recall trade-off for the
            positive/default class. It is useful when classes
            are imbalanced.

            **NPA Reduction: {npa_reduction * 100:.2f}%**

            This is the business backtest metric supplied by the API.
            Its interpretation depends on the project's baseline
            and evaluation methodology.

            **Approval Rate: {approval_rate * 100:.2f}%**

            The proportion of applicants approved under the
            evaluated policy.
            """
        )

    st.markdown("### 🔬 Model Comparison")

    comparison_path = os.path.join(
        ART, "metrics", "model_comparison.csv"
    )
    comparison = load_csv(comparison_path)

    if not comparison.empty:
        if {"model", "roc_auc"}.issubset(comparison.columns):
            fig = px.bar(
                comparison,
                x="model",
                y="roc_auc",
                text_auto=".4f",
                title="Test ROC-AUC Comparison",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "Model comparison CSV must contain model and roc_auc columns."
            )
    else:
        st.info("Model comparison data is not available.")

    st.markdown("### 💼 Policy Threshold & Business Trade-off")

    display_image(
        os.path.join(ART, "plots", "threshold_business_curve.png"),
        "Validation-selected policy threshold and business trade-off",
    )

# ============================================================
# TAB 2: APPLICANT SCORING
# ============================================================

with scoring_tab:
    st.subheader("🔎 Real-time Applicant Scoring")

    st.write(
        "Enter applicant information to obtain a risk prediction "
        "from the FastAPI backend."
    )

    with st.form("credit_risk_scoring_form"):
        left, right = st.columns(2)

        with left:
            applicant_id = st.text_input("Applicant ID", "APP-001")

            util = st.number_input(
                "Revolving utilization",
                min_value=0.0,
                max_value=20.0,
                value=0.35,
                step=0.01,
            )

            age = st.number_input(
                "Age", min_value=18, max_value=120, value=35
            )

            d30 = st.number_input(
                "30–59 days past due",
                min_value=0,
                max_value=50,
                value=0,
            )

            debt = st.number_input(
                "Debt ratio",
                min_value=0.0,
                max_value=50.0,
                value=0.35,
                step=0.01,
            )

            income = st.number_input(
                "Monthly income",
                min_value=0.0,
                max_value=1_000_000.0,
                value=5000.0,
                step=100.0,
            )

        with right:
            open_lines = st.number_input(
                "Open credit lines + loans",
                min_value=0,
                max_value=200,
                value=8,
            )

            d90 = st.number_input(
                "90+ days late",
                min_value=0,
                max_value=50,
                value=0,
            )

            real = st.number_input(
                "Real-estate loans / lines",
                min_value=0,
                max_value=100,
                value=1,
            )

            d60 = st.number_input(
                "60–89 days past due",
                min_value=0,
                max_value=50,
                value=0,
            )

            deps = st.number_input(
                "Dependents",
                min_value=0.0,
                max_value=30.0,
                value=1.0,
                step=1.0,
            )

        submitted = st.form_submit_button(
            " Run Risk Assessment",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        payload = {
            "applicant_id": applicant_id,
            "RevolvingUtilizationOfUnsecuredLines": util,
            "age": age,
            "NumberOfTime30_59DaysPastDueNotWorse": d30,
            "DebtRatio": debt,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": d90,
            "NumberRealEstateLoansOrLines": real,
            "NumberOfTime60_89DaysPastDueNotWorse": d60,
            "NumberOfDependents": deps,
        }

        try:
            with st.spinner("Calculating applicant risk..."):
                result_response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30,
                )

            if not result_response.ok:
                st.error(
                    f"Prediction API returned status "
                    f"{result_response.status_code}."
                )
                st.code(result_response.text)
            else:
                result = result_response.json()

                risk_probability = float(
                    result.get("risk_probability", 0) or 0
                )
                risk_band = result.get("risk_band", "Unknown")
                policy_threshold = float(
                    result.get("policy_threshold", 0) or 0
                )
                decision = result.get("decision", "Unknown")

                st.markdown("### 📋 Prediction Result")

                a, b, c = st.columns(3)
                a.metric(
                    "Default Risk",
                    f"{risk_probability * 100:.2f}%",
                )
                b.metric("Risk Band", str(risk_band))
                c.metric(
                    "Policy Threshold",
                    f"{policy_threshold:.3f}",
                )

                st.progress(
                    min(max(risk_probability, 0.0), 1.0),
                    text=f"Risk score: {risk_probability:.4f}",
                )

                if risk_probability < policy_threshold:
                    st.success(f"Decision: {decision}")
                else:
                    st.warning(f"Decision: {decision}")

                st.divider()
                st.markdown("### 🧠 Applicant-level SHAP Explanation")

                st.write(
                    "SHAP contributions explain how features influence "
                    "the model output. They do not establish causation "
                    "and are not percentage changes in probability."
                )

                drivers = result.get("top_model_drivers", [])

                if drivers:
                    applicant_values = {
                        "RevolvingUtilizationOfUnsecuredLines": util,
                        "age": age,
                        "NumberOfTime30_59DaysPastDueNotWorse": d30,
                        "DebtRatio": debt,
                        "MonthlyIncome": income,
                        "NumberOfOpenCreditLinesAndLoans": open_lines,
                        "NumberOfTimes90DaysLate": d90,
                        "NumberRealEstateLoansOrLines": real,
                        "NumberOfTime60_89DaysPastDueNotWorse": d60,
                        "NumberOfDependents": deps,
                    }

                    rows = []

                    for driver in drivers:
                        feature = driver.get("feature", "Unknown")

                        try:
                            shap_value = float(
                                driver.get("shap_value", 0) or 0
                            )
                        except (ValueError, TypeError):
                            shap_value = 0.0

                        if shap_value > 0:
                            direction = "Higher-risk direction"
                            explanation = (
                                "This feature pushed the model output "
                                "toward higher default risk."
                            )
                        elif shap_value < 0:
                            direction = "Lower-risk direction"
                            explanation = (
                                "This feature pushed the model output "
                                "toward lower default risk."
                            )
                        else:
                            direction = "Neutral"
                            explanation = (
                                "This feature had a zero or negligible "
                                "directional contribution."
                            )

                        rows.append(
                            {
                                "Feature": feature,
                                "Applicant Value": applicant_values.get(
                                    feature, "Engineered feature"
                                ),
                                "SHAP Value": shap_value,
                                "Direction": direction,
                                "Explanation": explanation,
                            }
                        )

                    driver_df = pd.DataFrame(rows)

                    lower_risk = driver_df[
                        driver_df["SHAP Value"] < 0
                    ].sort_values("SHAP Value")

                    higher_risk = driver_df[
                        driver_df["SHAP Value"] > 0
                    ].sort_values(
                        "SHAP Value", ascending=False
                    )

                    st.markdown("#### 🟢 Factors pushing risk lower")
                    if lower_risk.empty:
                        st.info("No lower-risk SHAP contributions returned.")
                    else:
                        st.dataframe(
                            lower_risk,
                            use_container_width=True,
                            hide_index=True,
                        )

                    st.markdown("#### 🔴 Factors pushing risk higher")
                    if higher_risk.empty:
                        st.info("No higher-risk SHAP contributions returned.")
                    else:
                        st.dataframe(
                            higher_risk,
                            use_container_width=True,
                            hide_index=True,
                        )

                    st.markdown("#### 📊 SHAP Contribution Chart")

                    chart_df = driver_df.copy()
                    chart_df["Absolute SHAP"] = (
                        chart_df["SHAP Value"].abs()
                    )
                    chart_df = chart_df.sort_values("Absolute SHAP")

                    fig = px.bar(
                        chart_df,
                        x="SHAP Value",
                        y="Feature",
                        orientation="h",
                        title="Applicant-level SHAP Contributions",
                    )
                    fig.add_vline(x=0, line_width=1)
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(
                        "SHAP explains model behaviour; it does not "
                        "prove that a feature causes default."
                    )
                else:
                    st.info(
                        "The API did not return applicant-level SHAP drivers."
                    )

        except requests.RequestException as e:
            st.error(f"Could not connect to FastAPI: {e}")
        except (ValueError, TypeError) as e:
            st.error(f"Could not process the prediction response: {e}")

# ============================================================
# TAB 3: MODEL EXPLAINABILITY
# ============================================================

with explainability_tab:
    st.subheader("🧠 Model Explainability")

    st.markdown(
        """
        This section explains the model's evaluation and feature
        contributions.

        - **ROC-AUC:** Overall ability to rank default cases above
          non-default cases.
        - **PR-AUC:** Precision-recall performance for the default class.
        - **Global SHAP:** Features with the largest average impact
          on the model output.
        - **Applicant-level SHAP:** Features that influenced one
          applicant's prediction.
        """
    )

    st.markdown("### 1️⃣ ROC Curve")

    st.metric("ROC-AUC", f"{roc_auc:.4f}")

    st.write(
        "ROC curves show the trade-off between the true positive rate "
        "and false positive rate as the classification threshold changes. "
        "ROC-AUC is not an applicant's probability of default."
    )

    display_image(
        os.path.join(ART, "plots", "roc_curve.png"),
        f"Champion ROC Curve — AUC = {roc_auc:.3f}",
    )

    st.markdown("### 2️⃣ Precision-Recall Curve")

    st.metric("PR-AUC", f"{pr_auc:.4f}")

    st.write(
        "Precision measures how many applicants predicted as risky "
        "were actual defaults. Recall measures how many actual defaults "
        "were identified by the model."
    )

    display_image(
        os.path.join(ART, "plots", "pr_curve.png"),
        f"Champion Precision-Recall Curve — PR-AUC = {pr_auc:.3f}",
    )

    st.markdown("### 3️⃣ Global SHAP Feature Importance")

    shap_path = os.path.join(
        ART, "metrics", "shap_global_importance.csv"
    )
    importance = load_csv(shap_path)

    if not importance.empty:
        required_columns = {"feature", "mean_abs_shap"}

        if required_columns.issubset(importance.columns):
            importance = importance.head(15).sort_values(
                "mean_abs_shap"
            )

            fig = px.bar(
                importance,
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Top Features by Mean Absolute SHAP Value",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Mean absolute SHAP describes average contribution "
                "magnitude, not whether a feature increases or decreases risk."
            )
        else:
            st.warning(
                "SHAP CSV must contain feature and mean_abs_shap columns."
            )
    else:
        st.info("Global SHAP importance file is not available.")

    st.divider()

    st.markdown("### 🔍 In simple words")

    st.info(
        """
        **ROC-AUC:** How well does the model distinguish risky and
        non-risky applicants overall?

        **PR-AUC:** How well does the model identify the default class?

        **Global SHAP:** Which features generally influence the model?

        **Applicant-level SHAP:** Which features pushed one prediction
        toward higher or lower risk?
        """
    )



