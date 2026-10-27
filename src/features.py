
import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"
ID_COL = "Unnamed: 0"

def engineer_features(df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
    """Application-time feature engineering for the Give Me Some Credit schema."""
    x = df.copy()

    if ID_COL in x.columns:
        x = x.drop(columns=[ID_COL])

    if not include_target and TARGET in x.columns:
        x = x.drop(columns=[TARGET])

    # Explicit missingness flags: missing income/dependents can itself carry signal.
    if "MonthlyIncome" in x:
        x["MonthlyIncome_missing"] = x["MonthlyIncome"].isna().astype("int8")
    if "NumberOfDependents" in x:
        x["NumberOfDependents_missing"] = x["NumberOfDependents"].isna().astype("int8")

    income = x["MonthlyIncome"].clip(lower=1) if "MonthlyIncome" in x else pd.Series(1, index=x.index)
    deps = x["NumberOfDependents"].fillna(0) if "NumberOfDependents" in x else pd.Series(0, index=x.index)
    age = x["age"].clip(lower=18) if "age" in x else pd.Series(18, index=x.index)

    # AntiGravity-style risk features adapted to the columns that actually exist here.
    x["annual_income"] = income * 12
    x["income_log"] = np.log1p(income)
    x["utilization_log"] = np.log1p(x["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0))
    x["debt_ratio_log"] = np.log1p(x["DebtRatio"].clip(lower=0))
    x["income_per_dependent"] = income / (deps + 1)
    x["open_lines_per_age"] = x["NumberOfOpenCreditLinesAndLoans"] / age
    x["real_estate_per_age"] = x["NumberRealEstateLoansOrLines"] / age
    x["dependents_to_age"] = deps / age
    x["credit_lines_to_real_estate"] = (
        x["NumberOfOpenCreditLinesAndLoans"] + 1
    ) / (x["NumberRealEstateLoansOrLines"] + 1)

    # Delinquency history aggregates.
    x["past_due_total"] = (
        x["NumberOfTime30-59DaysPastDueNotWorse"]
        + x["NumberOfTime60-89DaysPastDueNotWorse"]
        + x["NumberOfTimes90DaysLate"]
    )
    x["past_due_weighted"] = (
        x["NumberOfTime30-59DaysPastDueNotWorse"]
        + 2 * x["NumberOfTime60-89DaysPastDueNotWorse"]
        + 3 * x["NumberOfTimes90DaysLate"]
    )
    x["any_90_plus"] = (x["NumberOfTimes90DaysLate"] > 0).astype("int8")
    x["any_60_plus"] = (
        x["NumberOfTimes90DaysLate"] + x["NumberOfTime60-89DaysPastDueNotWorse"] > 0
    ).astype("int8")
    x["delinquency_flag"] = (x["past_due_total"] > 0).astype("int8")

    # Risk flags / interactions.
    x["high_utilization_flag"] = (
        x["RevolvingUtilizationOfUnsecuredLines"] > 0.80
    ).astype("int8")
    x["extreme_utilization_flag"] = (
        x["RevolvingUtilizationOfUnsecuredLines"] > 1.00
    ).astype("int8")
    x["high_debt_ratio_flag"] = (x["DebtRatio"] > 0.50).astype("int8")
    x["very_high_debt_ratio_flag"] = (x["DebtRatio"] > 1.00).astype("int8")
    x["young_borrower_flag"] = (x["age"] < 25).astype("int8")
    x["senior_flag"] = (x["age"] >= 65).astype("int8")
    x["utilization_x_debt_ratio"] = (
        x["RevolvingUtilizationOfUnsecuredLines"] * x["DebtRatio"]
    )
    x["late_payment_severity"] = (
        x["NumberOfTime30-59DaysPastDueNotWorse"]
        + 1.5 * x["NumberOfTime60-89DaysPastDueNotWorse"]
        + 2.5 * x["NumberOfTimes90DaysLate"]
    )

    return x.replace([np.inf, -np.inf], np.nan)
