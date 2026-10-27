import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from features import engineer_features

def test_engineer_features_shape_and_target_drop():
    df=pd.DataFrame([{
        "Unnamed: 0":1,"SeriousDlqin2yrs":0,"RevolvingUtilizationOfUnsecuredLines":0.3,
        "age":40,"NumberOfTime30-59DaysPastDueNotWorse":0,"DebtRatio":0.2,
        "MonthlyIncome":5000,"NumberOfOpenCreditLinesAndLoans":5,"NumberOfTimes90DaysLate":0,
        "NumberRealEstateLoansOrLines":1,"NumberOfTime60-89DaysPastDueNotWorse":0,"NumberOfDependents":1
    }])
    out=engineer_features(df)
    assert "SeriousDlqin2yrs" not in out.columns
    assert len(out)==1
    assert "past_due_weighted" in out.columns
