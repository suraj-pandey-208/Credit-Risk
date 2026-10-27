from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts" / "monitoring"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
NUMERIC = ["RevolvingUtilizationOfUnsecuredLines","age","NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome","NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate","NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse","NumberOfDependents"]

def psi(expected, actual, bins=10):
    expected=np.asarray(expected,dtype=float); actual=np.asarray(actual,dtype=float)
    expected=expected[np.isfinite(expected)]; actual=actual[np.isfinite(actual)]
    if len(expected)==0 or len(actual)==0: return None
    qs=np.unique(np.quantile(expected,np.linspace(0,1,bins+1)))
    if len(qs)<3: return 0.0
    e=np.histogram(expected,qs)[0]/len(expected); a=np.histogram(actual,qs)[0]/len(actual)
    e=np.clip(e,1e-6,None); a=np.clip(a,1e-6,None)
    return float(np.sum((a-e)*np.log(a/e)))

def build_monitoring_report(reference: pd.DataFrame, current: pd.DataFrame):
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        report=Report([DataDriftPreset()])
        snapshot=report.run(current_data=current, reference_data=reference)
        path=REPORT_DIR / "evidently_data_drift.html"
        snapshot.save_html(str(path))
        return {"engine":"evidently", "report":str(path)}
    except Exception as e:
        rows=[]
        for c in NUMERIC:
            if c in reference and c in current:
                rows.append({"feature":c,"psi":psi(reference[c],current[c])})
        out=pd.DataFrame(rows).sort_values("psi",ascending=False)
        path=REPORT_DIR / "psi_fallback.csv"
        out.to_csv(path,index=False)
        return {"engine":"psi_fallback", "report":str(path), "reason":str(e)}
