from pathlib import Path
import pandas as pd
from src.features import engineer_features
from src.monitor import build_monitoring_report

root=Path(__file__).resolve().parents[1]
df=pd.read_csv(root/"data/raw/cs-training.csv")
ref=df.drop(columns=["SeriousDlqin2yrs"]).sample(30000,random_state=42)
# Example current window; in production replace with recent scored applicant features.
cur=df.drop(columns=["SeriousDlqin2yrs"]).sample(30000,random_state=7)
print(build_monitoring_report(ref,cur))
