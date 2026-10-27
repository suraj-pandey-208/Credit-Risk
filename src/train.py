
import json, os, time, joblib, numpy as np, pandas as pd, shap, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from features import engineer_features, TARGET

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(ROOT,"data","raw","cs-training.csv")
ART=os.path.join(ROOT,"artifacts")
for d in ["models","plots","metrics"]: os.makedirs(os.path.join(ART,d),exist_ok=True)

raw=pd.read_csv(DATA)
X=engineer_features(raw)
y=raw[TARGET].astype(int)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,stratify=y,random_state=42)
X_tr,X_val,y_tr,y_val=train_test_split(X_train,y_train,test_size=.20,stratify=y_train,random_state=42)

models={
"Logistic Regression": Pipeline([
("imputer",SimpleImputer(strategy="median")),
("scaler",StandardScaler()),
("model",LogisticRegression(max_iter=2000,class_weight="balanced",C=.5,random_state=42))
]),
"Random Forest": RandomForestClassifier(
n_estimators=400,max_depth=12,min_samples_leaf=3,class_weight="balanced_subsample",
max_features="sqrt",n_jobs=-1,random_state=42)
}
results=[]; fitted={}
for name,model in models.items():
    t=time.time(); model.fit(X_train,y_train); p=model.predict_proba(X_test)[:,1]
    results.append({"model":name,"roc_auc":roc_auc_score(y_test,p),"pr_auc":average_precision_score(y_test,p),"train_seconds":round(time.time()-t,2)})
    fitted[name]=model

xgb=XGBClassifier(
n_estimators=1200,max_depth=4,learning_rate=.025,subsample=.90,colsample_bytree=.90,
min_child_weight=5,reg_alpha=.05,reg_lambda=3.0,objective="binary:logistic",
eval_metric="auc",tree_method="hist",n_jobs=-1,random_state=42,early_stopping_rounds=80)
t=time.time(); xgb.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
p_val=xgb.predict_proba(X_val)[:,1]; p_test=xgb.predict_proba(X_test)[:,1]
results.append({"model":"XGBoost Champion","roc_auc":roc_auc_score(y_test,p_test),"pr_auc":average_precision_score(y_test,p_test),
"train_seconds":round(time.time()-t,2),"best_iteration":int(xgb.best_iteration)})
fitted["XGBoost Champion"]=xgb

metrics_df=pd.DataFrame(results).sort_values("roc_auc",ascending=False)
metrics_df.to_csv(os.path.join(ART,"metrics","model_comparison.csv"),index=False)

# Policy threshold is selected ONLY on validation data.
base_val=float(y_val.mean()); scan=[]
for th in np.arange(.02,.801,.005):
    approved=p_val<th
    npa=float(y_val[approved].mean()) if approved.any() else np.nan
    red=float(1-npa/base_val) if approved.any() else np.nan
    scan.append({"threshold":round(float(th),3),"approval_rate":float(approved.mean()),"npa":npa,"npa_reduction":red})
scan_df=pd.DataFrame(scan)
qual=scan_df[(scan_df.approval_rate>=.75)&(scan_df.npa_reduction>=.30)]
threshold=float(qual.threshold.max()) if not qual.empty else .30

approved=p_test<threshold
base_test=float(y_test.mean()); test_npa=float(y_test[approved].mean())
business={"threshold":threshold,"approval_rate":float(approved.mean()),"baseline_npa":base_test,
"approved_portfolio_npa":test_npa,"npa_reduction":float(1-test_npa/base_test),
"target_npa_reduction":.30,"target_approval_floor":.75}
business["auc_target_met"]=bool(metrics_df.iloc[0].roc_auc>=.85)
business["npa_target_met"]=bool(business["npa_reduction"]>=.30)

pred=(p_test>=threshold).astype(int)
champ={"roc_auc":float(roc_auc_score(y_test,p_test)),"pr_auc":float(average_precision_score(y_test,p_test)),
"accuracy":float(accuracy_score(y_test,pred)),"precision":float(precision_score(y_test,pred,zero_division=0)),
"recall":float(recall_score(y_test,pred,zero_division=0)),"f1":float(f1_score(y_test,pred,zero_division=0)),
"brier_score":float(brier_score_loss(y_test,p_test)),"threshold":threshold}

scan_df.to_csv(os.path.join(ART,"metrics","threshold_scan_validation.csv"),index=False)
pd.DataFrame({"y_true":y_test.values,"risk_score":p_test}).to_csv(os.path.join(ART,"metrics","test_predictions.csv"),index=False)
json.dump(business,open(os.path.join(ART,"metrics","business_metrics.json"),"w"),indent=2)
json.dump(champ,open(os.path.join(ART,"metrics","champion_metrics.json"),"w"),indent=2)

joblib.dump(xgb,os.path.join(ART,"models","champion_xgboost.joblib"))
joblib.dump(fitted["Logistic Regression"],os.path.join(ART,"models","logistic_regression.joblib"))
joblib.dump(fitted["Random Forest"],os.path.join(ART,"models","random_forest.joblib"))
joblib.dump(list(X.columns),os.path.join(ART,"models","feature_names.joblib"))

# Explainability
explainer=shap.TreeExplainer(xgb)
sample=X_test.sample(min(1000,len(X_test)),random_state=42)
sv=explainer.shap_values(sample)
if isinstance(sv,list): sv=sv[1]
importance=pd.DataFrame({"feature":sample.columns,"mean_abs_shap":np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap",ascending=False)
importance.to_csv(os.path.join(ART,"metrics","shap_global_importance.csv"),index=False)
joblib.dump(explainer,os.path.join(ART,"models","shap_explainer.joblib"))

# Visuals
plt.figure(figsize=(8,6)); RocCurveDisplay.from_predictions(y_test,p_test)
plt.title(f"ROC Curve — AUC {champ['roc_auc']:.3f}"); plt.tight_layout()
plt.savefig(os.path.join(ART,"plots","roc_curve.png"),dpi=160); plt.close()
plt.figure(figsize=(8,6)); PrecisionRecallDisplay.from_predictions(y_test,p_test)
plt.title(f"Precision-Recall Curve — AP {champ['pr_auc']:.3f}"); plt.tight_layout()
plt.savefig(os.path.join(ART,"plots","pr_curve.png"),dpi=160); plt.close()
plt.figure(figsize=(9,6))
plt.plot(scan_df.threshold,scan_df.approval_rate,label="Approval rate")
plt.plot(scan_df.threshold,scan_df.npa_reduction,label="NPA reduction")
plt.axvline(threshold,linestyle="--",label=f"Selected threshold {threshold:.3f}")
plt.axhline(.30,linestyle=":",label="NPA target 30%")
plt.xlabel("Risk threshold"); plt.ylabel("Rate"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(ART,"plots","threshold_business_curve.png"),dpi=160); plt.close()
plt.figure(figsize=(10,7)); shap.summary_plot(sv,sample,show=False,max_display=15)
plt.tight_layout(); plt.savefig(os.path.join(ART,"plots","shap_summary.png"),dpi=160,bbox_inches="tight"); plt.close()

metadata={"dataset":"cs-training.csv","rows":len(raw),"target":TARGET,"positive_rate":float(y.mean()),
"feature_count":X.shape[1],"split":{"train":len(X_train),"validation":len(X_val),"test":len(X_test),"random_state":42},
"champion":"XGBoost Champion","threshold":threshold,"business_metrics":business,
"top_features":importance.head(15).to_dict("records"),
"notes":["Real uploaded dataset only; no synthetic fallback.","ID excluded from modeling.","Threshold chosen on validation and evaluated on untouched test set.","SHAP is model explanation, not causal or regulatory adverse-action reasoning."]}
json.dump(metadata,open(os.path.join(ART,"models","model_metadata.json"),"w"),indent=2)
print(metrics_df.to_string(index=False)); print(json.dumps(business,indent=2))
