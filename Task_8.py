import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, RocCurveDisplay, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, n_jobs=None, solver="lbfgs"))
])

pipe_rf = Pipeline([
    ("clf", RandomForestClassifier(random_state=42))
])

pipe_lr.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
       
        try:
            y_proba = model.decision_function(X_test)
        except:
            y_proba = None

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    return y_proba

proba_lr = evaluate("Logistic Regression (baseline)", pipe_lr, X_test, y_test)
proba_rf = evaluate("Random Forest (baseline)", pipe_rf, X_test, y_test)

plt.figure()
RocCurveDisplay.from_predictions(y_test, proba_lr, name="LogReg")
RocCurveDisplay.from_predictions(y_test, proba_rf, name="RandomForest")
plt.title("ROC Curves (Baseline)")
plt.show()

rf_est = pipe_rf.named_steps["clf"]
importances = rf_est.feature_importances_
indices = np.argsort(importances)[::-1][:10]  
plt.figure()
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha="right")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

rf_grid = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2]
}
grid_rf = GridSearchCV(
    estimator=Pipeline([("clf", RandomForestClassifier(random_state=42))]),
    param_grid=rf_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print("\nBest RF params:", grid_rf.best_params_)


lr_grid = {
    "scaler": [StandardScaler()], 
    "clf__C": [0.1, 1.0, 3.0, 10.0],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs"]
}
grid_lr = GridSearchCV(
    estimator=Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
    param_grid=lr_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
print("Best LR params:", grid_lr.best_params_)

proba_lr_tuned = evaluate("Logistic Regression (tuned)", best_lr, X_test, y_test)
proba_rf_tuned = evaluate("Random Forest (tuned)", best_rf, X_test, y_test)

plt.figure()
RocCurveDisplay.from_predictions(y_test, proba_lr_tuned, name="LogReg (tuned)")
RocCurveDisplay.from_predictions(y_test, proba_rf_tuned, name="RandomForest (tuned)")
plt.title("ROC Curves (Tuned Models)")
plt.show()

def winner(a, b):
    return "Model A" if a > b else ("Model B" if b > a else "Tie")

auc_lr = roc_auc_score(y_test, proba_lr_tuned)
auc_rf = roc_auc_score(y_test, proba_rf_tuned)
print(f"\nAUC — LR (tuned): {auc_lr:.4f} | RF (tuned): {auc_rf:.4f}")
print("✅ Better AUC:", "Logistic Regression" if auc_lr > auc_rf else "Random Forest" if auc_rf > auc_lr else "Tie")