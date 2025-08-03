from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_auc_score, roc_curve
)
import xgboost as xgb
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np


def get_model(model_name: str, params: dict = None, random_state: int = 42):
    model_name = model_name.lower()
    params = params or {}
    params["random_state"] = random_state

    if model_name == "random_forest":
        return RandomForestClassifier(**params)
    elif model_name == "logistic_regression":
        return LogisticRegression(max_iter=5000, **params)
    elif model_name == "logistic_regression_l1":
        return LogisticRegression(max_iter=5000, penalty="l1", solver='saga', **params)
    elif model_name == "logistic_regression_l2":
        return LogisticRegression(max_iter=5000, penalty="l2", solver='saga', **params)
    elif model_name == "logistic_regression_elastic":
        return LogisticRegression(max_iter=5000, penalty="elasticnet", solver='saga', l1_ratio=0.5, **params)
    elif model_name == "svm":
        return SVC(probability=True, **params)
    elif model_name == "xgboost":
        return xgb.XGBClassifier(eval_metric="logloss", **params)
    elif model_name == "neural_network":
        return MLPClassifier(max_iter=5000, **params)
    elif model_name == "lightgbm":
        return LGBMClassifier(verbosity=-1, **params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def needs_scaling(model):
    return not isinstance(model, (xgb.XGBClassifier, RandomForestClassifier, LGBMClassifier))

def cross_validate_model(
    df: pd.DataFrame,
    model_name: str,
    model_params: dict = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:

    df = df.drop(columns=["ID"], errors="ignore")
    y = LabelEncoder().fit_transform(df["Diagnosis"])
    X = df.drop(columns=["Diagnosis"])

    model = get_model(model_name, params=model_params, random_state=random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics_per_fold = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }

    roc_data = []

    for train_idx, test_idx in skf.split(X, y):

        if needs_scaling(model):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X.iloc[train_idx])
            X_test = scaler.transform(X.iloc[test_idx])
        else:
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]          
        
        model.fit(X_train, y[train_idx])
        y_pred = model.predict(X_test)
        y_true = y[test_idx]

        metrics_per_fold["accuracy"].append(accuracy_score(y_true, y_pred))
        metrics_per_fold["precision"].append(precision_score(y_true, y_pred, zero_division=0))
        metrics_per_fold["recall"].append(recall_score(y_true, y_pred, zero_division=0))
        metrics_per_fold["f1"].append(f1_score(y_true, y_pred, zero_division=0))

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = None

        if y_prob is not None:
            metrics_per_fold["roc_auc"].append(roc_auc_score(y_true, y_prob))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_data.append({
                "fpr": fpr,
                "tpr": tpr,
                "fold": len(roc_data)
            })

    results = {"model": model_name, "params": model_params or {}}
    for metric, values in metrics_per_fold.items():
        clean_values = [v for v in values if not np.isnan(v)]
        results[f"{metric}_mean"] = np.mean(clean_values)
        results[f"{metric}_std"] = np.std(clean_values)
        results[f"{metric}_min"] = np.min(clean_values)
        results[f"{metric}_max"] = np.max(clean_values)

        results["roc_data"] = roc_data

    return results


def refit_model(df: pd.DataFrame, model_name: str, model_params: dict = None, random_state: int = 42):
    '''
    Model training on entire dataset
    '''
    df = df.drop(columns=["ID"], errors="ignore")
    y = LabelEncoder().fit_transform(df["Diagnosis"])
    X = df.drop(columns=["Diagnosis"])

    model = get_model(model_name, params=model_params, random_state=random_state)
    
    if needs_scaling(model):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    model.fit(X, y)
    return model