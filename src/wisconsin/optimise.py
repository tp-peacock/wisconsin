from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
import xgboost as xgb
import pandas as pd
import numpy as np
import optuna


def get_model(model_name, trial=None, params=None, random_state=42):
    if trial is not None:
        if model_name == "logistic_regression":
            return LogisticRegression(
                C=trial.suggest_loguniform("C", 1e-5, 1e3),
                penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
                solver="liblinear",  # only supports l1/l2
                max_iter=5000,
                random_state=random_state
            )

        elif model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 1000),
                max_depth=trial.suggest_int("max_depth", 2, 50),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
                random_state=random_state,
                n_jobs=-1
            )

        elif model_name == "svc":
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
            params = {
                "C": trial.suggest_loguniform("C", 1e-3, 1e2),
                "kernel": kernel,
                "probability": True,
                "random_state": random_state
            }
            if kernel in ["rbf", "poly", "sigmoid"]:
                params["gamma"] = trial.suggest_loguniform("gamma", 1e-4, 1e1)
            if kernel == "poly":
                params["degree"] = trial.suggest_int("degree", 2, 5)
            return SVC(**params)

        elif model_name == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                learning_rate=trial.suggest_loguniform("learning_rate", 1e-4, 0.3),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                gamma=trial.suggest_float("gamma", 0, 5),
                reg_alpha=trial.suggest_loguniform("reg_alpha", 1e-5, 10),
                reg_lambda=trial.suggest_loguniform("reg_lambda", 1e-5, 10),
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1
            )

        elif model_name == "neural_network":
            return MLPClassifier(
                hidden_layer_sizes=trial.suggest_categorical(
                    "hidden_layer_sizes", [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)]
                ),
                activation=trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
                solver=trial.suggest_categorical("solver", ["adam", "sgd"]),
                alpha=trial.suggest_loguniform("alpha", 1e-6, 1e-2),
                learning_rate=trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
                learning_rate_init=trial.suggest_loguniform("learning_rate_init", 1e-4, 1e-1),
                max_iter=5000,
                early_stopping=True,
                random_state=random_state
            )
    else:
            if model_name == "random_forest":
                return RandomForestClassifier(**params)
            elif model_name == "logistic_regression":
                return LogisticRegression(max_iter=5000, **params)
            elif model_name == "svm":
                return SVC(probability=True, **params)
            elif model_name == "xgboost":
                return xgb.XGBClassifier(eval_metric="logloss", **params)
            elif model_name == "neural_network":
                return MLPClassifier(max_iter=5000, **params)
            else:
                raise ValueError(f"Unsupported model: {model_name}")


def needs_scaling(model):
    return not isinstance(model, (xgb.XGBClassifier, RandomForestClassifier))


def cross_validate_model(df, model_name, model_params, n_splits=10, random_state=42):
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
        "roc_auc": []
    }

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
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        y_true = y[test_idx]

        metrics_per_fold["accuracy"].append(accuracy_score(y_true, y_pred))
        metrics_per_fold["precision"].append(precision_score(y_true, y_pred, zero_division=0))
        metrics_per_fold["recall"].append(recall_score(y_true, y_pred, zero_division=0))
        metrics_per_fold["f1"].append(f1_score(y_true, y_pred, zero_division=0))
        if y_prob is not None:
            metrics_per_fold["roc_auc"].append(roc_auc_score(y_true, y_prob))

    results = {"model": model_name, "params": model_params or {}}
    for metric, values in metrics_per_fold.items():
        clean_values = [v for v in values if not np.isnan(v)]
        if not clean_values:
            continue
        results[f"{metric}_mean"] = np.mean(clean_values)
        results[f"{metric}_std"] = np.std(clean_values)
        results[f"{metric}_min"] = np.min(clean_values)
        results[f"{metric}_max"] = np.max(clean_values)

    return results


def run_optuna(df, model_name, n_trials=30, n_splits=10, random_state=42):
    def objective(trial):
        model = get_model(model_name, trial=trial, random_state=random_state)
        df_local = df.drop(columns=["ID"], errors="ignore")
        y = LabelEncoder().fit_transform(df_local["Diagnosis"])
        X = df_local.drop(columns=["Diagnosis"])


        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        scores = []
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
            scores.append(accuracy_score(y[test_idx], y_pred))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.params

    return cross_validate_model(df, model_name, best_params, n_splits=n_splits, random_state=random_state)
