from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def evaluate_cv(train_model_fn, X, y_np, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    auc_list = []

    for train_idx, valid_idx in skf.split(X, y_np):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y_np[train_idx], y_np[valid_idx]

        model = train_model_fn(X_train, y_train, X_valid, y_valid)
        y_pred = model.predict_proba(X_valid)[:, 1]

        auc = roc_auc_score(y_valid, y_pred)
        auc_list.append(auc)

    return float(np.mean(auc_list)), float(np.std(auc_list))
