from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# ========================
# Path
# ========================

ROOT = Path().resolve().parents[1]

RAW_TRAIN = ROOT / "data/raw/train.csv"
RAW_TEST  = ROOT / "data/raw/test.csv"


# ========================
# Columns
# ========================

COLS = [
    "age","workclass","fnlwgt","education","education-num","marital-status","occupation",
    "relationship","race","sex","capital-gain","capital-loss","hours-per-week",
    "native-country","income"
]

NUM_COLS = ["age","education-num","capital-gain","capital-loss","hours-per-week"]

CAT_COLS = [
    "workclass","education","marital-status","occupation","relationship","race","sex","native-country"
]

TARGET_COL = "income"

def preprocess(train_path):
    train_df = pd.read_csv(
        train_path,
        names=COLS,
        na_values="?",
        skipinitialspace=True
    )

    median = train_df[NUM_COLS].median()
    cat_mode = train_df[CAT_COLS].mode(dropna=True).iloc[0]

    X = train_df[NUM_COLS + CAT_COLS].copy()
    X[NUM_COLS] = X[NUM_COLS].fillna(median)
    X[CAT_COLS] = X[CAT_COLS].fillna(cat_mode)

    X = pd.get_dummies(X, columns=CAT_COLS)

    y = train_df[TARGET_COL].str.replace(".", "", regex=False) \
                            .map({">50K": 1, "<=50K": 0}) \
                            .astype(int)

    return X, y, X.columns




def train_model(X_train, y_train):
    """
    sklearn LogisticRegression を学習して返す
    - preprocess() の出力をそのまま受ける想定（X: DataFrame, y: Series）
    - y は (N,) で渡す（torch版みたいに view(-1,1) は不要）
    """
    # DataFrame/Series -> numpy でもOKだけど、sklearnはDataFrameのままでも学習できる
    if hasattr(y_train, "to_numpy"):
        y = y_train.to_numpy()
    else:
        y = np.asarray(y_train)

    # 念のため 0/1 の int に寄せる
    y = y.astype(int)

    model = LogisticRegression(
        C=1.0, # 正則化の強さ
        solver="lbfgs", # 最適化アルゴリズム
        max_iter=5000, # 最大反復回数
        n_jobs=None,      # lbfgs では基本使われない 並列計算指定
        random_state=42, # 乱数シード固定
    )

    model.fit(X_train, y)
    return model

def compute_diagnostics_logistic(model, X, y):
    X_mat = np.asarray(X, dtype=np.float64)
    y_vec = np.asarray(y, dtype=np.float64)

    p = model.predict_proba(X_mat)[:, 1]
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    r_dev = np.sign(y_vec - p) * np.sqrt(
        -2.0 * (y_vec * np.log(p) + (1.0 - y_vec) * np.log(1.0 - p))
    )

    w = p * (1.0 - p)
    sw = np.sqrt(w)
    Xw = X_mat * sw[:, None]

    XtWX = Xw.T @ Xw  # (d,d)

    try:
        A = np.linalg.solve(XtWX, Xw.T).T  # (N,d)
    except np.linalg.LinAlgError:
        A = (np.linalg.pinv(XtWX) @ Xw.T).T

    h = np.einsum("ij,ij->i", A, Xw)

    d = X_mat.shape[1]
    cook = (r_dev**2 * h) / ((1.0 - h)**2 * d)

    return {"p": p, "r_dev": r_dev, "leverage": h, "cook": cook}