import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RAW_TRAIN = "data/raw/adult.data"

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


def preprocess_pca(train_path):
    """PCA用: adult.data を読み込み、NUM_COLSを欠損補完→標準化して返す（yも返す）"""
    train_df = pd.read_csv(
        train_path,
        header=None,
        names=COLS,
        skipinitialspace=True
    )

    # "?" を欠損扱い（adult.data は '?' が多い）
    train_df = train_df.replace({"?": pd.NA})

    # NUM_COLS を数値化（念のため）
    train_df[NUM_COLS] = train_df[NUM_COLS].apply(pd.to_numeric, errors="coerce")

    # 欠損を中央値で補完
    median = train_df[NUM_COLS].median(numeric_only=True)
    train_df[NUM_COLS] = train_df[NUM_COLS].fillna(median)

    # y（可視化色分け用）
    y_train = train_df[TARGET_COL]

    # 標準化（PCA用）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df[NUM_COLS])

    feature_cols = NUM_COLS  # PCA対象

    return X_scaled, y_train, feature_cols, scaler


def fit_pca(X_scaled, n_components=None, random_state=42):
    """
    PCAをfitして返す。
    - n_components=None の場合は d 成分（=フル）を保持
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_scaled)
    return pca


def compute_scores(pca, X_scaled, n_components=None):
    """PC得点（scores）を返す。n_components指定で先頭k成分だけ切り出し。"""
    scores = pca.transform(X_scaled)
    if n_components is None:
        return scores
    return scores[:, :n_components]


def compute_loadings(pca, feature_cols, n_components=None):
    """
    負荷量（loadings）をDataFrameで返す。
    sklearnの components_ は (k, d) なので、列=PC, 行=feature にする。
    """
    comps = pca.components_
    if n_components is not None:
        comps = comps[:n_components, :]

    loadings = pd.DataFrame(
        comps.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(comps.shape[0])]
    )
    return loadings


def compute_T2_Q(X_scaled, pca, n_components=None):
    """
    T^2: Hotelling's T-squared（PC空間内の大きさ）
    Q  : SPE（再構成誤差の二乗和）
    """
    n, d = X_scaled.shape

    if n_components is None:
        n_components = pca.n_components_  # PCA()なら通常 d

    # スコア（PC得点）
    scores = pca.transform(X_scaled)[:, :n_components]     # (n, k)

    # 固有値（分散）
    eigvals = pca.explained_variance_[:n_components]       # (k,)

    # T^2 = sum_j (t_ij^2 / lambda_j)
    T2 = np.sum((scores ** 2) / eigvals, axis=1)

    # 再構成（k成分のみで復元）→ 残差 → Q = ||resid||^2
    Pk = pca.components_[:n_components, :]                 # (k, d)
    X_hat = scores @ Pk                                    # (n, d)
    resid = X_scaled - X_hat
    Q = np.sum(resid ** 2, axis=1)

    return T2, Q
