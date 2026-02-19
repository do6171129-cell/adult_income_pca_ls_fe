# Adult Income – PCA–LS–LGBM 構造解析事例

## 概要
Adult Income データセットに対し、
LGBM ベースライン → PCA 構造解析 → Logistic Regression 構造解析 → Feature Engineering 検討
の流れでモデル挙動と特徴量構造を分析した事例。

## 目的

本事例では、分類性能の最大化ではなく、

- ベースラインモデルの構築
- 特徴量空間の構造把握（PCA）
- 線形モデルの係数・診断量解析（LR）
- 構造理解に基づく Feature Engineering 検討

を通じて、分析設計および構造解釈プロセスの可視化を目的とする。

## 内容

- Adult / Census Income データ使用
- LGBM によるベースライン評価
- PCA による分散構造解析
- Logistic Regression による係数・診断量解析
- PCA / LR 結果を踏まえた FE 検討

## 分析項目

- CV評価（AUC）
- Feature Importance（LGBM）
- 寄与率・負荷量（PCA）
- 係数・確率分布（LR）
- 診断量（residual / leverage / Cook）
- FE候補の逐次検証

## 出力

- ROC Curve
- Confusion Matrix
- Feature Importance
- 寄与率 / Loadings
- 主成分スコア可視化
- 診断量分布
- FE検証結果

## スコープ外

- スコア最適化競争
- ハイパーパラメータ探索沼
- 異常点除外による改善
- 因果解釈

```text
adult_income_pca_ls_fe/
├─ .gitattributes
├─ .gitignore
├─ requirements.txt
├─ data/
│  ├─ processed/
│  └─ raw/
│     ├─ adult.data
│     ├─ adult.names
│     ├─ adult.test
│     ├─ Index
│     └─ old.adult.names
├─ notebooks/
│  ├─ 00_analysis_overview.ipynb
│  ├─ 01_lgbm_baseline.ipynb
│  ├─ 02_pca_analysis.ipynb
│  ├─ 03_lr_analysis.ipynb
│  └─ 04_fe_improvement.ipynb
└─ src/
   ├─ feature_engineering.py
   ├─ lgbm.py
   ├─ lr.py
   └─ pca.py
```

## データ出典

本分析では以下の公開データセットを使用：

Adult (Census Income) Dataset

UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/2/adult





