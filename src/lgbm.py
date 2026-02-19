import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

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

def preprocess(train_path):
    train_df = pd.read_csv(train_path, header=None, names=COLS, skipinitialspace=True)

    # " ?" を欠損扱いにする
    train_df = train_df.replace("?", pd.NA)

    X_train = train_df[NUM_COLS + CAT_COLS].copy()

    X_train[CAT_COLS] = X_train[CAT_COLS].astype("category")

    y_train = train_df[TARGET_COL].str.strip().map({
        "<=50K": 0,
        ">50K": 1
    })

    feature_cols = NUM_COLS + CAT_COLS

    return X_train, y_train, feature_cols

def train_model(X_train, y_train, X_valid, y_valid):

    model = LGBMClassifier(
        learning_rate=0.05,
        n_estimators=5000,
        num_leaves=63,
        min_child_samples=500,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        random_state=42,
        verbosity=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[
            early_stopping(50),
            log_evaluation(0)
        ],
    )

    return model
