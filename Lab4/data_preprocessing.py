import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def print_stats_summary(train_df, test_df, ignore_cols=('SK_ID_CURR', 'TARGET')):
    features = train_df.drop(columns=list(ignore_cols), errors='ignore')
    missing = train_df.isnull().sum().sum()
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"Features: {features.shape[1]} (Num: {features.select_dtypes(np.number).shape[1]}, Cat: {features.select_dtypes(exclude=np.number).shape[1]})")
    print(f"Missing: {missing:,} ({missing / train_df.size * 100:.1f}%) | Duplicates: {train_df.duplicated().sum()}")


def prepare_data(train_df, test_df, missing_thresh=0.5):
    test_ids = test_df['SK_ID_CURR']
    y = train_df['TARGET']
    X = train_df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    X_test = test_df.drop(columns=['SK_ID_CURR'], errors='ignore')

    keep_cols = X.columns[X.isnull().mean() <= missing_thresh]
    X, X_test = X[keep_cols].copy(), X_test[keep_cols].copy()

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    X = pd.get_dummies(X, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, drop_first=True, dtype=int)
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

    return X, y, X_test, test_ids, X.columns.tolist()


def get_folds(X, y, X_test, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_te = X_test.to_numpy()

    for train_idx, val_idx in skf.split(X, y):
        X_tr = X.iloc[train_idx].to_numpy()
        X_va = X.iloc[val_idx].to_numpy()

        scaler = StandardScaler()
        yield {
            'X_train': X_tr,
            'X_val': X_va,
            'X_test_imputed': X_te,
            'X_train_scaled': scaler.fit_transform(X_tr),
            'X_val_scaled': scaler.transform(X_va),
            'X_test_scaled': scaler.transform(X_te),
            'y_train': y.iloc[train_idx],
            'y_val': y.iloc[val_idx],
        }