import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def get_dataset_stats(train_df, test_df):
    num_train = len(train_df)
    num_test = len(test_df)
    num_vars = train_df.shape[1] - 1
    total_missing = train_df.isnull().sum().sum()
    total_cells = train_df.shape[0] * train_df.shape[1]
    missing_pct = (total_missing / total_cells) * 100
    duplicates = train_df.duplicated().sum()
    num_features = len(train_df.select_dtypes(include=[np.number]).columns) - 1
    cat_features = len(train_df.select_dtypes(include=['object', 'category']).columns)

    return {
        'num_train': num_train,
        'num_test': num_test,
        'num_vars': num_vars,
        'total_missing': total_missing,
        'missing_pct': missing_pct,
        'duplicates': duplicates,
        'num_features': num_features,
        'cat_features': cat_features
    }


def print_stats_summary(stats):
    print(f"Observations (train): {stats['num_train']:,}")
    print(f"Observations (test):  {stats['num_test']:,}")
    print(f"Variables:            {stats['num_vars']}")
    print(f"Missing values:       {stats['total_missing']:,} ({stats['missing_pct']:.1f}% of all cells)")
    print(f"Duplicates:           {stats['duplicates']}")
    print(f"Numeric features:     {stats['num_features']}")
    print(f"Categorical features: {stats['cat_features']}")


def prepare_pipeline(train_df, test_df):
    test_ids = test_df['SK_ID_CURR']
    y = train_df['TARGET']
    X = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
    X_test = test_df.drop(columns=['SK_ID_CURR'])

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in cat_cols:
        if X[col].nunique() <= 2:
            vals = sorted(pd.concat([X[col], X_test[col]]).dropna().unique())
            mapping = {v: i for i, v in enumerate(vals)}
            X[col] = X[col].map(mapping)
            X_test[col] = X_test[col].map(mapping)
        else:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
            X_test = pd.get_dummies(X_test, columns=[col], drop_first=True)

    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
    feature_names = X.columns.tolist()

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_test_imputed = imputer.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_imputed)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'X_test_imputed': X_test_imputed,
        'X_test_scaled': X_test_scaled,
        'test_ids': test_ids,
        'feature_names': feature_names
    }