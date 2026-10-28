import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


class RandomForestTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def fit_baseline(self, X_train, y_train, X_val, y_val):
        self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)

    def tune(self, X_train, y_train, X_val, y_val, n_iter=5):
        param_dist = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_leaf': [10, 30]
        }
        base = RandomForestClassifier(class_weight='balanced', random_state=self.random_state, n_jobs=-1)
        search = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=n_iter, scoring='roc_auc', cv=3, random_state=self.random_state, n_jobs=-1)
        search.fit(X_train, y_train)
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        y_pred = self.model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred), self.best_params

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]