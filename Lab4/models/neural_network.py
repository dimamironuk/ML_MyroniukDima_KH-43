import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


class NeuralNetworkTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def fit_baseline(self, X_train_scaled, y_train, X_val_scaled, y_val):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            max_iter=40, early_stopping=True, random_state=self.random_state
        )
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict_proba(X_val_scaled)[:, 1]
        return roc_auc_score(y_val, y_pred)

    def tune(self, X_train_scaled, y_train, X_val_scaled, y_val, n_iter=4):
        param_dist = {
            'hidden_layer_sizes': [(64, 32), (100,)],
            'alpha': [0.0001, 0.01]
        }
        base = MLPClassifier(activation='relu', solver='adam', max_iter=40, early_stopping=True, random_state=self.random_state)
        search = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=n_iter, scoring='roc_auc', cv=3, random_state=self.random_state, n_jobs=-1)
        search.fit(X_train_scaled, y_train)
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        y_pred = self.model.predict_proba(X_val_scaled)[:, 1]
        return roc_auc_score(y_val, y_pred), self.best_params

    def predict_proba(self, X_scaled):
        return self.model.predict_proba(X_scaled)[:, 1]