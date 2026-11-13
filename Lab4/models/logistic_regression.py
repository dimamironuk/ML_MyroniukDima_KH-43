from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


class LogisticRegressionTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def fit_baseline(self, X_train, y_train, X_val, y_val):
        self.model = SGDClassifier(loss='log_loss', penalty='l2', class_weight='balanced', max_iter=100, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)

    def tune(self, X_train, y_train, X_val, y_val):
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0], 'max_iter': [50, 100, 200]}
        base = SGDClassifier(loss='log_loss', penalty='l2', class_weight='balanced', random_state=self.random_state)
        search = GridSearchCV(base, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=4)
        search.fit(X_train, y_train)
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        y_pred = self.model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred), self.best_params

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]