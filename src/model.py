from sklearn import svm
from sklearn.model_selection import GridSearchCV
import joblib
import os

MODEL_PATH = "models/svm_model.joblib"

def train_with_cv(X_train, y_train):
    """
    Trains an SVM using 5-fold cross-validation.
    """
    param_grid = {
        "C": [0.1, 1, 10]
    }

    svc = svm.SVC(kernel="linear", random_state=42)

    grid = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_

def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first.")
    return joblib.load(MODEL_PATH)
