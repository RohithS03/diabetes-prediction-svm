import numpy as np
from src.model import train_with_cv

def test_model_training_runs():
    X = np.random.rand(30, 5)
    y = np.random.randint(0, 2, 30)

    model, score = train_with_cv(X, y)

    assert model is not None
    assert 0.0 <= score <= 1.0
