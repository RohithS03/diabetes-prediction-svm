from sklearn.metrics import accuracy_score

def evaluate(model, X_test, y_test):
    """
    Evaluates model performance on unseen data.
    """
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)
