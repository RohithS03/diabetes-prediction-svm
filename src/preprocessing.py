import pandas as pd
from sklearn.preprocessing import StandardScaler

EXPECTED_ZERO_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

def clean_and_scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Replaces invalid zero values using train-set medians
    and applies standard scaling.

    Returns:
        X_train_scaled, X_test_scaled, fitted_scaler
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    missing = set(EXPECTED_ZERO_COLS) - set(X_train.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in EXPECTED_ZERO_COLS:
        valid_values = X_train[X_train[col] != 0][col]
        median = valid_values.median() if not valid_values.empty else 0
        X_train[col] = X_train[col].replace(0, median)
        X_test[col] = X_test[col].replace(0, median)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
