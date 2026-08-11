import pytest
import pandas as pd
from src.preprocessing import clean_and_scale, EXPECTED_ZERO_COLS

def test_clean_and_scale():
    df_train = pd.DataFrame({
        "Glucose": [0, 100, 120, 0],
        "BloodPressure": [80, 0, 70, 90],
        "SkinThickness": [20, 30, 0, 0],
        "Insulin": [0, 0, 150, 160],
        "BMI": [25.0, 30.0, 0, 0],
        "Outcome": [0, 1, 0, 1]
    })
    
    df_test = pd.DataFrame({
        "Glucose": [0, 110],
        "BloodPressure": [0, 75],
        "SkinThickness": [0, 25],
        "Insulin": [0, 155],
        "BMI": [0, 27.5],
        "Outcome": [0, 1]
    })

    X_train_scaled, X_test_scaled, scaler = clean_and_scale(df_train, df_test)

    assert X_train_scaled.shape == df_train.shape
    assert X_test_scaled.shape == df_test.shape
    # Check that there are no NaNs
    assert not pd.isna(X_train_scaled).any()
    assert not pd.isna(X_test_scaled).any()
