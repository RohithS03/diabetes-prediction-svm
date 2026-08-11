from src.data_loader import load_data
from src.preprocessing import clean_and_scale
from src.train import split_data
from src.model import train_with_cv, save_model
from src.evaluate import evaluate

def main():
    data = load_data("data/diabetes.csv")

    X = data.drop(columns="Outcome")
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled, _ = clean_and_scale(
        X_train, X_test
    )

    model, cv_score = train_with_cv(X_train_scaled, y_train)
    save_model(model)

    test_accuracy = evaluate(model, X_test_scaled, y_test)

    print(f"Cross-Validation Accuracy: {cv_score:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
