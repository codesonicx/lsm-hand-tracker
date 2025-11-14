from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.training.metrics import class_report, plot_metrics
from src.utils.path_config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR


def load_balanced_data(
    path: Path = PROCESSED_DIR / "gestures_balanced.csv",
) -> pd.DataFrame:
    """
    Load the balanced dataset CSV.
    """
    return pd.read_csv(path)


def train_model(
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: Path = MODELS_DIR / "gesture_classifier.joblib",
) -> None:
    """
    Train a RandomForest classifier, evaluate it, save metrics and model.
    """
    # Load data
    df = load_balanced_data()
    X = df.drop(columns=["label"])
    y = df["label"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    # Predict & metrics
    y_pred = clf.predict(X_test)
    class_report(y_test, y_pred)

    # Plot metrics
    plot_metrics(
        y_true=y_test,
        y_pred=y_pred,
        classes=list(clf.classes_),
        figures_dir=FIGURES_DIR,
    )

    # Save model
    joblib.dump(clf, model_path)
    print(f"Saved trained model to {model_path}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    pass
