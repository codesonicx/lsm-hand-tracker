import pandas as pd
from pathlib import Path
import joblib
from typing import Optional, cast

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.path_config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR


def load_balanced_data(path: Path = PROCESSED_DIR / 'gestures_balanced.csv') -> pd.DataFrame:
    """
    Load the balanced dataset CSV.
    """
    return pd.read_csv(path)


def plot_metrics(
    y_true,
    y_pred,
    classes: list[str],
    figures_dir: Path
) -> None:
    """
    Generate and save confusion matrix and per-class metric plots.
    """
    # Ensure figures_dir exists
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=classes,
        yticklabels=classes,
        cmap='Blues'
    )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    cm_path = figures_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # Per-class precision, recall, f1-score
    metrics_df = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True)
    ).T
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics_df[['precision', 'recall', 'f1-score']].iloc[:-3].plot(
        kind='bar', ax=ax
    )
    ax.set_ylabel('Score')
    ax.set_title('Per-class Precision, Recall, and F1-score')
    plt.xticks(rotation=45, ha='right')
    metrics_path = figures_dir / 'classification_metrics.png'
    fig.tight_layout()
    fig.savefig(metrics_path)
    plt.close(fig)
    print(f"Saved classification metrics plot to {metrics_path}")


def train_model(
    data: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: Path = MODELS_DIR / 'gesture_classifier.joblib'
) -> None:
    """
    Train a RandomForest classifier, evaluate it, save metrics and model.
    """
    # Load data
    df = data if data is not None else load_balanced_data()
    X = df.drop(columns=['label'])
    y = df['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    # Predict & metrics
    y_pred = clf.predict(X_test)
    report = cast(str, classification_report(y_test, y_pred))

    # Save report
    report_file = REPORTS_DIR / 'classification_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved classification report to {report_file}")

    # Plot metrics
    plot_metrics(
        y_true=y_test,
        y_pred=y_pred,
        classes=list(clf.classes_),
        figures_dir=FIGURES_DIR
    )

    # Save model
    joblib.dump(clf, model_path)
    print(f"Saved trained model to {model_path}")


def main() -> None:
    train_model()


if __name__ == '__main__':
    main()