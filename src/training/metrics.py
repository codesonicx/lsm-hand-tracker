from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.path_config import REPORTS_DIR


def plot_metrics(y_true, y_pred, classes: list[str], figures_dir: Path) -> None:
    """
    Generate and save confusion matrix and per-class metric plots.
    """
    # Ensure figures_dir exists
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues"
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # Per-class precision, recall, f1-score
    metrics_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics_df[["precision", "recall", "f1-score"]].iloc[:-3].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision, Recall, and F1-score")
    plt.xticks(rotation=45, ha="right")
    metrics_path = figures_dir / "classification_metrics.png"
    fig.tight_layout()
    fig.savefig(metrics_path)
    plt.close(fig)
    print(f"Saved classification metrics plot to {metrics_path}")


def class_report(y_test, y_pred):
    """
    Report of what?
    """
    report = cast(str, classification_report(y_test, y_pred))
    report_file = REPORTS_DIR / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved classification report to {report_file}")
