import mlflow
import os
import re

def parse_classification_report(filepath):
    """
    Parse classification_report.txt and return a dictionary of metrics.
    """
    metrics = {}
    if not os.path.exists(filepath):
        print(f"❌ Report file not found: {filepath}")
        return metrics

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Regular expression to match metrics rows
    pattern = re.compile(
        r"^\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$"
    )

    for line in lines:
        match = pattern.match(line)
        if match:
            label, precision, recall, f1, support = match.groups()
            metrics[f"{label}_precision"] = float(precision)
            metrics[f"{label}_recall"] = float(recall)
            metrics[f"{label}_f1_score"] = float(f1)
            metrics[f"{label}_support"] = int(support)

        if line.strip().startswith("accuracy"):
            accuracy = float(line.strip().split()[1])
            metrics["accuracy"] = accuracy

    return metrics


def log_latest_classification_report():
    """
    Log classification metrics to MLflow.
    """
    report_path = os.path.join("reports", "classification_report.txt")

    mlflow.set_experiment("Mexican Sign Language")

    with mlflow.start_run():
        metrics = parse_classification_report(report_path)
        if metrics:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            mlflow.log_artifact(report_path)
            print("📈 MLflow metrics and artifact logged.")
        else:
            print("⚠️ No metrics extracted.")
