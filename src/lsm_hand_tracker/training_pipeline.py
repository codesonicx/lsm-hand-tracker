from lsm_hand_tracker.processing.data_extraction import data_extraction
from lsm_hand_tracker.processing.flatten import flatten_local_images
from lsm_hand_tracker.processing.cleaning import clean_local_dataset
from lsm_hand_tracker.processing.transformations import prepare_training_dataset
from lsm_hand_tracker.training import train_model
#Adding mlflow
from lsm_hand_tracker.processing.log_mlflow import log_latest_classification_report


def run_pipeline():
    """
    Execute the full processing pipeline:
      1) Generate metadata (JSON with landmarks and engineered features).
      2) Flatten metadata into a CSV file.
      3) Clean the dataset (drop unused columns, select preferred hand, remove NaNs).
      4) Transform features and balance classes (PowerTransformer, PCA, SMOTE).
      5) (Optional) Train and evaluate the predictive model.
    """
    print("1) Generating metadata…")
    data_extraction()

    print("2) Flattening metadata to CSV…")
    flatten_local_images()

    print("3) Cleaning the dataset…")
    clean_local_dataset()

    print("4) Transforming and balancing features…")
    prepare_training_dataset()

    print("5) Training the model…")
    train_model()
#Adding mlflow
    print("6) Logging results to MLflow…")
    log_latest_classification_report()

    print("✅ Pipeline complete!")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
