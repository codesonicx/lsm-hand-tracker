from src.dataflow.data_extraction import data_extraction
from src.dataflow.flatten import flatten_local_images
from src.dataflow.cleaning import clean_local_dataset
from src.dataflow.transformations import prepare_training_dataset
from src.models.training import train_model


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

    print("\n2) Flattening metadata to CSV…")
    flatten_local_images()

    print("\n3) Cleaning the dataset…")
    clean_local_dataset()

    print("\n4) Transforming and balancing features…")
    prepare_training_dataset()

    print("\n5) Training the model…")
    train_model()

    print("\n✅ Pipeline complete!")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
