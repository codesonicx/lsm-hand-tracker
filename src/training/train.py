from src.training.extraction import data_extraction
from src.training.preprocessing import prepare_data_for_training
from src.training.training import train_model


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

    print("\n2) Preprocessing…")
    prepare_data_for_training()

    print("\n3) Training the model…")
    train_model()

    print("\n✅ Pipeline complete!")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
