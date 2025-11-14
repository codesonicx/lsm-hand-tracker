import json

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE

from src.core.preprocssing import clean_dataset, flatten_metadata
from src.core.transformations import build_preprocessing_pipeline
from src.utils.path_config import INTERIM_DIR, MODELS_DIR, PROCESSED_DIR

# FILE PATHS
GESTURES_JSON = INTERIM_DIR / "gestures.json"
FLAT_CSV = INTERIM_DIR / "gestures_flat.csv"
CLEAN_CSV = INTERIM_DIR / "gestures_clean.csv"
BALANCED_CSV = PROCESSED_DIR / "gestures_balanced.csv"
PIPELINE_PATH = MODELS_DIR / "preprocess_pipeline.joblib"


# TRANSFORMERS


def flatten_and_clean_data() -> pd.DataFrame:
    """
    Load JSON metadata, flatten to DataFrame, and clean the data.
    Returns cleaned DataFrame.
    """
    print("Loading metadata...")
    with GESTURES_JSON.open("r", encoding="utf-8") as f:
        records = json.load(f)

    print("Flattening to DataFrame...")
    df = flatten_metadata(records)
    df.to_csv(FLAT_CSV, index=False)
    print(f"✓ Saved {len(df)} rows → {FLAT_CSV.name}")

    print("Cleaning data...")
    df = clean_dataset(df)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"✓ Saved {len(df)} rows → {CLEAN_CSV.name}")

    return df


def transform_and_balance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing pipeline (encoding, scaling, PCA) and SMOTE balancing.
    Returns balanced DataFrame ready for training.
    """
    print("Applying preprocessing pipeline...")
    y = df.pop("label")

    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(df)

    # Save pipeline for later use in inference
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"✓ Saved pipeline → {PIPELINE_PATH.name}")

    # Create feature names based on PCA components
    n_components = pipeline.named_steps["pca"].n_components_
    feature_names = [f"PC{i + 1}" for i in range(n_components)]
    df_transformed = pd.DataFrame(X_transformed, columns=feature_names)

    print("Applying SMOTE for class balancing...")
    k = max(1, min(5, y.value_counts().min() - 1))
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_resampled, y_resampled = smote.fit_resample(df_transformed, y)  # type: ignore

    df_balanced = pd.concat(
        [
            pd.DataFrame(X_resampled, columns=feature_names),
            y_resampled.reset_index(drop=True).rename("label"),  # type: ignore
        ],
        axis=1,
    )

    df_balanced.to_csv(BALANCED_CSV, index=False)
    print(f"✓ Saved balanced dataset → {BALANCED_CSV.name}")

    return df_balanced


def prepare_data_for_training() -> None:
    """
    Complete preprocessing pipeline: flatten, clean, transform, and balance data.
    """
    print("STEP 1: Flatten and Clean Data")
    df_clean = flatten_and_clean_data()

    print("STEP 2: Transform and Balance Data")
    df_balanced = transform_and_balance_data(df_clean)

    print("Preprocessing Pipeline Complete!")
    print(f"Final dataset: {len(df_balanced)} rows")
    print(f"Ready for training: {BALANCED_CSV}")


def main() -> None:
    prepare_data_for_training()


if __name__ == "__main__":
    main()
