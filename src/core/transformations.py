import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils.path_config import MODELS_DIR

PIPELINE_PATH = MODELS_DIR / "preprocess_pipeline.joblib"


# INFERENCE TRANSFORM
def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Load saved preprocessing pipeline and transform new samples."""
    pipeline: Pipeline = joblib.load(PIPELINE_PATH)
    X_transformed = pipeline.transform(df)

    n_components = pipeline.named_steps["pca"].n_components_
    feature_names = [f"PC{i + 1}" for i in range(n_components)]

    return pd.DataFrame(X_transformed, columns=feature_names)
