import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, PowerTransformer

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


def encode_handedness(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'handedness' column to numeric."""
    df = df.copy()
    df["handedness"] = df["handedness"].map({"left": 0, "right": 1})
    return df


def build_preprocessing_pipeline() -> Pipeline:
    """Create preprocessing steps for hand-gesture features."""
    return Pipeline(
        [
            (
                "handedness_encoding",
                FunctionTransformer(encode_handedness, validate=False),
            ),
            ("power_transform", PowerTransformer(method="yeo-johnson")),
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
        ]
    )
