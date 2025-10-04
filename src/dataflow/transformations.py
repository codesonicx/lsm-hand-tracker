import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, FunctionTransformer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from src.utils.path_config import INTERIM_DIR, MODELS_DIR, PROCESSED_DIR

INPUT_CSV      = INTERIM_DIR    / "gestures_clean.csv"
PIPELINE_FILE  = MODELS_DIR     / "feature_pipeline.joblib"
OUTPUT_CSV     = PROCESSED_DIR  / "gestures_balanced.csv"

def encode_handedness(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X['handedness'] = X['handedness'].map({'left': 0, 'right': 1})
    return X


def transform_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the fitted transformation pipeline to the input DataFrame.
    """
    pipeline = joblib.load(PIPELINE_FILE)
    return pipeline.transform(X)


def prepare_training_dataset():
    # Load & split
    df = pd.read_csv(INPUT_CSV)
    y  = df.pop('label')

    # Build & fit transformation pipeline
    pipeline = Pipeline([
        ('encode', FunctionTransformer(encode_handedness, validate=False)),
        ('pt',     PowerTransformer(method='yeo-johnson')),
        ('scale',  MinMaxScaler()),
        ('pca',    PCA(n_components=0.95, random_state=42)),
    ])
    X_pca = pipeline.fit_transform(df)
    joblib.dump(pipeline, PIPELINE_FILE)

    # Rename PCA columns
    n_pc    = pipeline.named_steps['pca'].n_components_
    columns = [f"PC{i+1}" for i in range(n_pc)]
    df_trans = pd.DataFrame(X_pca, columns=columns)

    # Apply SMOTE if more than one class
    if y.nunique() > 1:
        k = max(1, min(5, y.value_counts().min() - 1))
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = sm.fit_resample(df_trans, y)
        df_out = pd.concat(
            [pd.DataFrame(X_res, columns=columns), 
             y_res.reset_index(drop=True).rename('label')], axis=1
        )
    else:
        df_out = pd.concat([df_trans, y.reset_index(drop=True).rename('label')], axis=1)

    # Save balanced data
    df_out.to_csv(OUTPUT_CSV, index=False)

def main():
    prepare_training_dataset()
if __name__ == "__main__":
    main()
