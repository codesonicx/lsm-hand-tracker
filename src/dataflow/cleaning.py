import pandas as pd
import numpy as np
from pathlib import Path

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with no predictive value based on EDA.
    """
    return df.drop(columns=["file_name", "width", "height"])

def mark_preferred_hand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing confidence values, then add a column indicating
    which hand (left/right) has higher confidence.
    """
    df = df.copy()
    df["confidence_left"] = df["confidence_left"].fillna(0)
    df["confidence_right"] = df["confidence_right"].fillna(0)
    df["preferred_hand"] = np.where(
        df["confidence_left"] >= df["confidence_right"],
        "left",
        "right"
    )
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame selecting the more confident hand's landmarks, distances, and angles.
    """
    mask_left = df["preferred_hand"] == "left"

    clean_df = pd.DataFrame({
        "label": df["label"],
        "hand_count": df["hand_count"],
        "handedness": df["preferred_hand"],
        "confidence": df["confidence_left"].where(mask_left, df["confidence_right"])
    })

    # landmarks    
    for axis in ("x", "y", "z"):
        for i in range(21):
            left_col  = f"left_{axis}{i}"
            right_col = f"right_{axis}{i}"
            output_col   = f"{axis}{i}"
            clean_df[output_col] = np.where(
                df["preferred_hand"] == "left",
                df[left_col],
                df[right_col]
            )

    # engineered distances and angles
    fingers = ["thumb","index","middle","ring","pinky"]
    for finger in fingers:
        clean_df[f"{finger}_dist"]  = np.where(
            df["preferred_hand"] == "left",
            df[f"left_{finger}_dist"],
            df[f"right_{finger}_dist"]
        )
        clean_df[f"{finger}_angle"] = np.where(
            df["preferred_hand"] == "left",
            df[f"left_{finger}_angle"],
            df[f"right_{finger}_angle"]
        )

    return clean_df

def removing_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not useful for training the model.
    This includes 'confidence' and 'hand_count' as they don't provide enough variance
    """
    
    # Droping confidence because our dataset doesn't contain so much variance in this column but in the future is possible to recover it
    # 'hand_count' has a similar case, where our model is just trained with one hand, if we solve this issue in the future we can include it again
    return df.drop(columns=["confidence", "hand_count"])

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by filling missing values, extracting features to work just with one hand,
    and removing unnecessary columns.
    """
    df = df.copy()
    df = drop_unused_columns(df)
    df = mark_preferred_hand(df)
    df = extract_features(df)
    df = removing_extra_features(df)
    return df

def load_dataset(
    input_path: Path
) -> pd.DataFrame:
    """
    Load a CSV from disk into a DataFrame.
    """
    return pd.read_csv(input_path)


def save_dataset(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save a DataFrame to CSV on disk.
    """
    df.to_csv(output_path, index=False)
    print(f"Wrote cleaned data with {len(df)} rows to {output_path}")

def clean_local_dataset():
    from utils.path_config import INTERIM_DIR
    input_path = INTERIM_DIR / "gestures_flat.csv"
    output_path = INTERIM_DIR / "gestures_cleaned.csv"

    df = load_dataset(input_path)
    df = clean_dataset(df)
    save_dataset(df, output_path)

def main():
    clean_local_dataset()
    

if __name__ == "__main__":
    main()
