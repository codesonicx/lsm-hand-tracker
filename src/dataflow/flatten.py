import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

def load_metadata(metadata_json: Path) -> List[Dict[str, Any]]:
    """
    Load the JSON metadata file and return it as a list of records.
    """
    with open(metadata_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records

def flatten_metadata(records: list[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten the metadata records into a DataFrame with one row per image.
    """

    # Build a flat list of dicts
    flat_rows = []
    expected_fingers = ("thumb", "index", "middle", "ring", "pinky")
    num_landmarks = 21

    for rec in records:
        # Base metadata
        row = {
            "file_name": rec["file_name"],
            "label":     rec["label"],
            "width":     rec["image_size"][0],
            "height":    rec["image_size"][1],
            "hand_count":rec["hand_count"],
            # Handedness as one-hot flags
            "handedness_left": int("left" in rec["handedness"]),
            "handedness_right": int("right" in rec["handedness"]),
            # Confidence per side
            "confidence_left":  rec["hand_confidence"]["left"],
            "confidence_right": rec["hand_confidence"]["right"],
        }

        # Flatten raw landmarks: left_x0,left_y0,left_z0,…, right_x20,right_y20,right_z20
        for side in ("left", "right"):
            lms = rec["landmarks"].get(side, [])
            if lms:
                for idx, lm in enumerate(lms):
                    if idx < num_landmarks: # Ensure we don't go out of bounds if data is malformed
                        row[f"{side}_x{idx}"] = lm.get("x", float("nan"))
                        row[f"{side}_y{idx}"] = lm.get("y", float("nan"))
                        row[f"{side}_z{idx}"] = lm.get("z", float("nan"))

                # Fill any remaining landmark columns for this side with NaN if fewer than num_landmarks were provided       
                for idx in range(len(lms), num_landmarks):
                    row[f"{side}_x{idx}"] = float("nan")
                    row[f"{side}_y{idx}"] = float("nan")
                    row[f"{side}_z{idx}"] = float("nan")

            else: # If no landmarks for this side, explicitly add NaN columns
                for idx in range(num_landmarks):
                    row[f"{side}_x{idx}"] = float("nan")
                    row[f"{side}_y{idx}"] = float("nan")
                    row[f"{side}_z{idx}"] = float("nan")

        # Flatten engineered distances and angles
        for side_key in ("left", "right"): # Iterate over expected sides
            feats = rec["engineered"].get(side_key) # Get can return None

            if not feats: # Covers if side_key was missing or if feats was an empty dict/list
                for finger in expected_fingers:
                    row[f"{side_key}_{finger}_dist"]  = float("nan")
                    row[f"{side_key}_{finger}_angle"] = float("nan")
            else:
                # Distances Data
                actual_distances = feats.get("distances", {})
                for finger in expected_fingers:
                    # Use .get with a default of NaN for each specific distance name (e.g., "thumb_dist")
                    row[f"{side_key}_{finger}_dist"] = actual_distances.get(f"{finger}_dist", float("nan"))

                # Angles Data
                actual_angles = feats.get("angles", {})
                for finger in expected_fingers:
                    # Use .get with a default of NaN for each specific angle name (e.g., "thumb_angle")
                    row[f"{side_key}_{finger}_angle"] = actual_angles.get(f"{finger}_angle", float("nan"))
        
        flat_rows.append(row)

    return pd.DataFrame(flat_rows)


def write_flat_csv(
    df: pd.DataFrame,
    flattened_csv: Path
) -> None:
    
    """Write the flattened DataFrame back out to CSV."""
    df.to_csv(flattened_csv, index=False)
    print(f"Wrote flat CSV with {len(df)} rows to {flattened_csv}")


def flatten_local_images():
    from utils.path_config import INTERIM_DIR
    records = load_metadata(INTERIM_DIR / "gestures.json")
    df = flatten_metadata(records)
    write_flat_csv(df, INTERIM_DIR / "gestures_flat.csv")

def main():
    flatten_local_images()
    
if __name__ == "__main__":
    main()