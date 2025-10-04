import cv2
import json
import uuid
import numpy as np
import math
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

def gather_image_records(raw_dir: Path) -> List[Tuple[str, Path]]:
    """
    Scan a directory of letter-folders and return a list of (label, image_path) tuples.
    """
    exts_img = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    records = []
    for letter_dir in sorted(raw_dir.iterdir()):
        if not letter_dir.is_dir():
            continue
        letter = letter_dir.name
        for img_path in sorted(letter_dir.iterdir()):
            if img_path.suffix.lower() in exts_img:
                records.append((letter, img_path))
    print(f"Found {len(records)} images.")
    return records

def create_landmarker(model_path: Path):
    """
    Initialize MediaPipe HandLandmarker from a .task file.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"HandLandmarker model not found at {model_path}")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.55,
    )
    return HandLandmarker.create_from_options(options)

def process_one_image(
        letter: str,
        bgr: Optional[np.ndarray],
        landmarker
) -> Optional[Dict[str, Any]]:
    """
    Process a single image and return its metadata dict, or None on failure.
    """
    if bgr is None:
        print(f"Error: Image for letter '{letter}' is None.")
        return None

    # Prepare image for MediaPipe
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = landmarker.detect(mp_image)

    if not detection.hand_landmarks:
        return None

    # --- Build metadata & landmark dicts ---
    height, width = bgr.shape[:2]
    out_name = f"{letter}_{uuid.uuid4().hex[:8]}.png"
    hand_count = len(detection.hand_landmarks)

    # List of handedness strings
    handedness_list = [h[0].category_name.lower() for h in detection.handedness]

    # Map each side to its confidence (or None)
    hand_confidence = {
        "left": next(
            (h[0].score for h in detection.handedness
            if h[0].category_name.lower() == "left"),
            None
        ),
        "right": next(
            (h[0].score for h in detection.handedness
            if h[0].category_name.lower() == "right"),
            None
        ),
    }

    # Gather raw landmarks per hand
    landmarks_dict = {"left": [], "right": []}
    for h, lm_list in zip(detection.handedness, detection.hand_landmarks):
        side = h[0].category_name.lower()
        landmarks_dict[side] = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in lm_list]

    # --- Compute engineered features per hand ---
    engineered = {"left": {}, "right": {}}

    # fingertip and joint indices as per MediaPipe
    fingertip_idxs = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    mcp_idxs       = {"thumb": 2, "index": 5, "middle": 9,  "ring": 13, "pinky": 17}
    pip_idxs       = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

    for side in ("left", "right"):
        lm = landmarks_dict.get(side, [])
        if not lm:
            continue

        # wrist landmark
        w = lm[0]

        # 1) Distances wrist → each fingertip
        dists = {}
        for finger, idx in fingertip_idxs.items():
            tip = lm[idx]
            d = math.sqrt(
                (tip["x"] - w["x"])**2 +
                (tip["y"] - w["y"])**2 +
                (tip["z"] - w["z"])**2
            )
            dists[f"{finger}_dist"] = d

        # 2) Angles at each MCP joint
        angs = {}
        for finger in mcp_idxs:
            mcp = np.array([lm[mcp_idxs[finger]]["x"],
                            lm[mcp_idxs[finger]]["y"],
                            lm[mcp_idxs[finger]]["z"]])
            pip = np.array([lm[pip_idxs[finger]]["x"],
                            lm[pip_idxs[finger]]["y"],
                            lm[pip_idxs[finger]]["z"]])
            # Vector from MCP → PIP
            v1 = pip - mcp
            # Vector from MCP → wrist
            v2 = np.array([w["x"], w["y"], w["z"]]) - mcp

            # Compute angle (in degrees) between v1 and v2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = float(math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0))))
            angs[f"{finger}_angle"] = angle

        engineered[side] = {
            "distances": dists,
            "angles":    angs
        }


    # Append the full record
    return {
        "file_name":        out_name,
        "label":            letter,
        "image_size":       [width, height],
        "hand_count":       hand_count,
        "handedness":       handedness_list,
        "hand_confidence":  hand_confidence,
        "landmarks":        landmarks_dict,
        "engineered":       engineered
    }

def generate_metadata_from_files(
    image_records: List[Tuple[str, Path]],
    model_path: Path
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Batch-process a list of (label, path) records and return (results, failed_names).
    """

    landmarker = create_landmarker(model_path)
    results: List[Dict[str, Any]] = []
    failed: List[str] = []

    with landmarker:
        print(f"Total images to process: {len(image_records)}")
        for i, (letter, img_path) in enumerate(image_records, start=1):
            print(f"Processing {i}/{len(image_records)}: {img_path.name} ({letter})", end="\r")
            
            bgr = cv2.imread(str(img_path))
            rec = process_one_image(letter, bgr, landmarker)
            if rec:
                results.append(rec)
            else:
                failed.append(img_path.name)

    print(f"\nDone. Processed {len(image_records)} images: {len(results)} successful, {len(failed)} failures.")
    return results, failed

def write_metadata(
    records: List[Dict[str, Any]],
    out_json: Path
) -> None:
    out_json.write_text(
        json.dumps(records, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )
    print(f"Wrote {len(records)} records to {out_json}")

def write_failure_log(
    failed_log: List[str],
    log_path: Path
) -> None:
    log_path.write_text("\n".join(failed_log), encoding="utf-8")
    print(f"Wrote failure log ({len(failed_log)} lines) to {log_path}")


def data_extraction():
    from src.utils.path_config import RAW_DIR, INTERIM_DIR, MODELS_DIR, REPORTS_DIR

    image_records = gather_image_records(RAW_DIR)
    results, failed = generate_metadata_from_files(
        image_records,
        model_path=MODELS_DIR / "hand_landmarker.task"
    )

    write_metadata(results, INTERIM_DIR / "gestures.json")
    write_failure_log(failed, REPORTS_DIR / "failed_images_log.txt")

def main():
    data_extraction()

if __name__ == "__main__":
    main()
