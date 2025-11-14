import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from src.core.image_loader import process_one_image
from src.core.mediapipe import create_landmarker


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
    print(f"Total images found: {len(records)}")
    return records


def generate_metadata_from_files(
    image_records: List[Tuple[str, Path]], model_path: Path
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
            print(
                f"Processing {i}/{len(image_records)}: {img_path.name} ({letter})",
                end="\r",
            )

            pil_img = Image.open(img_path).convert("RGB")
            rgb = np.array(pil_img)
            rec = process_one_image(letter, rgb, landmarker)
            if rec:
                results.append(rec)
            else:
                failed.append(img_path.name)

    print(
        f"\nDone. Processed {len(image_records)} images: {len(results)} successful, {len(failed)} failures."
    )
    return results, failed


class FileWriter:
    @staticmethod
    def json(data, path: Path):
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8"
        )
        print(f"Saved {len(data)} Json items to {path}")

    @staticmethod
    def text(lines, path: Path):
        text = "\n".join(lines) if isinstance(lines, list) else str(lines)
        path.write_text(text, encoding="utf-8")
        print(f"Saved {len(lines)} text lines to {path}")


def load_metadata(metadata_json: Path) -> List[Dict[str, Any]]:
    """
    Load the JSON metadata file and return it as a list of records.
    """
    with open(metadata_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records


def data_extraction():
    from src.utils.path_config import INTERIM_DIR, MODELS_DIR, RAW_DIR, REPORTS_DIR

    print("Data extraction process initiated")
    print(f"Collecting images from the subfolders located in: {RAW_DIR}")
    image_records = gather_image_records(RAW_DIR)
    results, failed = generate_metadata_from_files(
        image_records, model_path=MODELS_DIR / "hand_landmarker.task"
    )
    print("\nSaving results and errors of mediapipe locally on disk")
    FileWriter.json(results, INTERIM_DIR / "gestures.json")
    FileWriter.text(failed, REPORTS_DIR / "failed_images_log.txt")


def main():
    data_extraction()


if __name__ == "__main__":
    main()
