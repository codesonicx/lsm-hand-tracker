from pathlib import Path

import mediapipe as mp


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


if __name__ == "__main__":
    from src.utils.path_config import MODELS_DIR

    model_path = MODELS_DIR / "hand_landmarker.task"
    landmarker = create_landmarker(model_path)
    print("HandLandmarker created successfully.")
