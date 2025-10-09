import cv2
import mediapipe as mp
import json
import math
import numpy as np
from pathlib import Path
from utils.path_config import RAW_DIR, INTERIM_DIR


def find_videos(base_path: Path, extensions=(".mp4", ".avi", ".mov")):
    """Recursively search for all video files under base_path."""
    return [p for p in base_path.rglob("*") if p.suffix.lower() in extensions]


def compute_engineered_features(landmarks_dict):
    """Compute distances and angles for each hand."""
    engineered = {"left": {}, "right": {}}

    fingertip_idxs = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    mcp_idxs = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
    pip_idxs = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

    for side in ("left", "right"):
        lm = landmarks_dict.get(side, [])
        if not lm:
            continue

        w = lm[0]  # wrist

        # Distances wrist → fingertip
        dists = {}
        for finger, idx in fingertip_idxs.items():
            tip = lm[idx]
            d = math.sqrt(
                (tip["x"] - w["x"]) ** 2 +
                (tip["y"] - w["y"]) ** 2 +
                (tip["z"] - w["z"]) ** 2
            )
            dists[f"{finger}_dist"] = d

        # Angles at MCP joints
        angs = {}
        for finger in mcp_idxs:
            mcp = np.array([lm[mcp_idxs[finger]]["x"],
                            lm[mcp_idxs[finger]]["y"],
                            lm[mcp_idxs[finger]]["z"]])
            pip = np.array([lm[pip_idxs[finger]]["x"],
                            lm[pip_idxs[finger]]["y"],
                            lm[pip_idxs[finger]]["z"]])
            v1 = pip - mcp
            v2 = np.array([w["x"], w["y"], w["z"]]) - mcp
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = float(math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0))))
            angs[f"{finger}_angle"] = angle

        engineered[side] = {"distances": dists, "angles": angs}

    return engineered


def extract_hand_landmarks(video_path: Path, hands, frame_skip: int = 5):
    """Extract rich hand landmarks from a video using an existing MediaPipe Hands instance."""
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    all_frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                height, width = frame.shape[:2]
                frame_data = {
                    "frame": frame_idx,
                    "image_size": [width, height],
                    "hand_count": len(results.multi_hand_landmarks),
                    "handedness": [],
                    "hand_confidence": {"left": None, "right": None},
                    "landmarks": {"left": [], "right": []},
                    "engineered": {"left": {}, "right": {}}
                }

                # collect handedness + landmarks
                for h, lm_list in zip(results.multi_handedness, results.multi_hand_landmarks):
                    side = h.classification[0].label.lower()
                    score = h.classification[0].score
                    frame_data["handedness"].append(side)
                    frame_data["hand_confidence"][side] = float(score)
                    frame_data["landmarks"][side] = [
                        {"x": lm.x, "y": lm.y, "z": lm.z} for lm in lm_list.landmark
                    ]

                # compute engineered features
                frame_data["engineered"] = compute_engineered_features(frame_data["landmarks"])
                all_frames_data.append(frame_data)

        frame_idx += 1

    cap.release()
    return all_frames_data


def build_dataset_json(base_dir: Path, output_json: Path):
    """Scan all videos and process them with MediaPipe."""
    videos = find_videos(base_dir)
    dataset = []

    print(f"Found {len(videos)} videos to process.\n")

    mp_hands = mp.solutions.hands  # type: ignore
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:
        for i, video_path in enumerate(videos, start=1):
            print(f"[{i}/{len(videos)}] Processing: {video_path.name}")
            frames = extract_hand_landmarks(video_path, hands)
            label = video_path.parent.name  # Target label for classification
            dataset.append({
                "file_name": video_path.name,
                "label": label,
                "frame_count": len(frames),
                "frames": frames
            })
    finally:
        hands.close()

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"\n✅ Dataset saved to {output_json}")


if __name__ == "__main__":
    build_dataset_json(RAW_DIR, INTERIM_DIR / "hand_landmarks_dataset.json")
