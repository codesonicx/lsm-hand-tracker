import math
import uuid
from typing import Any, Dict, Optional

import mediapipe as mp
import numpy as np


def process_one_image(
    letter: str, rgb: Optional[np.ndarray], landmarker
) -> Optional[Dict[str, Any]]:
    """
    Process a single image and return its metadata dict, or None on failure.
    """
    if rgb is None:
        print(f"Error: Image for letter '{letter}' is None.")
        return None

    # Prepare image for MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = landmarker.detect(mp_image)

    if not detection.hand_landmarks:
        print("Mediapipe doesn't detect a hand in the image")
        return None

    # Build metadata & landmark dicts
    height, width = rgb.shape[:2]
    out_name = f"{letter}_{uuid.uuid4().hex[:8]}.png"
    hand_count = len(detection.hand_landmarks)

    # List of handedness strings
    handedness_list = [h[0].category_name.lower() for h in detection.handedness]

    # Map each side to its confidence (or None)
    hand_confidence = {
        "left": next(
            (
                h[0].score
                for h in detection.handedness
                if h[0].category_name.lower() == "left"
            ),
            None,
        ),
        "right": next(
            (
                h[0].score
                for h in detection.handedness
                if h[0].category_name.lower() == "right"
            ),
            None,
        ),
    }

    # Gather raw landmarks per hand
    landmarks_dict = {"left": [], "right": []}
    for h, lm_list in zip(detection.handedness, detection.hand_landmarks):
        side = h[0].category_name.lower()
        landmarks_dict[side] = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in lm_list]

    # Compute engineered features per hand
    engineered = {"left": {}, "right": {}}

    # fingertip and joint indices as per MediaPipe
    fingertip_idxs = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    mcp_idxs = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
    pip_idxs = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

    for side in ("left", "right"):
        lm = landmarks_dict.get(side, [])
        if not lm:
            continue

        # wrist landmark
        w = lm[0]

        # Distances wrist → each fingertip
        dists = {}
        for finger, idx in fingertip_idxs.items():
            tip = lm[idx]
            d = math.sqrt(
                (tip["x"] - w["x"]) ** 2
                + (tip["y"] - w["y"]) ** 2
                + (tip["z"] - w["z"]) ** 2
            )
            dists[f"{finger}_dist"] = d

        # Angles at each MCP joint
        angs = {}
        for finger in mcp_idxs:
            mcp = np.array(
                [
                    lm[mcp_idxs[finger]]["x"],
                    lm[mcp_idxs[finger]]["y"],
                    lm[mcp_idxs[finger]]["z"],
                ]
            )
            pip = np.array(
                [
                    lm[pip_idxs[finger]]["x"],
                    lm[pip_idxs[finger]]["y"],
                    lm[pip_idxs[finger]]["z"],
                ]
            )
            # Vector from MCP → PIP
            v1 = pip - mcp
            # Vector from MCP → wrist
            v2 = np.array([w["x"], w["y"], w["z"]]) - mcp

            # Compute angle (in degrees) between v1 and v2
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
            )
            angle = float(math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0))))
            angs[f"{finger}_angle"] = angle

        engineered[side] = {"distances": dists, "angles": angs}

    # Append the full record
    return {
        "file_name": out_name,
        "label": letter,
        "image_size": [width, height],
        "hand_count": hand_count,
        "handedness": handedness_list,
        "hand_confidence": hand_confidence,
        "landmarks": landmarks_dict,
        "engineered": engineered,
    }
