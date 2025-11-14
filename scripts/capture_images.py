# Library to Image processing
import json

# Library to handle file system
import os

# Library to GUI
import tkinter as tk
import uuid
from datetime import datetime, timezone
from tkinter import ttk

import cv2
import mediapipe as mp
from PIL import Image, ImageTk

# Load variables from .env file for local development or get it from the environment variables in production
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    pass

base_path = os.getenv("LSM_BASE")
if not base_path:
    raise ValueError("❌ Environment variable 'LSM_BASE' is not set!")

# Ensure output folders exist
os.makedirs(os.path.join(base_path, "data/images"), exist_ok=True)
os.makedirs(os.path.join(base_path, "data/metadata"), exist_ok=True)


# Setup MediaPipe HandLandmarker
model_path = os.path.join(base_path, "models", "hand_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)
hand_landmarker = HandLandmarker.create_from_options(options)


# Tkinter UI
cap = cv2.VideoCapture(0)
root = tk.Tk()
root.title("LSM Webcam UI")

# LSM: letters that REQUIRE motion
# Sources   J, K, M, Q, X need an explicit wrist/hand motion to be recognized.
#           Z is also drawn in the air as a 'Z'
DYNAMIC_LETTERS = {"J", "K", "M", "Q", "X", "Z"}

# Letter selector
selected_letter = tk.StringVar(value="A")
letter_dropdown = ttk.Combobox(
    root, textvariable=selected_letter, values=[chr(i) for i in range(65, 91)], width=3
)
letter_dropdown.pack(pady=4)

# Webcam preview label
video_lbl = tk.Label(root)
video_lbl.pack()


# Helper: draw connections once per frame (for *all* hands)
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (0, 17),  # palm span
]

HAND_COLORS = {  # consistent color-coding
    "Right": (0, 255, 0),  # BGR green
    "Left": (255, 128, 0),  # BGR orange-blue
}


# Capture-button callback
def capture_frame():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        return

    # Meta & file paths
    image_id = str(uuid.uuid4())
    letter = selected_letter.get()
    gesture_type = "dynamic" if letter in DYNAMIC_LETTERS else "static"
    img_name = f"{image_id}.png"
    img_path = os.path.join(base_path, "data/images", img_name)
    json_path = os.path.join(base_path, "data/metadata", f"{image_id}.json")

    # Save raw image
    cv2.imwrite(img_path, frame)
    print(f"✅ Saved image: {img_path}")

    # Hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(mp_img)

    # Build landmark / confidence dicts
    landmarks_dict = {"right_hand": None, "left_hand": None}
    confidence_dict = {"right_hand": None, "left_hand": None}
    handedness_list = []  # for quick filtering

    if result.hand_landmarks and result.handedness:
        for lm_list, handedness in zip(result.hand_landmarks, result.handedness):
            side = handedness[0].category_name.lower()  # "right"/"left"
            handedness_list.append(side)
            landmarks_dict[f"{side}_hand"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z} for lm in lm_list
            ]
            confidence_dict[f"{side}_hand"] = handedness[0].score

    hand_count = len(handedness_list)

    # Final metadata
    metadata = {
        "image": img_name,
        "letter": letter,
        "gesture_type": gesture_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_size": list(frame.shape[1::-1]),  # [width, height]
        "hand_count": hand_count,
        "handedness_detected": handedness_list,
        "hand_confidence": confidence_dict,
        "landmarks": landmarks_dict,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Log status
    if hand_count:
        print(
            f"📝 Saved metadata for letter '{letter}' "
            f"({', '.join(handedness_list)} hand{'s' if hand_count > 1 else ''})."
        )
    else:
        print(f"⚠️  Saved metadata for letter '{letter}', but **no hand detected**.")


# Capture button
tk.Button(root, text="Capture", command=capture_frame).pack(pady=4)


# Real-time preview with landmark overlay (not saved)
def update_preview():
    ret, frame = cap.read()
    if ret:
        draw = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = hand_landmarker.detect(mp_img)

        if res.hand_landmarks and res.handedness:
            h, w = frame.shape[:2]
            for lm_list, handedness in zip(res.hand_landmarks, res.handedness):
                label = handedness[0].category_name  # "Right"/"Left"
                color = HAND_COLORS[label]
                # draw label at wrist
                wx, wy = int(lm_list[0].x * w), int(lm_list[0].y * h)
                cv2.putText(
                    draw,
                    label,
                    (wx - 10, wy + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                # connections
                for s, e in HAND_CONNECTIONS:
                    x1, y1 = int(lm_list[s].x * w), int(lm_list[s].y * h)
                    x2, y2 = int(lm_list[e].x * w), int(lm_list[e].y * h)
                    cv2.line(draw, (x1, y1), (x2, y2), color, 2)
                # dots
                for lm in lm_list:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(draw, (cx, cy), 4, (255, 255, 255), -1)

        # Show in Tkinter
        tk_img = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
        )
        video_lbl.configure(image=tk_img)
        video_lbl.image = tk_img

    root.after(10, update_preview)


update_preview()
root.mainloop()
cap.release()
