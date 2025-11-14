import string
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from src.core.image_loader import (
    process_one_image,
)
from src.core.mediapipe import create_landmarker
from src.core.preprocssing import clean_dataset, flatten_metadata
from src.core.transformations import preprocess_for_inference
from src.model import predict_label_proba
from src.utils.path_config import MODELS_DIR, RAW_DIR

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_LETTERS = set(string.ascii_uppercase) | {"Ñ"}
LANDMARKER = create_landmarker(MODELS_DIR / "hand_landmarker.task")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/process/")
async def process_image(label: str = Form(...), image: UploadFile = File(...)):
    """
    Process an image upload and extract hand landmarks.
    - `label`: The label for the image, must be a single uppercase letter or 'Ñ'.
    - `image`: The image file to process.
    Returns a JSON response with the detected label, predicted label, confidence, and metadata.
    """
    letter = label.strip().upper()
    if letter not in VALID_LETTERS:
        raise HTTPException(400, detail=f"Invalid label: '{label}'")
    content = await image.read()
    pil_img = Image.open(BytesIO(content))
    pil_img = pil_img.convert("RGB")
    rgb = np.array(pil_img)

    if rgb is None:
        raise HTTPException(422, detail="Could not decode image.")
    try:
        metadata = process_one_image(letter, rgb, LANDMARKER)
    except Exception as e:
        raise HTTPException(500, detail=f"Error during detection: {e}")
    if metadata is None:
        raise HTTPException(422, detail="No hands detected in image.")

    df_flat = flatten_metadata([metadata])
    clean_df = clean_dataset(df_flat).drop(columns=["label"])
    X_df = preprocess_for_inference(clean_df)

    X = X_df.values

    pred_label, confidence = predict_label_proba(X)
    print(f"Predicted label: {pred_label}, Confidence: {confidence:.2f}")

    if confidence > 0.85 and pred_label == letter:
        dest_dir = RAW_DIR / letter
    else:
        dest_dir = RAW_DIR / "review" / letter

    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = image.filename if image.filename is not None else "uploaded_image.jpg"
    save_path = dest_dir / filename
    save_path.write_bytes(content)

    return JSONResponse(
        {
            "detected_as": metadata["label"],
            "predicted_as": pred_label,
            "confidence": confidence,
            "metadata": metadata,
        }
    )
