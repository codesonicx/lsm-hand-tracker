import joblib
import numpy as np
from typing import Tuple, Any

from utils.path_config import MODELS_DIR
MODEL_FILE = MODELS_DIR / "gesture_classifier.joblib"

_model: Any = None

def get_model() -> Any:
    """
    Lazily load and cache the classifier.
    Raises RuntimeError with a clear message if loading fails.
    """
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_FILE)
        except FileNotFoundError as e:
            raise RuntimeError(f"Model file not found at {MODEL_FILE}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e
    return _model


def predict_label_proba(X: np.ndarray) -> Tuple[str, float]:
    """
    Predict the label and confidence for a single feature vector.

    Parameters:
        X: array-like of shape (n_features,) or (1, n_features)

    Returns:
        (best_label, confidence) where best_label is the class with highest
        probability and confidence is its predicted probability.
    """
    model = get_model()

    # Ensure we have a (1, n_features) 2D array
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2 or arr.shape[0] != 1:
        raise ValueError(
            f"predict_label_proba expects a single sample; got array with shape {arr.shape}"
        )

    # Predict probabilities
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not implement predict_proba()")
    probs = model.predict_proba(arr)[0]

    # Get classes_
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise RuntimeError("Loaded model has no attribute 'classes_'")

    best_idx = int(np.argmax(probs))
    return classes[best_idx], float(probs[best_idx])

if __name__ == "__main__":
    # Simple test to ensure the model can be loaded and used
    try:
        model = get_model()
        print(f"Model loaded successfully: {model}")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
    
    # Example usage with dummy data
    example_data = np.random.rand(1, 15)  # Assuming 15 features
    label, confidence = predict_label_proba(example_data)
    print(f"Predicted label: {label}, Confidence: {confidence:.2f}")
