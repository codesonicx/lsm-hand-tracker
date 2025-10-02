import os
from pathlib import Path

# 1) Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# 2) Determine base path
def _get_base_path() -> Path:
    env = os.getenv("LSM_BASE")
    if env:
        return Path(env)
    cwd = Path.cwd()
    print(f"⚠️ Environment variable 'LSM_BASE' not set. Using current directory: {cwd}")
    return cwd

BASE_PATH = _get_base_path()

# 3) Define all directories
EXTERNAL_DIR    = BASE_PATH / "data" / "external"
INTERIM_DIR     = BASE_PATH / "data" / "interim"
PROCESSED_DIR   = BASE_PATH / "data" / "processed"
RAW_DIR         = BASE_PATH / "data" / "raw"
MODELS_DIR      = BASE_PATH / "models"
REPORTS_DIR     = BASE_PATH / "reports"
FIGURES_DIR     = REPORTS_DIR / "figures"

# 4) Ensure they exist
for d in (EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR, RAW_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 5) Registry for lookup
_DIRS = {
    "base":         BASE_PATH,
    "external":     EXTERNAL_DIR,
    "interim":      INTERIM_DIR,
    "processed":    PROCESSED_DIR,
    "raw":          RAW_DIR,
    "models":       MODELS_DIR,
    "reports":      REPORTS_DIR,
    "figures":      FIGURES_DIR,
}

def get_dir(name: str) -> Path:
    """
    Return the Path for one of the keys:
    base, raw, processed, images, videos, not_detected, metadata, models.
    """
    try:
        return _DIRS[name]
    except KeyError:
        valid = ", ".join(_DIRS.keys())
        raise KeyError(f"Unknown directory '{name}'. Valid keys: {valid}")

def set_base_path(path):
    """
    Override BASE_PATH and recompute all directories.
    """
    global BASE_PATH, RAW_DIR, PROCESSED_DIR, IMAGES_DIR
    global VIDEOS_DIR, NOT_DETECTED_DIR, METADATA_DIR, MODELS_DIR, MEDIAPIPE_MODEL, _DIRS

    BASE_PATH       = Path(path)
    EXTERNAL_DIR    = BASE_PATH / "data" / "external"
    INTERIM_DIR     = BASE_PATH / "data" / "interim"
    PROCESSED_DIR   = BASE_PATH / "data" / "processed"
    RAW_DIR         = BASE_PATH / "data" / "raw"
    MODELS_DIR      = BASE_PATH / "models"
    REPORTS_DIR     = BASE_PATH / "reports"
    FIGURES_DIR     = REPORTS_DIR / "figures"

    # Recreate dirs
    for d in (EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR, RAW_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Update registry
    _DIRS = {
    "base":         BASE_PATH,
    "external":     EXTERNAL_DIR,
    "interim":      INTERIM_DIR,
    "processed":    PROCESSED_DIR,
    "raw":          RAW_DIR,
    "models":       MODELS_DIR,
    "reports":      REPORTS_DIR,
    "figures":      FIGURES_DIR,
}
