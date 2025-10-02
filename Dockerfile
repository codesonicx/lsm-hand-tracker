FROM python:3.10-slim

# Install OS-level libs that cv2 needs (libGL.so.1)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Directory of workspace
WORKDIR /app

# install UV for dependency management
RUN pip install --no-cache-dir uv

# Copy the configuration for dependency to cache the layer
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY models ./models

# Sync and install dependencies + the package (editable)
RUN uv sync --locked --no-dev --no-editable

# Copy the rest of the code
COPY . .

# Environment variable for your application
ENV LSM_BASE=/app

# Expose the Uvicorn port
EXPOSE 8000

# Start with uv run, which automatically activates the environment
# shell form so $PORT is substituted at runtime
CMD uv run uvicorn lsm_hand_tracker.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000}
