FROM python:3.10-slim

# Install OS-level libs that cv2 needs
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy dependency configs first (to leverage Docker cache)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-dev --no-editable

# Copy project files
COPY . .

# Environment variables
ENV LSM_BASE=/app

# Expose FastAPI/Uvicorn port
EXPOSE 8000

# Run FastAPI with uvicorn (pointing to your app inside src/routes.py)
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
