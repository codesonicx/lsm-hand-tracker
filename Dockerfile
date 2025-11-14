# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Setup non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

WORKDIR /app

# Compile bytecode for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy lockfile and pyproject first (for caching)
COPY pyproject.toml uv.lock ./

# Install only prod dependencies first
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy only necessary application files
COPY src/app.py src/app.py
COPY src/model.py src/model.py
COPY src/core/ src/core/
COPY src/utils/ src/utils/
COPY models/ models/

# Create directories and change ownership to nonroot
RUN mkdir -p data/external data/interim data/processed data/raw \
             models figures reports \
    && chown -R nonroot:nonroot /app

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Put venv on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Switch to non-root
USER nonroot

# Expose FastAPI port
EXPOSE 8000

# Run in production mode with uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]