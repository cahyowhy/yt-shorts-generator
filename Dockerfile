FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create directories
RUN mkdir -p /app/data /app/output /app/models

# Expose API port
EXPOSE 8000

# Default command
CMD ["uvicorn", "shortgen.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
