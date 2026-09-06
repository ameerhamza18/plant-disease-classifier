FROM python:3.11-slim

WORKDIR /code

# Install system dependencies needed by torch/torchvision/pillow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first - this is a Docker caching trick:
# if requirements.txt doesn't change, Docker reuses the cached layer
# instead of reinstalling everything on every build, making rebuilds
# much faster during development.
COPY requirements.txt .

# Install CPU-only torch/torchvision first, from PyTorch's dedicated
# CPU wheel index - this avoids pulling the CUDA build (2-3GB) since
# this container only runs inference, never training, and most
# deployment platforms don't have GPUs anyway.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install everything else normally
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
