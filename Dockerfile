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
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
