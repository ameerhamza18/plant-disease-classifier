
import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.predict import load_config, load_model_for_inference, predict_image

app = FastAPI(
    title="Plant Disease Classifier API",
    description="Upload a leaf image to detect plant disease using a ResNet18 transfer learning model.",
    version="1.0.0",
)

# Allow requests from any origin - needed since the demo frontend
# (if any) will run on a different domain than this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model ONCE at startup, not per-request - loading from disk on
# every single API call would make the API unusably slow.
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names = load_model_for_inference(config, device)


@app.get("/")
def root():
    return {
        "message": "Plant Disease Classifier API is running.",
        "num_classes": len(class_names),
        "device": str(device),
    }


@app.get("/health")
def health_check():
    """Basic health check - useful for deployment platforms to verify the service is alive."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, returns top-3 predicted diseases
    with confidence scores.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = predict_image(
        model, class_names, image,
        image_size=config["data"]["image_size"],
        device=device,
        top_k=3,
    )

    return {
        "filename": file.filename,
        "predictions": results,
    }
