
import yaml
import torch
from PIL import Image

from src.transforms import get_eval_transform
from src.model import build_model


def load_config(config_path="config/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_for_inference(config, device):
    """
    Loads the trained model + class names from the saved checkpoint.
    This is the ONLY function that should know about the checkpoint file -
    both predict.py's CLI use and app/main.py's API use will call this
    same function, so there's one source of truth for "how to load the model."
    """
    checkpoint_path = f"{config['paths']['checkpoint_dir']}/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_model(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


def predict_image(model, class_names, image, image_size, device, top_k=3):
    """
    Takes a PIL Image (already loaded) and returns top_k predictions
    with confidence scores, not just the single best guess.
    """
    transform = get_eval_transform(image_size)
    image_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class": class_names[idx.item()],
            "confidence": round(prob.item(), 4)
        })

    return results


def predict_from_path(image_path, config=None, device=None):
    """Convenience function for CLI/script use - handles loading everything from a file path."""
    if config is None:
        config = load_config()
    if device is None:
        device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    model, class_names = load_model_for_inference(config, device)
    image = Image.open(image_path).convert("RGB")

    return predict_image(model, class_names, image, config["data"]["image_size"], device)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    results = predict_from_path(image_path)

    print(f"\nPredictions for {image_path}:\n")
    for r in results:
        print(f"  {r['class']:50s} confidence: {r['confidence']:.2%}")
