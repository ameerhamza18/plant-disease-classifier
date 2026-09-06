
import yaml
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.transforms import get_train_transform, get_eval_transform
from src.model import build_model


def load_config(config_path="config/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_on_test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Saves a confusion matrix heatmap. Because there are 38 classes,
    we skip numeric annotations (would be unreadable) and just show
    the color-coded matrix with class names on the axes.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True, include_values=False, cmap="Blues")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    config = load_config()
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint_path = f"{config['paths']['checkpoint_dir']}/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    print(f"Loaded checkpoint with val_acc={checkpoint['val_acc']:.4f}")
    print(f"Number of classes: {len(class_names)}")

    # Rebuild model architecture, then load trained weights into it
    model = build_model(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        pretrained=False,  # we're loading OUR weights, not ImageNet ones
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Data - only need test_loader here, but get_dataloaders returns all three
    image_size = config["data"]["image_size"]
    train_transform = get_train_transform(image_size)
    eval_transform = get_eval_transform(image_size)
    _, _, test_loader, _ = get_dataloaders(config, train_transform, eval_transform)

    # Run evaluation
    y_true, y_pred = evaluate_on_test(model, test_loader, device)

    # Detailed report: precision/recall/F1 PER CLASS, not just overall accuracy
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)

    # Save report to a file so it's part of your portfolio evidence
    with open("experiments/test_classification_report.txt", "w") as f:
        f.write(report)

    print("Report saved to experiments/test_classification_report.txt")

    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path="experiments/confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
