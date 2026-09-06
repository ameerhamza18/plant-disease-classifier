
import os
import yaml
import torch
import torch.nn as nn
import mlflow
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.dataset import get_dataloaders
from src.transforms import get_train_transform, get_eval_transform
from src.model import build_model, get_trainable_params


def load_config(config_path="config/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    config = load_config()
    torch.manual_seed(config["seed"])

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    image_size = config["data"]["image_size"]
    train_transform = get_train_transform(image_size)
    eval_transform = get_eval_transform(image_size)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        config, train_transform, eval_transform
    )
    print(f"Classes: {len(class_names)}")

    # Model
    model = build_model(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        get_trainable_params(model),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # MLflow setup
    mlflow.set_tracking_uri(f"file:{config['paths']['mlflow_tracking_dir']}")
    mlflow.set_experiment("plant-disease-classifier")

    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0

    with mlflow.start_run():
        mlflow.log_params({
            "architecture": config["model"]["architecture"],
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["data"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "dropout": config["model"]["dropout"],
        })

        for epoch in range(config["training"]["epochs"]):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(f"Epoch {epoch+1}/{config['training']['epochs']} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            # Save only the BEST model, not every epoch
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                }, checkpoint_path)
                print(f"  New best model saved (val_acc={val_acc:.4f})")

        mlflow.log_metric("best_val_acc", best_val_acc)

    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
