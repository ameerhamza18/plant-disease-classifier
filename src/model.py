
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes, dropout=0.3, pretrained=True):
    """
    Load ResNet18 pretrained on ImageNet, freezes the early layers,
    and replaces the final classification layer to output num_classes
    instead of ImageNet's 1000 classes.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze all layers first — I don't want to retrain everything from
    # scratch, since the early layers already know how to detect edges,
    # textures, colors etc. from ImageNet training.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last residual block (layer4) so the model can adapt
    # its higher-level features to leaves/disease patterns specifically,
    # not just reuse generic ImageNet features untouched.
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully-connected layer.
    # Original: model.fc maps 512 features -> 1000 ImageNet classes.
    # We replace it with a small head mapping 512 -> num_classes,
    # with dropout to reduce overfitting since our dataset is smaller
    # than ImageNet.
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model


def get_trainable_params(model):
    """Returns only the parameters that require gradients (for the optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]
