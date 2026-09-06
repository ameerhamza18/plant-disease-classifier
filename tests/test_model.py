
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import build_model, get_trainable_params


def test_model_output_shape():
    """
    The model must output exactly num_classes values per image.
    If someone changes num_classes in config later and forgets to
    check this, this test catches it immediately instead of failing
    silently deep inside training.
    """
    num_classes = 38
    model = build_model(num_classes=num_classes, pretrained=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)  # batch of 1, RGB, 224x224
    output = model(dummy_input)

    assert output.shape == (1, num_classes), f"Expected shape (1, {num_classes}), got {output.shape}"


def test_model_batch_output_shape():
    """Same check, but with a batch of multiple images - not just batch size 1."""
    num_classes = 38
    batch_size = 8
    model = build_model(num_classes=num_classes, pretrained=False)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (batch_size, num_classes)


def test_frozen_layers_are_actually_frozen():
    """
    Confirms the freeze/unfreeze logic in model.py actually works as intended -
    early layers should NOT require gradients, layer4 and fc SHOULD.
    This catches a subtle bug: if this silently breaks, you would not get
    an error - you would just accidentally train the whole network, which
    changes results without any obvious signal something is wrong.
    """
    model = build_model(num_classes=38, pretrained=False)

    # Early layer should be frozen
    assert model.conv1.weight.requires_grad == False

    # layer4 should be trainable
    assert model.layer4[0].conv1.weight.requires_grad == True

    # Final classifier should be trainable
    for param in model.fc.parameters():
        assert param.requires_grad == True


def test_get_trainable_params_excludes_frozen():
    """Confirms get_trainable_params() does not accidentally include frozen params."""
    model = build_model(num_classes=38, pretrained=False)
    trainable_params = get_trainable_params(model)

    for param in trainable_params:
        assert param.requires_grad == True
