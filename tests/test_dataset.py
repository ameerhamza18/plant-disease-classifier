
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.transforms import get_train_transform, get_eval_transform
import torch


def test_train_transform_output_shape():
    """
    Confirms transforms actually produce the expected tensor shape.
    Catches config mistakes like a wrong image_size silently propagating.
    """
    from PIL import Image
    import numpy as np

    dummy_image = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    transform = get_train_transform(image_size=224)
    result = transform(dummy_image)

    assert result.shape == (3, 224, 224), f"Expected (3, 224, 224), got {result.shape}"


def test_eval_transform_output_shape():
    from PIL import Image
    import numpy as np

    dummy_image = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    transform = get_eval_transform(image_size=224)
    result = transform(dummy_image)

    assert result.shape == (3, 224, 224)


def test_eval_transform_is_deterministic():
    """
    Eval transform must NOT have randomness - the same image must produce
    the exact same tensor every time. This directly guards against
    accidentally leaving augmentation in the eval pipeline, which would
    make your test/val accuracy numbers not reproducible.
    """
    from PIL import Image
    import numpy as np

    dummy_image = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    transform = get_eval_transform(image_size=224)

    result1 = transform(dummy_image)
    result2 = transform(dummy_image)

    assert torch.allclose(result1, result2), "Eval transform should be deterministic but produced different outputs"
