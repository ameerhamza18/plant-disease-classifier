
import os
import yaml
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch


def load_config(config_path="config/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_full_dataset(raw_dir, transform):
    dataset = ImageFolder(root=raw_dir, transform=transform)
    return dataset


def split_dataset(dataset, val_split, test_split, seed):
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    return train_ds, val_ds, test_ds


def get_dataloaders(config, train_transform, eval_transform):

    raw_dir = config["data"]["raw_dir"]
    seed = config["seed"]

    
    full_dataset_train_view = ImageFolder(root=raw_dir, transform=train_transform)
    full_dataset_eval_view = ImageFolder(root=raw_dir, transform=eval_transform)

    class_names = full_dataset_train_view.classes

    train_ds, _, _ = split_dataset(
        full_dataset_train_view,
        config["data"]["val_split"],
        config["data"]["test_split"],
        seed,
    )
    _, val_ds, test_ds = split_dataset(
        full_dataset_eval_view,
        config["data"]["val_split"],
        config["data"]["test_split"],
        seed,
    )

    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names
