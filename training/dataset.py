# training/dataset.py
import torch
from torchvision import datasets, transforms

IMAGE_SIZE = 224

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def get_dataloaders(data_root="../data", batch_size=16, num_workers=2):
    """
    Expects:
    ../data/train/real, ../data/train/fake
    ../data/val/real,   ../data/val/fake
    ../data/test/real,  ../data/test/fake
    """
    train_tf, eval_tf = get_transforms()

    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(f"{data_root}/val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(f"{data_root}/test",  transform=eval_tf)

    print("Class to index mapping:", train_ds.class_to_idx)
    # Usually: {'fake': 0, 'real': 1}

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_ds
