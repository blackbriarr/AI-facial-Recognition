import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(image_size=112, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.10), ratio=(0.3, 3.3), value='random')
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def collect_identity_folders(root_dir):
    root = Path(root_dir)
    folders = [p for p in root.iterdir() if p.is_dir()]
    folders = sorted(folders)
    return folders


def build_small_vggface2_subset(
    root_dir,
    num_identities=250,
    images_per_identity=20,
    val_images_per_identity=4,
    seed=42
):
    random.seed(seed)
    identity_folders = collect_identity_folders(root_dir)

    valid_folders = []
    for folder in identity_folders:
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        if len(imgs) >= images_per_identity:
            valid_folders.append(folder)

    valid_folders = valid_folders[:num_identities]

    train_samples = []
    val_samples = []
    class_to_idx = {}

    for class_idx, folder in enumerate(valid_folders):
        class_to_idx[folder.name] = class_idx
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        imgs = sorted(imgs)
        random.shuffle(imgs)
        imgs = imgs[:images_per_identity]

        val_imgs = imgs[:val_images_per_identity]
        train_imgs = imgs[val_images_per_identity:]

        for img_path in train_imgs:
            train_samples.append((str(img_path), class_idx))

        for img_path in val_imgs:
            val_samples.append((str(img_path), class_idx))

    return train_samples, val_samples, class_to_idx


class FaceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label