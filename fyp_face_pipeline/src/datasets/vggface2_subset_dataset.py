from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VGGFace2SubsetDataset(Dataset):
    def __init__(self, root_dir, image_size=224, augment=False):
        self.root_dir = Path(root_dir)

        self.identities = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {p.name: i for i, p in enumerate(self.identities)}

        self.samples = []
        for identity_dir in self.identities:
            label = self.class_to_idx[identity_dir.name]
            for img_path in sorted(identity_dir.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((img_path, label))

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)