from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LFWCW2Dataset(Dataset):
    def __init__(self, split_file, data_root, image_size=224, augment=False):
        self.split_file = Path(split_file)
        self.data_root = Path(data_root)

        self.samples = []
        with open(self.split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.rsplit(" ", 1)
                self.samples.append((rel_path.replace("\\", "/"), int(label)))

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, rel_path):
        rel_path = rel_path.replace("\\", "/")
        if rel_path.startswith("raw/lfw/"):
            rel_path = rel_path[len("raw/lfw/"):]
        elif rel_path.startswith("lfw/"):
            rel_path = rel_path[len("lfw/"):]

        candidates = [
            self.data_root / rel_path,
            self.data_root / "raw" / "lfw" / rel_path,
            self.data_root.parent / rel_path,
        ]

        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(f"Image not found for '{rel_path}'. Tried: {candidates}")

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = self._resolve_image_path(rel_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)