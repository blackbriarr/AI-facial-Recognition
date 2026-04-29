from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

THIS_FILE = Path(__file__).resolve()
PIPELINE_ROOT = THIS_FILE.parents[2]
REPO_ROOT = THIS_FILE.parents[3]
DATASET_ROOT = REPO_ROOT / "data" / "vggface2"


class VGGFace2SplitDataset(Dataset):
    def __init__(self, split_file, transform=None):
        self.split_file = Path(split_file).resolve()
        self.transform = transform
        self.samples = []

        with open(self.split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Invalid line in {self.split_file}: {line}")
                label = int(parts[-1])
                img_path = " ".join(parts[:-1])
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, img_path_str):
        raw = Path(img_path_str)
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(PIPELINE_ROOT / raw)
            candidates.append(REPO_ROOT / raw)
            parts = raw.parts
            if len(parts) >= 2 and parts[0].lower() == "data" and parts[1].lower() == "vggface2":
                candidates.append(DATASET_ROOT / Path(*parts[2:]))
            elif len(parts) >= 1 and parts[0].lower() == "vggface2":
                candidates.append(DATASET_ROOT / Path(*parts[1:]))
            else:
                candidates.append(DATASET_ROOT / raw)

        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"Could not find image: {img_path_str}\nTried:\n" + "\n".join(str(x) for x in candidates))

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        resolved = self._resolve_image_path(img_path)
        img = Image.open(resolved).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, str(resolved)


def build_train_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])