from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.datasets.lfw_cw2_dataset import LFWCW2Dataset
from src.models.resnet18_arcface import FaceEmbeddingModel


BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 250


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            _, logits = model(imgs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]
    old_project_root = project_root.parent

    test_split = project_root / "data" / "splits" / "lfw_cw2" / "lfw_cw2_test.txt"
    data_root = old_project_root / "data"
    ckpt_path = project_root / "outputs" / "checkpoints" / "best_lfw_resnet18.pth"

    print(f"Using device: {device}")
    print(f"Using test split: {test_split}")
    print(f"Using data root: {data_root}")
    print(f"Using checkpoint: {ckpt_path}")

    test_ds = LFWCW2Dataset(test_split, data_root=data_root, image_size=IMAGE_SIZE, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FaceEmbeddingModel(num_classes=NUM_CLASSES, embedding_dim=128, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()