from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.lfw_cw2_dataset import LFWCW2Dataset
from src.models.resnet18_arcface import FaceEmbeddingModel


BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 250
IMAGE_SIZE = 224


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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parents[2]
    old_project_root = project_root.parent

    train_split = project_root / "data" / "splits" / "lfw_cw2" / "lfw_cw2_train.txt"
    val_split = project_root / "data" / "splits" / "lfw_cw2" / "lfw_cw2_val.txt"

    data_root = old_project_root / "data"

    print(f"Using data root: {data_root}")
    print(f"Using train split: {train_split}")
    print(f"Using val split: {val_split}")

    train_ds = LFWCW2Dataset(train_split, data_root=data_root, image_size=IMAGE_SIZE, augment=True)
    val_ds = LFWCW2Dataset(val_split, data_root=data_root, image_size=IMAGE_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FaceEmbeddingModel(num_classes=NUM_CLASSES, embedding_dim=128, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            _, logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_dir = project_root / "outputs" / "checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_lfw_resnet18.pth")
            print(f"Saved best model with val acc: {val_acc:.2f}%")

    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train()