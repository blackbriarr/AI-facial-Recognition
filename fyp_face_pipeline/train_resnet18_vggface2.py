from pathlib import Path
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.lfw_cw2_dataset import LFWCW2Dataset
from src.models.resnet18_arcface import FaceEmbeddingModel


BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-3
NUM_CLASSES = 250
IMAGE_SIZE = 224
EMBEDDING_DIM = 128


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="valid", leave=False):
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

    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}")

    train_split = project_root / "data" / "splits" / "lfw_cw2" / "lfw_cw2_train.txt"
    val_split = project_root / "data" / "splits" / "lfw_cw2" / "lfw_cw2_val.txt"

    print(f"Train split: {train_split}")
    print(f"Val split: {val_split}")

    if not train_split.exists():
        raise FileNotFoundError(f"Train split file not found: {train_split}")

    if not val_split.exists():
        raise FileNotFoundError(f"Validation split file not found: {val_split}")

    train_ds = LFWCW2Dataset(
        train_split,
        project_root=project_root,
        image_size=IMAGE_SIZE,
        augment=True
    )
    val_ds = LFWCW2Dataset(
        val_split,
        project_root=project_root,
        image_size=IMAGE_SIZE,
        augment=False
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = FaceEmbeddingModel(
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        pretrained=True
    ).to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(
        list(model.embedding_head.parameters()) + list(model.classifier.parameters()),
        lr=LR
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    ckpt_dir = project_root / "outputs" / "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    class_mapping = {str(i): i for i in range(NUM_CLASSES)}
    with open(ckpt_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    print(f"Checkpoint directory: {ckpt_dir}")
    print("Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loader_tqdm = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{EPOCHS}",
            leave=True
        )

        for imgs, labels in train_loader_tqdm:
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
            train_loader_tqdm.set_postfix(
                loss=f"{train_loss:.4f}",
                acc=f"{train_acc:.2f}%"
            )

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        ckpt_path = ckpt_dir / f"resnet18_arcface_epoch{epoch + 1}.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": NUM_CLASSES,
                "embedding_dim": EMBEDDING_DIM,
                "image_size": IMAGE_SIZE,
            },
            ckpt_path
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "embedding_dim": EMBEDDING_DIM,
                    "image_size": IMAGE_SIZE,
                    "best_val_acc": best_val_acc,
                },
                ckpt_dir / "best_lfw_resnet18.pth"
            )
            print(f"Saved best model with val acc: {val_acc:.2f}%")

    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train()