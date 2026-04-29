from pathlib import Path
import os
import csv
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.lfw_cw2_dataset import LFWCW2Dataset
from src.models.resnet18_arcface import FaceEmbeddingModel


BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
NUM_CLASSES = 250
IMAGE_SIZE = 224


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

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

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    return running_loss / total, 100.0 * correct / total, all_labels, all_preds


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parents[1]

    train_split = project_root / "data" / "splits" / "lfw_cw2" / "lfw_cw2_train.txt"
    val_split = project_root / "data" / "splits" / "lfw_cw2" / "lfw_cw2_val.txt"
    data_root = project_root / "data"

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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    outputs_dir = project_root / "outputs"
    checkpoints_dir = outputs_dir / "checkpoints"
    reports_dir = outputs_dir / "reports"
    logs_dir = outputs_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    start_time = time.time()

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
        val_loss, val_acc, val_labels, val_preds = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoints_dir / "best_lfw_resnet18.pth")

            with open(reports_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                        "best_val_loss": val_loss,
                        "train_acc": train_acc,
                        "train_loss": train_loss
                    },
                    f,
                    indent=2
                )

            with open(reports_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["true", "pred"])
                for t, p in zip(val_labels, val_preds):
                    writer.writerow([t, p])

        torch.save(model.state_dict(), checkpoints_dir / "last_lfw_resnet18.pth")

        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best": is_best
        })

        with open(logs_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        with open(logs_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        scheduler.step()

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed/60:.1f} min. Best Val Acc: {best_val_acc:.2f}%")
    print(f"Best epoch: {best_epoch}")


if __name__ == "__main__":
    train()