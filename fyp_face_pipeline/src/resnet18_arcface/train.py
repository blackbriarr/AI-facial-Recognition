import os
import csv
import json
import time
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]      # ...\fyp_face_pipeline
OUTER_PROJECT_ROOT = PROJECT_ROOT.parent                # ...\AI-facial-Recognition

RAW_VGGFACE2 = OUTER_PROJECT_ROOT / "data" / "vggface2"
SUBSET_ROOT = OUTER_PROJECT_ROOT / "data" / "vggface2_subset_250"

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT =", PROJECT_ROOT, flush=True)
print("OUTER_PROJECT_ROOT =", OUTER_PROJECT_ROOT, flush=True)
print("RAW_VGGFACE2 =", RAW_VGGFACE2, flush=True)
print("SUBSET_ROOT =", SUBSET_ROOT, flush=True)
print("CHECKPOINT_DIR =", CHECKPOINT_DIR, flush=True)
print("REPORT_DIR =", REPORT_DIR, flush=True)
print("LOG_DIR =", LOG_DIR, flush=True)


NUM_IDENTITIES = 250
IMAGES_PER_IDENTITY = 20
VAL_IMAGES_PER_IDENTITY = 4

IMAGE_SIZE = 224
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def find_identity_dirs(root: Path):
    dirs = []
    for p in root.rglob("*"):
        if p.is_dir():
            imgs = list(p.glob("*.jpg")) + list(p.glob("*.jpeg")) + list(p.glob("*.png"))
            if len(imgs) >= IMAGES_PER_IDENTITY:
                dirs.append(p)
    return sorted(dirs)


def build_subset():
    if not RAW_VGGFACE2.exists():
        raise RuntimeError(f"Source dataset folder does not exist: {RAW_VGGFACE2}")

    if SUBSET_ROOT.exists():
        shutil.rmtree(SUBSET_ROOT)
    SUBSET_ROOT.mkdir(parents=True, exist_ok=True)

    identity_dirs = find_identity_dirs(RAW_VGGFACE2)
    print(f"Found {len(identity_dirs)} candidate identity folders", flush=True)

    if not identity_dirs:
        raise RuntimeError(f"No identity folders found under {RAW_VGGFACE2}")

    selected = identity_dirs[:NUM_IDENTITIES]
    if len(selected) < NUM_IDENTITIES:
        print(f"Warning: only {len(selected)} identities found, using all of them", flush=True)

    class_to_idx = {}
    train_rows = []
    val_rows = []

    for idx, src_dir in enumerate(selected):
        class_name = src_dir.name if src_dir.name else f"class_{idx:04d}"
        class_to_idx[class_name] = idx

        imgs = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.png"))
        imgs = sorted(imgs)
        random.shuffle(imgs)
        imgs = imgs[:IMAGES_PER_IDENTITY]

        if len(imgs) < 5:
            continue

        val_imgs = imgs[:VAL_IMAGES_PER_IDENTITY]
        train_imgs = imgs[VAL_IMAGES_PER_IDENTITY:]

        train_dst = SUBSET_ROOT / "train" / class_name
        val_dst = SUBSET_ROOT / "val" / class_name
        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            dst = train_dst / img.name
            shutil.copy2(img, dst)
            train_rows.append((str(dst), idx))

        for img in val_imgs:
            dst = val_dst / img.name
            shutil.copy2(img, dst)
            val_rows.append((str(dst), idx))

    return train_rows, val_rows, class_to_idx


class FaceDataset(Dataset):
    def __init__(self, samples, image_size=224, augment=False):
        self.samples = samples

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
        return image, torch.tensor(label, dtype=torch.long), img_path


class FaceEmbeddingModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=128, pretrained=False):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_head = nn.Linear(512, embedding_dim)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.backbone(x).flatten(1)
        embeddings = self.embedding_head(features)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        logits = self.classifier(features)
        return embeddings, logits


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_paths = []

    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            embeddings, logits = model(imgs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_paths.extend(list(paths))

    avg_loss = running_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)

    return avg_loss, acc, all_labels, all_preds, all_paths


def save_epoch_table(rows, csv_path, json_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "is_best"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def save_confusion_matrix_csv(cm, class_names, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row.tolist())


def save_predictions_csv(paths, labels, preds, class_names, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label_idx", "true_label_name", "pred_label_idx", "pred_label_name"])
        for p, y, yhat in zip(paths, labels, preds):
            writer.writerow([p, y, class_names[y], yhat, class_names[yhat]])


def main():
    print("Building subset...", flush=True)
    train_rows, val_rows, class_to_idx = build_subset()

    if not train_rows or not val_rows:
        raise RuntimeError("Subset build failed; no train/val samples created.")

    class_names = list(class_to_idx.keys())

    print(f"Train samples: {len(train_rows)}", flush=True)
    print(f"Val samples: {len(val_rows)}", flush=True)
    print(f"Classes: {len(class_to_idx)}", flush=True)

    with open(LOG_DIR / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    train_ds = FaceDataset(train_rows, image_size=IMAGE_SIZE, augment=True)
    val_ds = FaceDataset(val_rows, image_size=IMAGE_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model = FaceEmbeddingModel(
        num_classes=len(class_to_idx),
        embedding_dim=EMBEDDING_DIM,
        pretrained=False
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = -1

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")

        for imgs, labels, _ in bar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            embeddings, logits = model(imgs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            bar.set_postfix(
                loss=f"{running_loss / max(total, 1):.4f}",
                acc=f"{100.0 * correct / max(total, 1):.2f}%"
            )

        train_loss = running_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)

        val_loss, val_acc, val_labels, val_preds, val_paths = evaluate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        is_best = False
        if (val_loss < best_val_loss) or (abs(val_loss - best_val_loss) < 1e-8 and val_acc > best_val_acc):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            is_best = True

            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_lfw_resnet18.pth")

            cm = confusion_matrix(val_labels, val_preds, labels=list(range(len(class_names))))
            save_confusion_matrix_csv(cm, class_names, REPORT_DIR / "best_confusion_matrix.csv")

            report = classification_report(
                val_labels,
                val_preds,
                labels=list(range(len(class_names))),
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            with open(REPORT_DIR / "best_classification_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            save_predictions_csv(
                val_paths,
                val_labels,
                val_preds,
                class_names,
                REPORT_DIR / "best_predictions.csv"
            )

            with open(REPORT_DIR / "best_run_summary.json", "w", encoding="utf-8") as f:
                json.dump({
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "num_classes": len(class_names),
                    "train_samples": len(train_rows),
                    "val_samples": len(val_rows),
                    "image_size": IMAGE_SIZE,
                    "embedding_dim": EMBEDDING_DIM,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "learning_rate": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "checkpoint_path": str(CHECKPOINT_DIR / "best_lfw_resnet18.pth")
                }, f, indent=2)

        torch.save(model.state_dict(), CHECKPOINT_DIR / "last_lfw_resnet18.pth")

        history.append({
            "epoch": epoch,
            "lr": round(current_lr, 8),
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 4),
            "is_best": is_best
        })

        save_epoch_table(
            history,
            LOG_DIR / "training_results.csv",
            LOG_DIR / "training_results.json"
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Best Epoch: {best_epoch}",
            flush=True
        )

        scheduler.step()

    elapsed = time.time() - start_time

    with open(REPORT_DIR / "final_training_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "total_time_minutes": round(elapsed / 60.0, 2)
        }, f, indent=2)

    print(f"Training complete in {elapsed / 60:.2f} minutes", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best validation loss: {best_val_loss:.4f}", flush=True)
    print(f"Best validation accuracy: {best_val_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()