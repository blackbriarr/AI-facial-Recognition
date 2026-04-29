from pathlib import Path
import time
import json
import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from src.datasets.vggface2_300_dataset import (
    VGGFace2SplitDataset,
    build_train_transform,
    build_eval_transform,
)
from src.models.dinov2_frozen_classifier import DINOv2Classifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dinov2_aggressive_finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SPLIT = SPLIT_ROOT / "vggface2_300_train.txt"
VAL_SPLIT = SPLIT_ROOT / "vggface2_300_val.txt"

NUM_CLASSES = 300
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--train-sample-limit", type=int, default=0)
    p.add_argument("--val-sample-limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def topk_accuracy(logits, labels, k=5):
    k = min(k, logits.shape[1])
    topk = logits.topk(k=k, dim=1).indices
    return topk.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()


def make_subset(ds, limit, seed):
    if limit is None or limit <= 0 or limit >= len(ds):
        return ds
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return Subset(ds, idx[:limit])


def build_loaders(batch_size, num_workers, train_limit, val_limit, seed):
    train_ds = VGGFace2SplitDataset(TRAIN_SPLIT, transform=build_train_transform(IMAGE_SIZE))
    val_ds = VGGFace2SplitDataset(VAL_SPLIT, transform=build_eval_transform(IMAGE_SIZE))

    train_ds = make_subset(train_ds, train_limit, seed)
    val_ds = make_subset(val_ds, val_limit, seed + 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, log_every):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_top5 = 0.0

    for step, (images, labels, _) in enumerate(loader, 1):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc = accuracy(logits.detach(), labels)
        top5 = topk_accuracy(logits.detach(), labels, 5)

        total_loss += loss.item()
        total_acc += acc
        total_top5 += top5

        if step == 1 or step % log_every == 0 or step == len(loader):
            print(
                f"train step {step}/{len(loader)} | "
                f"loss={loss.item():.4f} acc={acc:.4f} top5={top5:.4f} | "
                f"avg_loss={total_loss/step:.4f} avg_acc={total_acc/step:.4f} avg_top5={total_top5/step:.4f}",
                flush=True,
            )

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n, total_top5 / n


@torch.no_grad()
def evaluate(model, loader, criterion, log_every):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_top5 = 0.0

    for step, (images, labels, _) in enumerate(loader, 1):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        acc = accuracy(logits, labels)
        top5 = topk_accuracy(logits, labels, 5)

        total_loss += loss.item()
        total_acc += acc
        total_top5 += top5

        if step == 1 or step % log_every == 0 or step == len(loader):
            print(
                f"val   step {step}/{len(loader)} | "
                f"loss={loss.item():.4f} acc={acc:.4f} top5={top5:.4f} | "
                f"avg_loss={total_loss/step:.4f} avg_acc={total_acc/step:.4f} avg_top5={total_top5/step:.4f}",
                flush=True,
            )

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n, total_top5 / n


def save_checkpoint(path, model, optimizer, epoch, metrics):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "num_classes": NUM_CLASSES,
        },
        path,
    )


def main():
    print("TRAIN SCRIPT STARTED", flush=True)

    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.fast_dev_run:
        args.epochs = 1
        args.batch_size = 16
        args.train_sample_limit = 2000
        args.val_sample_limit = 400
        args.log_every = 10

    train_loader, val_loader = build_loaders(
        args.batch_size,
        args.num_workers,
        args.train_sample_limit,
        args.val_sample_limit,
        args.seed,
    )

    model = DINOv2Classifier(num_classes=NUM_CLASSES, unfreeze_last_n=2).to(DEVICE)

    head_params = list(model.head_params())
    backbone_params = list(model.backbone_params())

    optimizer = AdamW(
        [
            {"params": head_params, "lr": args.head_lr},
            {"params": backbone_params, "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=1)
    cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs - 1, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[1])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"Device: {DEVICE}", flush=True)
    print(f"Train batches: {len(train_loader)}", flush=True)
    print(f"Val batches: {len(val_loader)}", flush=True)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====", flush=True)
        start = time.time()

        train_loss, train_acc, train_top5 = train_one_epoch(
            model, train_loader, optimizer, criterion, args.log_every
        )
        val_loss, val_acc, val_top5 = evaluate(
            model, val_loader, criterion, args.log_every
        )

        scheduler.step()

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_top5": train_top5,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_top5": val_top5,
            "seconds": time.time() - start,
        }
        history.append(metrics)

        save_checkpoint(OUTPUT_DIR / "latest_checkpoint.pt", model, optimizer, epoch, metrics)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(OUTPUT_DIR / "best_checkpoint.pt", model, optimizer, epoch, metrics)

        print(
            f"epoch {epoch} done | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_top5={train_top5:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f} | "
            f"best_val_acc={best_val_acc:.4f}",
            flush=True,
        )

    with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()