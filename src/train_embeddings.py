import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.lfw_cw2_dataset import LFWCW2Dataset
from src.model_embeddings import FaceEmbeddingModel


# -------------------
# Config
# -------------------
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 250
EMBED_DIM = 128  # (kept for completeness; your model file controls this)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_ds = LFWCW2Dataset("data/splits/lfw_cw2_train.txt")
    val_ds = LFWCW2Dataset("data/splits/lfw_cw2_val.txt")

    # IMPORTANT for Windows: start with num_workers=0 (no multiprocessing issues)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = FaceEmbeddingModel(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings, logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.3f}, Acc={acc:.1f}%")

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/face_embeddings_cw2.pth")
    print("Model saved to models/face_embeddings_cw2.pth")


if __name__ == "__main__":
    train()
