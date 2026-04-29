import cv2
import torch
import pandas as pd
import numpy as np
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

MANIFEST = "data/manifest/manifest.csv"

df = pd.read_csv(MANIFEST)
model = InceptionResnetV1(pretrained="vggface2").eval()

embeddings = []
labels = []

for _, row in df.iterrows():
    path = row["image_path"]
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        emb = model(x).numpy()[0]

    embeddings.append(emb)
    labels.append(row["identity"])

embeddings = np.vstack(embeddings)
print("Embeddings shape:", embeddings.shape)

sim01 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Cosine similarity (0 vs 1): {sim01:.3f}")
print("Label 0:", labels[0])
print("Label 1:", labels[1])
