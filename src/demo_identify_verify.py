import os
import random
import cv2
import torch
import pandas as pd
import numpy as np
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

MANIFEST = "data/manifest/prototype_manifest.csv"
THRESHOLD = 0.65          # start here for your current scores
TOPK = 3                  # show top-k predictions

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return x

def embed(model, img_tensor):
    with torch.no_grad():
        return model(img_tensor).numpy()[0]

def short_name(path):
    # nicer printing: "Person_Name/Person_Name_0001.jpg"
    return path.replace("\\", "/").split("demo/")[-1]

df = pd.read_csv(MANIFEST)
gallery_df = df[df["usage"] == "gallery"]
query_df = df[df["usage"] == "query"]

# Load model (downloads weights first run)
model = InceptionResnetV1(pretrained="vggface2").eval()

# -------------------------------
# 1) Build gallery (1 reference per identity)
# -------------------------------
gallery_ref_path = {}
gallery_emb = {}

for _, row in gallery_df.iterrows():
    identity = row["identity"]
    ref_path = row["image_path"]
    gallery_ref_path[identity] = ref_path
    gallery_emb[identity] = embed(model, load_image(ref_path))

names = list(gallery_emb.keys())
embs = np.vstack([gallery_emb[n] for n in names])

print("\n================= FACE RECOGNITION DEMO =================")
print(f"Identities in gallery: {len(names)}")
print("Reference images used (gallery enrolment):")
for n in names:
    print(f"  - {n}: {short_name(gallery_ref_path[n])}")

# -------------------------------
# 2) Pick random query
# -------------------------------
row = query_df.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
true_id = row["identity"]
query_path = row["image_path"]
q = embed(model, load_image(query_path))

print("\nQuery (test) image:")
print(f"  Path: {short_name(query_path)}")
print(f"  True identity label: {true_id}")

# -------------------------------
# 3) IDENTIFY: nearest neighbour
# -------------------------------
sims = cosine_similarity([q], embs)[0]
order = np.argsort(-sims)  # descending

best_idx = int(order[0])
pred_id = names[best_idx]
pred_sim = float(sims[best_idx])

print("\nIDENTIFICATION RESULT")
print(f"  Predicted identity: {pred_id}")
print(f"  Similarity to its reference: {pred_sim:.3f}")
print(f"  Reference image used: {short_name(gallery_ref_path[pred_id])}")

print(f"\nTop-{TOPK} matches:")
for rank, idx in enumerate(order[:TOPK], start=1):
    idx = int(idx)
    print(f"  {rank}) {names[idx]:<18} sim={float(sims[idx]):.3f}  ref={short_name(gallery_ref_path[names[idx]])}")

# -------------------------------
# 4) VERIFY: claim + threshold
# -------------------------------
claim_id = pred_id  # for demo: claim the predicted person
claim_sim = float(cosine_similarity([q], [gallery_emb[claim_id]])[0][0])
result = "MATCH ✅" if claim_sim >= THRESHOLD else "NO MATCH ❌"

print("\nVERIFICATION RESULT")
print(f"  Claim identity: {claim_id}")
print(f"  Similarity to claim reference: {claim_sim:.3f}")
print(f"  Threshold: {THRESHOLD:.2f}")
print(f"  Decision: {result}")
print("=========================================================\n")
