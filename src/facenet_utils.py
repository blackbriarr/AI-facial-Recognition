import os
import cv2
import torch
import numpy as np
import pandas as pd
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

MANIFEST = "data/manifest/prototype_manifest.csv"
IMG_SIZE = 160


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def load_manifest():
    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(f"Manifest file not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    df["image_path"] = df["image_path"].apply(normalize_path)
    return df


def load_rgb_image(path: str):
    path = normalize_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_image(img_rgb):
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return x


def load_facenet_model():
    model = InceptionResnetV1(pretrained="vggface2").eval()
    return model


def embed_tensor(model, img_tensor):
    with torch.no_grad():
        emb = model(img_tensor).cpu().numpy()[0]
    return emb


def embed_image_array(model, img_rgb):
    img_tensor = preprocess_image(img_rgb)
    return embed_tensor(model, img_tensor)


def build_gallery(model):
    df = load_manifest()

    if "usage" not in df.columns:
        raise ValueError("Manifest must contain a 'usage' column.")
    if "identity" not in df.columns or "image_path" not in df.columns:
        raise ValueError("Manifest must contain 'identity' and 'image_path' columns.")

    gallery_df = df[df["usage"] == "gallery"].copy()

    if gallery_df.empty:
        raise ValueError("No gallery rows found in manifest (usage == 'gallery').")

    gallery_ref_path = {}
    gallery_emb = {}

    for _, row in gallery_df.iterrows():
        identity = row["identity"]
        ref_path = row["image_path"]

        gallery_ref_path[identity] = ref_path
        ref_img = load_rgb_image(ref_path)
        gallery_emb[identity] = embed_image_array(model, ref_img)

    names = list(gallery_emb.keys())
    embs = np.vstack([gallery_emb[name] for name in names])

    return names, embs, gallery_ref_path, gallery_emb


def identify_face(query_emb, names, embs, topk=3):
    sims = cosine_similarity([query_emb], embs)[0]
    order = np.argsort(-sims)

    results = []
    for idx in order[:topk]:
        idx = int(idx)
        results.append({
            "identity": names[idx],
            "similarity": float(sims[idx])
        })

    return results


def verify_face(query_emb, identity, gallery_emb, threshold=0.65):
    claim_emb = gallery_emb[identity]
    sim = float(cosine_similarity([query_emb], [claim_emb])[0][0])
    decision = "MATCH ✅" if sim >= threshold else "NO MATCH ❌"
    return sim, decision