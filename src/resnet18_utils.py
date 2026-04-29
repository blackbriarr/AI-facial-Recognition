from pathlib import Path
import csv
import cv2
import numpy as np
import torch
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

from src.model_embeddings import FaceEmbeddingModel

IMG_SIZE = 160
DEFAULT_WEIGHTS = Path("models/face_embeddings_cw2.pth")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_resnet18_model(weights_path=DEFAULT_WEIGHTS, num_classes=250):
    model = FaceEmbeddingModel(num_classes=num_classes)
    state = torch.load(str(weights_path), map_location=_device())
    model.load_state_dict(state, strict=True)
    model.eval().to(_device())
    return model

def load_rgb_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def embed_image_array_resnet18(model, rgb_img):
    x = preprocess(rgb_img).unsqueeze(0).to(_device())
    with torch.no_grad():
        emb, _ = model(x)
    return emb.detach().cpu().numpy()[0]

def identify_face_resnet18(query_emb, names, embs, topk=3):
    sims = cosine_similarity([query_emb], embs)[0]
    order = np.argsort(-sims)[:topk]
    return [{"identity": names[i], "similarity": float(sims[i])} for i in order]

def verify_face_resnet18(query_emb, claim_identity, gallery_emb, threshold=0.65):
    sim = float(cosine_similarity([query_emb], [gallery_emb[claim_identity]])[0][0])
    return sim, ("MATCH ✅" if sim >= threshold else "NO MATCH ❌")

def _iter_images(root):
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def _identity_from_path(path):
    p = Path(path)
    if len(p.parts) >= 2:
        return p.parts[-2]
    return p.stem.split("_")[0]

def _first_image_per_identity(root):
    chosen = {}
    for p in sorted(_iter_images(root)):
        ident = _identity_from_path(p)
        if ident not in chosen:
            chosen[ident] = p
    return chosen

def build_resnet18_gallery(dataset_root, model=None, report_path=None):
    if model is None:
        model = load_resnet18_model()

    refs = _first_image_per_identity(dataset_root)
    names = []
    embs = []
    ref_paths = {}
    gallery_emb = {}

    for identity, img_path in refs.items():
        try:
            emb = embed_image_array_resnet18(model, load_rgb_image(img_path))
            names.append(identity)
            embs.append(emb)
            ref_paths[identity] = str(img_path)
            gallery_emb[identity] = emb
        except Exception:
            pass

    if not embs:
        raise RuntimeError(f"No gallery embeddings created from {dataset_root}")

    embs = np.vstack(embs)

    if report_path:
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["identity", "ref_path"])
            for k, v in ref_paths.items():
                w.writerow([k, v])

    return names, embs, ref_paths, gallery_emb