import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.models.resnet18_arcface import FaceEmbeddingModel


st.set_page_config(
    page_title="AI Biometric Face Recognition",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_lfw_resnet18.pth"
CLASS_MAPPING_PATH = CHECKPOINT_DIR / "class_mapping.json"

IMAGE_SIZE = 224
NUM_CLASSES = 250
EMBEDDING_DIM = 128

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_class_names():
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if all(isinstance(v, int) for v in data.values()):
            items = sorted(data.items(), key=lambda x: x[1])
            return [k for k, _ in items]
        if "class_names" in data:
            return data["class_names"]

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported class mapping format")


@st.cache_resource
def load_model():
    model = FaceEmbeddingModel(
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        pretrained=False
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def preprocess_image(image: Image.Image):
    return transform(image.convert("RGB")).unsqueeze(0)


def predict_identity(model, class_names, pil_image, topk=5):
    x = preprocess_image(pil_image)

    with torch.no_grad():
        output = model(x)
        _, logits = output if isinstance(output, tuple) else (None, output)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=min(topk, len(class_names)))

    results = []
    for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
        results.append({
            "identity": class_names[idx],
            "confidence": float(prob)
        })
    return results


st.title("🧠 AI Biometric Face Recognition System")
st.write("Upload a face image to predict the most likely identity from the trained classifier.")

if not CHECKPOINT_PATH.exists():
    st.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
    st.stop()

if not CLASS_MAPPING_PATH.exists():
    st.error(f"Class mapping not found: {CLASS_MAPPING_PATH}")
    st.stop()

class_names = load_class_names()
model = load_model()

with st.sidebar:
    st.header("Model Info")
    st.write(f"Checkpoint: `{CHECKPOINT_PATH.name}`")
    st.write(f"Classes loaded: **{len(class_names)}**")

uploaded_file = st.file_uploader("Upload a query face image", type=["jpg", "jpeg", "png"])
topk = st.slider("Show top-k predictions", 1, 10, 5)

if uploaded_file is None:
    st.info("Please upload an image to begin.")
    st.stop()

pil_image = Image.open(uploaded_file).convert("RGB")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Query Image")
    st.image(pil_image, use_container_width=True)

if st.button("Run recognition"):
    try:
        results = predict_identity(model, class_names, pil_image, topk=topk)
        best = results[0]

        with col2:
            st.subheader("Prediction")
            st.metric("Predicted Identity", best["identity"])
            st.metric("Confidence", f"{best['confidence']:.4f}")

        st.subheader(f"Top-{topk} Predictions")
        for i, r in enumerate(results, start=1):
            st.write(f"{i}. **{r['identity']}** — confidence: {r['confidence']:.4f}")

    except Exception as e:
        st.error(f"Recognition failed: {e}")