import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.models.resnet18_arcface import FaceEmbeddingModel

try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False


st.set_page_config(
    page_title="AI Biometric Face Recognition",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"

IMAGE_SIZE = 224
NUM_CLASSES = 250
EMBEDDING_DIM = 128


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def detect_available_models():
    models = {}

    resnet_ckpt = find_first_existing([
        CHECKPOINT_DIR / "best_lfw_resnet18.pth",
        CHECKPOINT_DIR / "best_resnet18.pth",
        CHECKPOINT_DIR / "best_checkpoint.pth",
        CHECKPOINT_DIR / "last_checkpoint.pth",
    ])

    facenet_ckpt = find_first_existing([
        CHECKPOINT_DIR / "best_facenet.pth",
        CHECKPOINT_DIR / "facenet_best.pth",
        CHECKPOINT_DIR / "best_facenet_checkpoint.pth",
        CHECKPOINT_DIR / "last_facenet_checkpoint.pth",
        CHECKPOINT_DIR / "facenet_checkpoint.pth",
    ])

    class_map = find_first_existing([
        CHECKPOINT_DIR / "class_mapping.json",
        CHECKPOINT_DIR / "idx_to_class.json",
        CHECKPOINT_DIR / "class_names.json",
    ])

    if resnet_ckpt:
        models["ResNet18 ArcFace"] = {
            "type": "resnet",
            "checkpoint": resnet_ckpt,
            "class_mapping": class_map,
        }

    if facenet_ckpt:
        models["FaceNet"] = {
            "type": "facenet",
            "checkpoint": facenet_ckpt,
            "class_mapping": class_map,
        }

    return models


def load_class_names(class_mapping_path):
    if class_mapping_path is None or not class_mapping_path.exists():
        return [f"class_{i}" for i in range(NUM_CLASSES)]

    with open(class_mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if all(isinstance(v, int) for v in data.values()):
            items = sorted(data.items(), key=lambda x: x[1])
            return [k for k, _ in items]
        if "class_names" in data and isinstance(data["class_names"], list):
            return data["class_names"]

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported class mapping format")


@st.cache_resource
def load_model(model_name, checkpoint_path):
    available_models = detect_available_models()
    config = available_models[model_name]
    model_type = config["type"]

    if model_type == "resnet":
        model = FaceEmbeddingModel(
            num_classes=NUM_CLASSES,
            embedding_dim=EMBEDDING_DIM,
            pretrained=False
        )
    elif model_type == "facenet":
        if not FACENET_AVAILABLE:
            raise ImportError("facenet-pytorch is not installed.")
        model = InceptionResnetV1(
            pretrained=None,
            classify=True,
            num_classes=NUM_CLASSES
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def preprocess_image(image: Image.Image):
    return transform(image.convert("RGB")).unsqueeze(0)


def predict_identity(model, model_type, class_names, pil_image, topk=5):
    x = preprocess_image(pil_image)

    with torch.no_grad():
        output = model(x)

        if model_type == "resnet":
            _, logits = output if isinstance(output, tuple) else (None, output)
        else:
            logits = output

        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=min(topk, len(class_names)))

    results = []
    for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
        identity = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        results.append({
            "identity": identity,
            "confidence": float(prob)
        })

    return results


available_models = detect_available_models()

st.title("🧠 AI Biometric Face Recognition System")
st.write("Upload a face image and choose one of the detected trained models.")

if not available_models:
    st.error(f"No supported checkpoints found in: {CHECKPOINT_DIR}")
    st.stop()

with st.sidebar:
    st.header("Model Selection")
    selected_model = st.selectbox("Choose model", list(available_models.keys()))
    topk = st.slider("Show top-k predictions", 1, 10, 5)

selected_config = available_models[selected_model]
checkpoint_path = selected_config["checkpoint"]
class_mapping_path = selected_config["class_mapping"]
model_type = selected_config["type"]

if model_type == "facenet" and not FACENET_AVAILABLE:
    st.error("FaceNet requires facenet-pytorch. Install it with: pip install facenet-pytorch")
    st.stop()

class_names = load_class_names(class_mapping_path)
model = load_model(selected_model, checkpoint_path)

with st.sidebar:
    st.header("Model Info")
    st.write(f"Selected model: **{selected_model}**")
    st.write(f"Model type: **{model_type}**")
    st.write(f"Checkpoint file: `{checkpoint_path.name}`")
    st.write(f"Classes loaded: **{len(class_names)}**")
    if class_mapping_path:
        st.write(f"Class map: `{class_mapping_path.name}`")
    else:
        st.write("Class map: using fallback class names")

uploaded_file = st.file_uploader("Upload a query face image", type=["jpg", "jpeg", "png"])

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
        results = predict_identity(model, model_type, class_names, pil_image, topk=topk)
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