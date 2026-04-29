from pathlib import Path

PROJECT_NAME = "fyp_face_pipeline"

folders = [
    "data/raw/vggface2",
    "data/raw/cplfw",
    "data/raw/cfp-fp",
    "data/processed/vggface2_aligned",
    "data/processed/cplfw_aligned",
    "data/processed/cfpfp_aligned",
    "data/splits/vggface2_subset",
    "data/splits/cplfw_pairs",
    "data/splits/cfpfp_pairs",
    "data/manifest",

    "src",
    "src/datasets",
    "src/models",
    "src/losses",
    "src/preprocessing",
    "src/training",
    "src/evaluation",
    "src/utils",

    "outputs/checkpoints",
    "outputs/logs",
    "outputs/reports",
    "outputs/plots",

    "notebooks",
    "app/demo_assets",
]

files = {
    "README.md": """# FYP Face Pipeline

Clean training and evaluation pipeline for face recognition.

## Structure
- src/preprocessing: face cleaning and alignment
- src/training: model training scripts
- src/evaluation: testing on CPLFW and CFP-FP
- outputs/checkpoints: saved models
- outputs/reports: metrics and CSV reports
- notebooks: optional analysis only
""",

    "requirements.txt": """torch
torchvision
numpy
pandas
matplotlib
scikit-learn
opencv-python
Pillow
tqdm
streamlit
""",

    ".gitignore": """__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/
.env
.ipynb_checkpoints/
outputs/checkpoints/
outputs/logs/
data/raw/
data/processed/
*.pth
*.pt
""",

    "src/__init__.py": "",
    "src/datasets/__init__.py": "",
    "src/models/__init__.py": "",
    "src/losses/__init__.py": "",
    "src/preprocessing/__init__.py": "",
    "src/training/__init__.py": "",
    "src/evaluation/__init__.py": "",
    "src/utils/__init__.py": "",

    "src/preprocessing/preprocess_faces.py": '''def main():
    print("Preprocessing faces...")

if __name__ == "__main__":
    main()
''',

    "src/preprocessing/align_faces.py": '''def main():
    print("Aligning faces...")

if __name__ == "__main__":
    main()
''',

    "src/models/resnet18_arcface.py": '''def build_model():
    print("Build ResNet18 + ArcFace model here")
''',

    "src/models/attention.py": '''def build_attention_module():
    print("Optional attention module here")
''',

    "src/models/dinov2_head.py": '''def build_dinov2_head():
    print("Build DINOv2 head here")
''',

    "src/losses/arcface_loss.py": '''class ArcFaceLoss:
    def __init__(self):
        pass
''',

    "src/datasets/imagefolder_dataset.py": '''class ImageFolderFaceDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
''',

    "src/datasets/pair_dataset.py": '''class PairFaceDataset:
    def __init__(self, pairs_file):
        self.pairs_file = pairs_file
''',

    "src/training/train_resnet18.py": '''def main():
    print("Training ResNet18...")

if __name__ == "__main__":
    main()
''',

    "src/training/train_dinov2.py": '''def main():
    print("Training DINOv2...")

if __name__ == "__main__":
    main()
''',

    "src/evaluation/evaluate_cplfw.py": '''def main():
    print("Evaluating on CPLFW...")

if __name__ == "__main__":
    main()
''',

    "src/evaluation/evaluate_cfpfp.py": '''def main():
    print("Evaluating on CFP-FP...")

if __name__ == "__main__":
    main()
''',

    "src/evaluation/metrics.py": '''def accuracy():
    pass
''',

    "src/utils/checkpoint.py": '''def save_checkpoint():
    pass
''',

    "src/utils/seed.py": '''def set_seed(seed=42):
    pass
''',

    "src/utils/logger.py": '''def get_logger():
    pass
''',

    "notebooks/results_analysis.ipynb": """{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
""",

    "app/streamlit_app.py": '''def main():
    print("Streamlit demo app placeholder")

if __name__ == "__main__":
    main()
'''
}

def create_project():
    root = Path(PROJECT_NAME)
    root.mkdir(exist_ok=True)

    for folder in folders:
        (root / folder).mkdir(parents=True, exist_ok=True)

    for file_path, content in files.items():
        full_path = root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if not full_path.exists():
            full_path.write_text(content, encoding="utf-8")

    print(f"Created project structure in: {root.resolve()}")
    print("Done.")

if __name__ == "__main__":
    create_project()