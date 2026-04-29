from pathlib import Path
import csv
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, UnidentifiedImageError
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.models.resnet18_arcface import FaceEmbeddingModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CPLFW_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "cplfw" / "cplfw" / "aligned images"
CPLFW_PAIRS = PROJECT_ROOT / "data" / "raw" / "cplfw" / "cplfw" / "pairs_CPLFW.txt"

CFP_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "cfp-fp" / "cfp-fp" / "Data" / "Images"
CFP_PROTOCOL_ROOT = PROJECT_ROOT / "data" / "raw" / "cfp-fp" / "cfp-fp" / "Protocol"

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_lfw_resnet18.pth"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ImageFile.LOAD_TRUNCATED_IMAGES = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def load_image(img_path):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img.load()
            return transform(img)
    except (OSError, UnidentifiedImageError, ValueError) as e:
        raise RuntimeError(f"Failed to load image: {img_path} | {e}")


def get_embeddings(model, batch, device):
    batch = batch.to(device)
    with torch.no_grad():
        output = model(batch)
        embeddings = output[0] if isinstance(output, tuple) else output
    return embeddings


def build_cplfw_map(root_dir):
    mapping = {}
    for img_path in Path(root_dir).rglob("*.jpg"):
        mapping[img_path.name] = img_path
    return mapping


def resolve_cplfw_path(image_map, token):
    token = token.strip()

    if token in image_map:
        return image_map[token]

    m = re.match(r"(.+?)_(\d+)\.jpg$", token, flags=re.IGNORECASE)
    if m:
        person = m.group(1)
        idx = int(m.group(2))

        candidates = [
            f"{person}_{idx:04d}.jpg",
            f"{person}_{idx}.jpg",
        ]

        for candidate in candidates:
            if candidate in image_map:
                return image_map[candidate]

        for key, path in image_map.items():
            if key.startswith(person) and (
                key.endswith(f"_{idx:04d}.jpg") or key.endswith(f"_{idx}.jpg")
            ):
                return path

    raise FileNotFoundError(f"Could not resolve CPLFW image: {token}")


def parse_cplfw_pairs_file(pairs_file):
    lines = []
    with open(pairs_file, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                lines.append((parts[0], int(parts[1])))

    pairs = []
    for i in range(0, len(lines) - 1, 2):
        img1, label1 = lines[i]
        img2, label2 = lines[i + 1]
        if label1 == 1 and label2 == 1:
            pairs.append((img1, img2, 1))
        elif label1 == 0 and label2 == 0:
            pairs.append((img1, img2, 0))
    return pairs


def load_index_map(txt_path, image_root):
    mapping = {}
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, rel = line.split(maxsplit=1)
            rel = rel.replace("../Data/Images/", "").replace("\\", "/")
            mapping[int(idx)] = image_root / rel
    return mapping


def load_split_pairs(pair_file, left_map, right_map, label):
    pairs = []
    with open(pair_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split(",")
            a = int(a.strip())
            b = int(b.strip())
            if a in left_map and b in right_map:
                pairs.append((left_map[a], right_map[b], label))
    return pairs


def load_cfp_all_folds(protocol_root, image_root, mode="FF"):
    frontal_map = load_index_map(protocol_root / "Pair_list_F.txt", image_root)
    profile_map = load_index_map(protocol_root / "Pair_list_P.txt", image_root)

    all_pairs = []
    for fold in range(1, 11):
        fold_dir = protocol_root / "Split" / mode / f"{fold:02d}"
        if mode == "FF":
            left_map = frontal_map
            right_map = frontal_map
        else:
            left_map = frontal_map
            right_map = profile_map

        same_pairs = load_split_pairs(fold_dir / "same.txt", left_map, right_map, 1)
        diff_pairs = load_split_pairs(fold_dir / "diff.txt", left_map, right_map, 0)

        print(f"{mode} fold {fold:02d}: same={len(same_pairs)}, diff={len(diff_pairs)}", flush=True)
        all_pairs.extend(same_pairs)
        all_pairs.extend(diff_pairs)

    return all_pairs


def find_best_threshold(labels, scores, thresholds=np.arange(-1.0, 1.0001, 0.01)):
    best_thr = None
    best_acc = -1.0

    labels = np.asarray(labels)
    scores = np.asarray(scores)

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)

    return best_thr, best_acc


class CPLFWPairDataset(Dataset):
    def __init__(self, pairs_file, image_root):
        self.image_map = build_cplfw_map(image_root)
        self.pairs = parse_cplfw_pairs_file(pairs_file)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_name, img2_name, label = self.pairs[idx]
        img1_path = resolve_cplfw_path(self.image_map, img1_name)
        img2_path = resolve_cplfw_path(self.image_map, img2_name)

        return (
            load_image(img1_path),
            load_image(img2_path),
            int(label),
            str(img1_path),
            str(img2_path),
        )


class FilePairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        return (
            load_image(img1_path),
            load_image(img2_path),
            int(label),
            str(img1_path),
            str(img2_path),
        )


def evaluate_pairs(model, dataloader, device, benchmark_name="benchmark"):
    model.eval()
    all_labels = []
    all_scores = []
    rows = []
    skipped = 0

    with torch.no_grad():
        for batch_idx, (img1, img2, labels, path1, path2) in enumerate(dataloader):
            try:
                print(f"[{benchmark_name}] Processing batch {batch_idx + 1}", flush=True)

                emb1 = get_embeddings(model, img1, device)
                emb2 = get_embeddings(model, img2, device)

                scores = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8).cpu().numpy()
                labels_np = labels.numpy()

                all_labels.extend(labels_np.tolist())
                all_scores.extend(scores.tolist())

                for i in range(len(labels_np)):
                    rows.append([
                        path1[i],
                        path2[i],
                        int(labels_np[i]),
                        float(scores[i]),
                    ])
            except Exception as e:
                skipped += 1
                print(f"[{benchmark_name}] Skipped batch {batch_idx + 1}: {e}", flush=True)

    if not all_labels:
        print(f"No valid pairs loaded for {benchmark_name}, skipping.", flush=True)
        return None, None

    best_threshold, best_accuracy = find_best_threshold(all_labels, all_scores)

    final_rows = []
    for row in rows:
        pred = int(row[3] >= best_threshold)
        final_rows.append([row[0], row[1], row[2], row[3], pred])

    csv_path = REPORT_DIR / f"{benchmark_name}_pair_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img1", "img2", "label", "cosine_similarity", "prediction"])
        writer.writerows(final_rows)

    print(f"[{benchmark_name}] Skipped batches: {skipped}", flush=True)
    return best_threshold, best_accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    model = FaceEmbeddingModel(
        num_classes=250,
        embedding_dim=128,
        pretrained=False
    ).to(device)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Loaded model:", CHECKPOINT_PATH, flush=True)

    summary_rows = []

    if CPLFW_PAIRS.exists():
        print("Loading CPLFW...", flush=True)
        cplfw_dataset = CPLFWPairDataset(CPLFW_PAIRS, CPLFW_IMAGE_ROOT)
        print(f"CPLFW pairs loaded: {len(cplfw_dataset)}", flush=True)

        if len(cplfw_dataset) > 0:
            cplfw_loader = DataLoader(cplfw_dataset, batch_size=8, shuffle=False, num_workers=0)
            cplfw_thr, cplfw_acc = evaluate_pairs(model, cplfw_loader, device, benchmark_name="cplfw")
            if cplfw_acc is not None:
                print(f"CPLFW Best Threshold: {cplfw_thr:.2f} | Best Accuracy: {cplfw_acc * 100:.2f}%", flush=True)
                summary_rows.append(["CPLFW", cplfw_thr, cplfw_acc, len(cplfw_dataset)])
        else:
            print("No valid pairs loaded for cplfw, skipping.", flush=True)
    else:
        print("CPLFW pairs file not found:", CPLFW_PAIRS, flush=True)

    if CFP_PROTOCOL_ROOT.exists():
        print("Loading CFP frontal...", flush=True)
        cfp_ff_pairs = load_cfp_all_folds(CFP_PROTOCOL_ROOT, CFP_IMAGE_ROOT, mode="FF")
        print(f"CFP frontal pairs loaded: {len(cfp_ff_pairs)}", flush=True)

        if cfp_ff_pairs:
            cfp_f_dataset = FilePairDataset(cfp_ff_pairs)
            cfp_f_loader = DataLoader(cfp_f_dataset, batch_size=8, shuffle=False, num_workers=0)
            cfp_f_thr, cfp_f_acc = evaluate_pairs(model, cfp_f_loader, device, benchmark_name="cfp_frontal")
            if cfp_f_acc is not None:
                print(f"CFP-FP Frontal Best Threshold: {cfp_f_thr:.2f} | Best Accuracy: {cfp_f_acc * 100:.2f}%", flush=True)
                summary_rows.append(["CFP-FP Frontal", cfp_f_thr, cfp_f_acc, len(cfp_f_dataset)])
        else:
            print("CFP-FP Frontal: no valid pairs found, skipped.", flush=True)

        print("Loading CFP profile...", flush=True)
        cfp_fp_pairs = load_cfp_all_folds(CFP_PROTOCOL_ROOT, CFP_IMAGE_ROOT, mode="FP")
        print(f"CFP profile pairs loaded: {len(cfp_fp_pairs)}", flush=True)

        if cfp_fp_pairs:
            cfp_p_dataset = FilePairDataset(cfp_fp_pairs)
            cfp_p_loader = DataLoader(cfp_p_dataset, batch_size=8, shuffle=False, num_workers=0)
            cfp_p_thr, cfp_p_acc = evaluate_pairs(model, cfp_p_loader, device, benchmark_name="cfp_profile")
            if cfp_p_acc is not None:
                print(f"CFP-FP Profile Best Threshold: {cfp_p_thr:.2f} | Best Accuracy: {cfp_p_acc * 100:.2f}%", flush=True)
                summary_rows.append(["CFP-FP Profile", cfp_p_thr, cfp_p_acc, len(cfp_p_dataset)])
        else:
            print("CFP-FP Profile: no valid pairs found, skipped.", flush=True)
    else:
        print("CFP protocol folder not found:", CFP_PROTOCOL_ROOT, flush=True)

    summary_csv = REPORT_DIR / "verification_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "best_threshold", "best_accuracy", "num_pairs"])
        writer.writerows(summary_rows)

    print("Saved summary to:", summary_csv, flush=True)


if __name__ == "__main__":
    main()