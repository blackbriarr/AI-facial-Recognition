from pathlib import Path
import argparse
import shutil

import cv2
from PIL import Image


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from: {cascade_path}")
    return detector


def detect_largest_face(detector, image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return x, y, w, h


def expand_box(x, y, w, h, img_w, img_h, scale=1.35):
    cx = x + w / 2.0
    cy = y + h / 2.0

    new_w = w * scale
    new_h = h * scale

    x1 = int(max(0, cx - new_w / 2.0))
    y1 = int(max(0, cy - new_h / 2.0))
    x2 = int(min(img_w, cx + new_w / 2.0))
    y2 = int(min(img_h, cy + new_h / 2.0))

    return x1, y1, x2, y2


def crop_face(detector, image_bgr, image_size):
    face_box = detect_largest_face(detector, image_bgr)
    if face_box is None:
        return None

    x, y, w, h = face_box
    img_h, img_w = image_bgr.shape[:2]
    x1, y1, x2, y2 = expand_box(x, y, w, h, img_w, img_h, scale=1.35)

    face = image_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb).resize((image_size, image_size), Image.Resampling.LANCZOS)
    return face_pil


def process_split(detector, split_in, split_out, image_size, copy_if_fail=False):
    total = 0
    saved = 0
    failed = 0

    class_dirs = [p for p in split_in.iterdir() if p.is_dir()]
    for class_dir in class_dirs:
        out_class = split_out / class_dir.name
        ensure_dir(out_class)

        images = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
        for img_path in images:
            total += 1
            out_path = out_class / img_path.name

            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                failed += 1
                continue

            try:
                face = crop_face(detector, image_bgr, image_size)
                if face is not None:
                    face.save(out_path)
                    saved += 1
                else:
                    failed += 1
                    if copy_if_fail:
                        shutil.copy2(img_path, out_path)
            except Exception:
                failed += 1
                if copy_if_fail:
                    shutil.copy2(img_path, out_path)

            if total % 500 == 0:
                print(f"{split_in.name}: processed {total} | saved {saved} | failed {failed}", flush=True)

    return {"total": total, "saved": saved, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="OpenCV face crop preprocessing for train/val/test dataset")
    parser.add_argument("--input-root", required=True, help="Dataset root containing train/val/test")
    parser.add_argument("--output-root", required=True, help="Output dataset root")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--copy-if-fail", action="store_true", help="Copy original image if no face is detected")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    for split in ["train", "val", "test"]:
        if not (input_root / split).exists():
            raise FileNotFoundError(f"Missing split folder: {input_root / split}")

    ensure_dir(output_root)

    print("Loading OpenCV Haar cascade detector...", flush=True)
    detector = load_face_detector()

    all_stats = {}
    for split in ["train", "val", "test"]:
        print(f"\nProcessing split: {split}", flush=True)
        split_in = input_root / split
        split_out = output_root / split
        ensure_dir(split_out)

        stats = process_split(
            detector=detector,
            split_in=split_in,
            split_out=split_out,
            image_size=args.image_size,
            copy_if_fail=args.copy_if_fail,
        )
        all_stats[split] = stats
        print(f"{split} done: {stats}", flush=True)

    print("\nAll preprocessing complete.", flush=True)
    print(all_stats, flush=True)


if __name__ == "__main__":
    main()