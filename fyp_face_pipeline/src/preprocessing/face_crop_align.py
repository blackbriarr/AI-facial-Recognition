from pathlib import Path
import argparse
import csv
import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError as e:
    raise ImportError(
        "insightface is not installed. Run: pip install insightface onnxruntime"
    ) from e


def iter_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def init_app():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def pad_resize_rgb(face_rgb, size=224):
    h, w = face_rgb.shape[:2]
    scale = size / max(h, w)
    nh = max(1, int(h * scale))
    nw = max(1, int(w * scale))
    resized = cv2.resize(face_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    yoff = (size - nh) // 2
    xoff = (size - nw) // 2
    canvas[yoff:yoff + nh, xoff:xoff + nw] = resized
    return canvas


def process_image(img_bgr, app, size=224):
    faces = app.get(img_bgr)
    if not faces:
        return None, "no_face"

    best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
    x1, y1, x2, y2 = map(int, best.bbox)
    h, w = img_bgr.shape[:2]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None, "bad_box"

    box_w = x2 - x1
    box_h = y2 - y1
    mx = int(box_w * 0.25)
    my = int(box_h * 0.25)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, "crop_fail"

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return pad_resize_rgb(crop_rgb, size), "insightface"


def process_dataset(input_root, output_root, size=224):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    report_path = output_root / "preprocess_report.csv"
    rows = []
    app = init_app()

    for img_path in iter_images(input_root):
        rel = img_path.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            rows.append([str(rel), "read_fail", ""])
            continue

        face, method = process_image(img, app, size=size)

        if face is None:
            ok = cv2.imwrite(str(out_path), img)
            rows.append([str(rel), "saved_original" if ok else "write_fail", method])
            continue

        ok = cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        rows.append([str(rel), "saved" if ok else "write_fail", method])

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "status", "method"])
        writer.writerows(rows)

    saved = sum(1 for r in rows if r[1] in {"saved", "saved_original"})
    print(f"Done. Saved {saved} images to {output_root}")
    print(f"Report: {report_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    process_dataset(args.input_root, args.output_root, size=args.size)


if __name__ == "__main__":
    main()