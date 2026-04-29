from pathlib import Path
import random
import shutil
import time

from PIL import Image
from torchvision import transforms


SOURCE_ROOT = Path(r"D:/projects/FYP/AI-facial-Recognition/AI-facial-Recognition/data/vggface2")
OUTPUT_ROOT = Path(r"D:/projects/FYP/AI-facial-Recognition/AI-facial-Recognition/data/vggface2_aligned_subset_250")
TRAIN_SPLIT = 0.8
SEED = 42
MAX_CLASSES = 250
IMAGE_SIZE = 224

random.seed(SEED)

resize = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
])


def is_image(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    img = resize(img)
    img.save(dst, quality=95)


def main():
    start_time = time.time()

    print(f"Source root: {SOURCE_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {SOURCE_ROOT}")

    identities = sorted([p for p in SOURCE_ROOT.iterdir() if p.is_dir()])[:MAX_CLASSES]
    print(f"Found {len(identities)} identity folders")

    if not identities:
        raise RuntimeError(f"No identity folders found in {SOURCE_ROOT}")

    if OUTPUT_ROOT.exists():
        print("Removing old output folder...")
        shutil.rmtree(OUTPUT_ROOT)

    (OUTPUT_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "val").mkdir(parents=True, exist_ok=True)

    counts = []
    total_written = 0

    for identity_idx, identity_dir in enumerate(identities, start=1):
        images = sorted([p for p in identity_dir.rglob("*") if p.is_file() and is_image(p)])

        print(f"[{identity_idx}/{len(identities)}] {identity_dir.name}: found {len(images)} images")

        if len(images) < 2:
            print(f"Skipping {identity_dir.name} because it has fewer than 2 images")
            continue

        rng = random.Random(SEED + abs(hash(identity_dir.name)) % 100000)
        rng.shuffle(images)

        split_idx = max(1, int(len(images) * TRAIN_SPLIT))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        if not val_images:
            val_images = train_images[-1:]
            train_images = train_images[:-1]

        for i, src in enumerate(train_images, start=1):
            rel = src.relative_to(identity_dir)
            dst = OUTPUT_ROOT / "train" / identity_dir.name / rel
            normalize_image(src, dst)
            total_written += 1

            if i % 100 == 0 or i == len(train_images):
                print(f"  train {identity_dir.name}: {i}/{len(train_images)}")

        for i, src in enumerate(val_images, start=1):
            rel = src.relative_to(identity_dir)
            dst = OUTPUT_ROOT / "val" / identity_dir.name / rel
            normalize_image(src, dst)
            total_written += 1

            if i % 50 == 0 or i == len(val_images):
                print(f"  val   {identity_dir.name}: {i}/{len(val_images)}")

        counts.append((identity_dir.name, len(train_images), len(val_images)))
        print(f"Finished {identity_dir.name} | train={len(train_images)} | val={len(val_images)} | total_written={total_written}")

    with open(OUTPUT_ROOT / "split_summary.csv", "w", encoding="utf-8") as f:
        f.write("identity,train_count,val_count\n")
        for name, tr, va in counts:
            f.write(f"{name},{tr},{va}\n")

    elapsed = time.time() - start_time
    print("\nDone.")
    print(f"Wrote aligned subset to: {OUTPUT_ROOT}")
    print(f"Identities used: {len(counts)}")
    print(f"Train images: {sum(c[1] for c in counts)}")
    print(f"Val images: {sum(c[2] for c in counts)}")
    print(f"Total written: {total_written}")
    print(f"Elapsed time: {elapsed/60:.2f} minutes")


if __name__ == "__main__":
    main()