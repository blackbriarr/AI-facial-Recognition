from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTER_ROOT = PROJECT_ROOT.parent
DATA_ROOT = OUTER_ROOT / "data"

VGGFACE2_ROOT = DATA_ROOT / "vggface2"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "vggface2_300_train.txt"
VAL_FILE = OUTPUT_DIR / "vggface2_300_val.txt"
TEST_FILE = OUTPUT_DIR / "vggface2_300_test.txt"
LABELS_FILE = OUTPUT_DIR / "vggface2_300_labels.txt"

SEED = 42
NUM_IDENTITIES = 300
MIN_IMAGES_PER_ID = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def collect_identity_images(root: Path):
    people = {}

    if not root.exists():
        raise FileNotFoundError(f"VGGFace2 root not found: {root}")

    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue

        images = sorted([p for p in person_dir.iterdir() if p.is_file() and is_image_file(p)])

        if len(images) >= MIN_IMAGES_PER_ID:
            people[person_dir.name] = images

    return people


def split_images(images):
    n = len(images)

    n_train = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    return train_imgs, val_imgs, test_imgs


def make_relative_to_outer_root(path: Path) -> str:
    return path.relative_to(OUTER_ROOT).as_posix()


def main():
    random.seed(SEED)

    people = collect_identity_images(VGGFACE2_ROOT)
    identities = sorted(people.keys())

    if len(identities) < NUM_IDENTITIES:
        raise ValueError(
            f"Only found {len(identities)} identities with at least {MIN_IMAGES_PER_ID} images, "
            f"but NUM_IDENTITIES={NUM_IDENTITIES}"
        )

    selected_identities = identities[:NUM_IDENTITIES]
    label_map = {name: idx for idx, name in enumerate(selected_identities)}

    train_lines = []
    val_lines = []
    test_lines = []

    for person in selected_identities:
        images = people[person][:]
        random.shuffle(images)

        train_imgs, val_imgs, test_imgs = split_images(images)
        label = label_map[person]

        for img_path in train_imgs:
            train_lines.append(f"{make_relative_to_outer_root(img_path)} {label}")

        for img_path in val_imgs:
            val_lines.append(f"{make_relative_to_outer_root(img_path)} {label}")

        for img_path in test_imgs:
            test_lines.append(f"{make_relative_to_outer_root(img_path)} {label}")

    TRAIN_FILE.write_text("\n".join(train_lines), encoding="utf-8")
    VAL_FILE.write_text("\n".join(val_lines), encoding="utf-8")
    TEST_FILE.write_text("\n".join(test_lines), encoding="utf-8")

    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        for identity, label in label_map.items():
            f.write(f"{label} {identity}\n")

    print(f"VGGFace2 root: {VGGFACE2_ROOT}")
    print(f"Selected identities: {len(selected_identities)}")
    print(f"Train images: {len(train_lines)}")
    print(f"Val images: {len(val_lines)}")
    print(f"Test images: {len(test_lines)}")
    print(f"Saved: {TRAIN_FILE}")
    print(f"Saved: {VAL_FILE}")
    print(f"Saved: {TEST_FILE}")
    print(f"Saved: {LABELS_FILE}")


if __name__ == "__main__":
    main()