import os
import random
from collections import defaultdict

# Paths (actual folder on disk)
lfw_root = "data/raw/lfw"

output_dir = "data/splits"
os.makedirs(output_dir, exist_ok=True)

# Find all people with 3+ images
people_images = defaultdict(list)
for person in os.listdir(lfw_root):
    person_path = os.path.join(lfw_root, person)
    if os.path.isdir(person_path):
        images = [f for f in os.listdir(person_path) if f.lower().endswith(".jpg")]
        if len(images) >= 3:
            people_images[person].extend(images)

print(f"Found {len(people_images)} people with 3+ images")

# Pick top 250 (alphabetical, reproducible)
selected_people = sorted(people_images.keys())[:250]
print(f"Selected {len(selected_people)} people")

train_lines, val_lines, test_lines = [], [], []
random.seed(42)

for i, person in enumerate(selected_people):
    images = people_images[person]
    random.shuffle(images)

    n = len(images)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    # Train
    for img in images[:n_train]:
        rel = f"raw/lfw/{person}/{img}"
        train_lines.append(f"{rel} {i}")

    # Val
    for img in images[n_train:n_train + n_val]:
        rel = f"raw/lfw/{person}/{img}"
        val_lines.append(f"{rel} {i}")

    # Test
    for img in images[n_train + n_val:]:
        rel = f"raw/lfw/{person}/{img}"
        test_lines.append(f"{rel} {i}")

# Save
with open(os.path.join(output_dir, "lfw_cw2_train.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(train_lines))
with open(os.path.join(output_dir, "lfw_cw2_val.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(val_lines))
with open(os.path.join(output_dir, "lfw_cw2_test.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(test_lines))

print("✅ Splits created:")
print(f"  Train: {len(train_lines)} images")
print(f"  Val:   {len(val_lines)} images")
print(f"  Test:  {len(test_lines)} images")
