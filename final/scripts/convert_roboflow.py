"""Convert Roboflow polygon-format dataset to YOLO bbox format.

- Converts polygon labels to bounding boxes
- Remaps class IDs to match existing best_color.pt model
- Creates train/val split (85/15)
- Writes data.yaml

Usage:
  python final/scripts/convert_roboflow.py
"""
import zipfile
import os
import random
import shutil

ZIP_PATH = "snooker_dataset_v5.zip"
OUT_DIR = "final/scripts/dataset"

# Roboflow v3 classes: 0=black 1=blue 2=brown 3=green 4=light 5=objects 6=pink 7=pocket 8=red 9=white 10=yellow
# Model classes:       0=black-ball 1=blue-ball 2=brown-ball 3=green-ball 4=pink-ball 5=pocket 6=red-ball 7=white-ball 8=yellow-ball
REMAP = {
    0: 0,   # black -> black-ball
    1: 1,   # blue -> blue-ball
    2: 2,   # brown -> brown-ball
    3: 3,   # green -> green-ball
    # 4: light -> SKIP
    # 5: objects -> SKIP
    6: 4,   # pink -> pink-ball
    7: 5,   # pocket -> pocket
    8: 6,   # red -> red-ball
    9: 7,   # white -> white-ball
    10: 8,  # yellow -> yellow-ball
}

MODEL_NAMES = [
    "black-ball", "blue-ball", "brown-ball", "green-ball",
    "pink-ball", "pocket", "red-ball", "white-ball", "yellow-ball",
]


def polygon_to_bbox(coords):
    """Convert polygon points [x1,y1,x2,y2,...] to [cx, cy, w, h]."""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def main():
    if not os.path.exists(ZIP_PATH):
        print(f"Zip not found: {ZIP_PATH}")
        return

    z = zipfile.ZipFile(ZIP_PATH)

    # Clean and create output dirs
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            p = os.path.join(OUT_DIR, sub, split)
            if os.path.exists(p):
                shutil.rmtree(p)
            os.makedirs(p, exist_ok=True)

    # Get all image files
    img_files = [n for n in z.namelist()
                 if "/images/" in n and n.endswith(".jpg")]
    random.seed(42)
    random.shuffle(img_files)

    val_count = max(1, int(len(img_files) * 0.15))
    val_set = set(img_files[:val_count])

    converted = 0
    skipped = 0
    class_counts = {name: 0 for name in MODEL_NAMES}

    for img_path in img_files:
        split = "val" if img_path in val_set else "train"
        basename = os.path.basename(img_path)
        name = os.path.splitext(basename)[0]

        # Extract image
        dst_img = os.path.join(OUT_DIR, "images", split, basename)
        with open(dst_img, "wb") as f:
            f.write(z.read(img_path))

        # Find and convert label
        label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
        try:
            label_data = z.read(label_path).decode().strip()
        except KeyError:
            label_data = ""

        new_lines = []
        for line in label_data.split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            class_id = int(parts[0])

            if class_id not in REMAP:
                skipped += 1
                continue

            new_class = REMAP[class_id]
            coords = [float(x) for x in parts[1:]]

            if len(coords) == 4:
                cx, cy, w, h = coords
            else:
                cx, cy, w, h = polygon_to_bbox(coords)

            new_lines.append(f"{new_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            class_counts[MODEL_NAMES[new_class]] += 1

        dst_label = os.path.join(OUT_DIR, "labels", split, name + ".txt")
        with open(dst_label, "w") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")

        converted += 1

    # Write data.yaml
    abs_out = os.path.abspath(OUT_DIR).replace("\\", "/")
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_out}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(MODEL_NAMES)}\n")
        f.write(f"names: {MODEL_NAMES}\n")

    train_n = len(os.listdir(os.path.join(OUT_DIR, "images", "train")))
    val_n = len(os.listdir(os.path.join(OUT_DIR, "images", "val")))

    print(f"Converted {converted} images (train={train_n}, val={val_n})")
    print(f"Skipped {skipped} 'objects' labels")
    print(f"\nPer-class counts:")
    for name, count in class_counts.items():
        print(f"  {name:>12s}: {count}")
    print(f"\ndata.yaml: {yaml_path}")
    print(f"\nReady to fine-tune:")
    print(f"  python final/scripts/train_model.py --dataset {yaml_path} "
          f"--base-model final/src/snookervision/data/model/best_color.pt "
          f"--epochs 50 --install")


if __name__ == "__main__":
    main()
