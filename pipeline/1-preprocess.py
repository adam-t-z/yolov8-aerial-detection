import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

VOC_PATH = "data-original"
OUTPUT_PATH = "data"

# Load preprocessing configuration (seed and split ratios) from config.yaml
with open(Path(__file__).with_name("config.yaml"), "r") as cfg_file:
    _config = yaml.safe_load(cfg_file)

_preprocess_cfg = _config.get("preprocess", {})
SPLIT = tuple(_preprocess_cfg.get("split", (0.7, 0.2, 0.1)))
SEED = _preprocess_cfg.get("seed", 42)


# -------- STEP 1: BUILD CLASS MAP AUTOMATICALLY --------
def build_class_map(xml_files):
    classes = set()

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.find("name").text
            classes.add(name)

    return {name: i for i, name in enumerate(sorted(classes))}


# -------- STEP 2: READ IMAGE SIZE --------
def get_size(xml_file):
    root = ET.parse(xml_file).getroot()
    size = root.find("size")
    return int(size.find("width").text), int(size.find("height").text)


# -------- STEP 3: CONVERT XML → YOLO --------
def convert(xml_file, class_map):
    root = ET.parse(xml_file).getroot()

    w, h = get_size(xml_file)
    labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in class_map:
            continue

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x = (xmin + xmax) / 2 / w
        y = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        labels.append(f"{class_map[name]} {x} {y} {bw} {bh}")

    return labels


# -------- STEP 4: SAVE FILES --------
def save(xml_file, split, class_map):
    img_file = xml_file.with_suffix(".jpg")
    if not img_file.exists():
        return

    labels = convert(xml_file, class_map)

    label_dir = Path(OUTPUT_PATH) / "labels" / split
    image_dir = Path(OUTPUT_PATH) / "images" / split

    label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    (label_dir / f"{xml_file.stem}.txt").write_text("\n".join(labels))
    shutil.copy(img_file, image_dir / img_file.name)


# -------- STEP 5: CREATE YAML --------
def create_yaml(class_map):
    data = {
        "train": str(Path(OUTPUT_PATH, "images/train").absolute()),
        "val": str(Path(OUTPUT_PATH, "images/val").absolute()),
        "test": str(Path(OUTPUT_PATH, "images/test").absolute()),
        "nc": len(class_map),
        "names": list(class_map.keys()),
    }

    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    with open(Path(OUTPUT_PATH) / "data.yaml", "w") as f:
        yaml.dump(data, f)


# -------- MAIN --------
def main():
    xml_files = list(Path(VOC_PATH).glob("*.xml"))
    # Ensure deterministic splits by fixing the random seed from config
    random.seed(SEED)
    random.shuffle(xml_files)

    class_map = build_class_map(xml_files)
    print("Classes found:", class_map)

    n = len(xml_files)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])

    splits = {
        "train": xml_files[:n_train],
        "val": xml_files[n_train:n_train + n_val],
        "test": xml_files[n_train + n_val:],
    }

    for split, files in splits.items():
        for f in files:
            save(f, split, class_map)

    create_yaml(class_map)
    print("Done")


if __name__ == "__main__":
    main()