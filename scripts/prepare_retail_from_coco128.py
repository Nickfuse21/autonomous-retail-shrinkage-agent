from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

# COCO class ids -> compact retail class ids
COCO_TO_RETAIL = {
    0: 0,   # person
    24: 1,  # backpack
    26: 2,  # handbag
    28: 3,  # suitcase
    39: 4,  # bottle
    41: 5,  # cup
    67: 6,  # cell phone
}
RETAIL_NAMES = ["person", "backpack", "handbag", "suitcase", "bottle", "cup", "cell phone"]


def download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"[skip] exists: {path}")
        return
    if path.exists():
        path.unlink(missing_ok=True)
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, path)


def copy_and_filter_split(src_root: Path, dst_root: Path, split: str) -> tuple[int, int]:
    src_images = src_root / "images" / split
    src_labels = src_root / "labels" / split
    dst_images = dst_root / "images" / split
    dst_labels = dst_root / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    img_count = 0
    box_count = 0
    for img in src_images.glob("*.*"):
        label = src_labels / f"{img.stem}.txt"
        if not label.exists():
            continue
        out_lines: list[str] = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            mapped = COCO_TO_RETAIL.get(cls)
            if mapped is None:
                continue
            out_lines.append(f"{mapped} {' '.join(parts[1:])}")
        if not out_lines:
            continue
        shutil.copy2(img, dst_images / img.name)
        (dst_labels / f"{img.stem}.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        img_count += 1
        box_count += len(out_lines)
    return img_count, box_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare compact retail detector dataset from COCO128.")
    parser.add_argument("--output-root", default="data/datasets/retail_yolo_coco128")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    archives = output_root / "archives"
    zip_path = archives / "coco128.zip"
    raw_dir = output_root / "raw"
    src_root = raw_dir / "coco128"

    download_if_missing(COCO128_URL, zip_path)
    if not src_root.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    _, train_boxes = copy_and_filter_split(src_root, output_root, "train2017")
    # Split from train into train/val for quick local experiments.
    # Move 20% of files to val.
    val_dir = output_root / "images" / "val2017"
    val_lbl_dir = output_root / "labels" / "val2017"
    val_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_root / "images" / "train2017"
    train_lbl_dir = output_root / "labels" / "train2017"
    all_train = sorted(train_dir.glob("*.*"))
    moved_to_val = 0
    for i, img in enumerate(all_train):
        if i % 5 != 0:
            continue
        lbl = train_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.move(str(img), str(val_dir / img.name))
            shutil.move(str(lbl), str(val_lbl_dir / lbl.name))
            moved_to_val += 1

    yaml_text = "\n".join(
        [
            f"path: {output_root.as_posix()}",
            "train: images/train2017",
            "val: images/val2017",
            f"names: {RETAIL_NAMES}",
            "",
        ]
    )
    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(yaml_text, encoding="utf-8")

    summary = {
        "dataset_yaml": str(dataset_yaml),
        "classes": RETAIL_NAMES,
        "train_images": len(list((output_root / "images" / "train2017").glob("*.*"))),
        "val_images": len(list((output_root / "images" / "val2017").glob("*.*"))),
        "train_boxes": train_boxes,
        "moved_to_val": moved_to_val,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
