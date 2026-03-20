from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


RETAIL_RELEVANT_CLASSES = [
    "person",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "cell phone",
]


def yolo_box(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    cx = (x + (w / 2.0)) / img_w
    cy = (y + (h / 2.0)) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset with retail-relevant COCO classes.")
    parser.add_argument("--coco-root", default="data/datasets/coco", help="COCO assets root directory")
    parser.add_argument(
        "--output-root",
        default="data/datasets/retail_yolo",
        help="Destination YOLO dataset directory",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=600,
        help="Maximum number of relevant images to include",
    )
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation ratio (0-0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    val_split = min(0.5, max(0.05, args.val_split))

    coco_root = Path(args.coco_root).resolve()
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    images_dir = coco_root / "val2017"
    manifest_file = coco_root / "retail_relevant_val_manifest.json"

    if not ann_file.exists() or not images_dir.exists():
        raise FileNotFoundError(
            "COCO data not found. Run scripts/download_retail_relevant_data.py first."
        )

    payload = json.loads(ann_file.read_text(encoding="utf-8"))
    categories = {cat["id"]: cat["name"] for cat in payload["categories"]}
    keep_cat_ids = {cat_id for cat_id, name in categories.items() if name in RETAIL_RELEVANT_CLASSES}

    class_to_idx = {name: i for i, name in enumerate(RETAIL_RELEVANT_CLASSES)}
    cat_id_to_idx = {
        cat_id: class_to_idx[categories[cat_id]]
        for cat_id in keep_cat_ids
        if categories[cat_id] in class_to_idx
    }

    images_by_id = {img["id"]: img for img in payload["images"]}
    anns_by_img: dict[int, list[dict[str, object]]] = defaultdict(list)
    for ann in payload["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in keep_cat_ids:
            continue
        anns_by_img[ann["image_id"]].append(ann)

    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        candidate_img_ids = [item["id"] for item in manifest if item["id"] in anns_by_img]
    else:
        candidate_img_ids = [img_id for img_id in anns_by_img.keys() if img_id in images_by_id]

    random.shuffle(candidate_img_ids)
    selected_ids = candidate_img_ids[: max(10, args.sample_count)]
    if not selected_ids:
        raise RuntimeError("No suitable annotated images found for selected retail classes.")

    split_idx = int(len(selected_ids) * (1.0 - val_split))
    train_ids = selected_ids[:split_idx]
    val_ids = selected_ids[split_idx:]
    if not val_ids:
        val_ids = train_ids[-1:]
        train_ids = train_ids[:-1]

    out = Path(args.output_root).resolve()
    for rel in [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    def materialize(split_name: str, image_ids: list[int]) -> int:
        kept = 0
        for image_id in image_ids:
            image_meta = images_by_id.get(image_id)
            if image_meta is None:
                continue
            src_image = images_dir / image_meta["file_name"]
            if not src_image.exists():
                continue
            img_w = int(image_meta["width"])
            img_h = int(image_meta["height"])

            label_lines: list[str] = []
            for ann in anns_by_img.get(image_id, []):
                cat_id = ann["category_id"]
                cls_idx = cat_id_to_idx.get(cat_id)
                if cls_idx is None:
                    continue
                x, y, w, h = ann["bbox"]
                cx, cy, nw, nh = yolo_box(float(x), float(y), float(w), float(h), img_w, img_h)
                if nw <= 0 or nh <= 0:
                    continue
                label_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            if not label_lines:
                continue

            dst_image = out / "images" / split_name / image_meta["file_name"]
            dst_label = out / "labels" / split_name / f"{Path(image_meta['file_name']).stem}.txt"
            shutil.copy2(src_image, dst_image)
            dst_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            kept += 1
        return kept

    train_count = materialize("train", train_ids)
    val_count = materialize("val", val_ids)
    if train_count < 5 or val_count < 2:
        raise RuntimeError(
            f"Dataset too small after filtering (train={train_count}, val={val_count}). "
            "Increase sample-count."
        )

    yaml_text = "\n".join(
        [
            f"path: {out.as_posix()}",
            "train: images/train",
            "val: images/val",
            f"names: {RETAIL_RELEVANT_CLASSES}",
            "",
        ]
    )
    (out / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    summary = {
        "train_images": train_count,
        "val_images": val_count,
        "classes": RETAIL_RELEVANT_CLASSES,
        "dataset_yaml": str((out / "dataset.yaml")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
