from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path


COCO_ANNOTATIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGES_URL = "https://images.cocodataset.org/zips/val2017.zip"
RETAIL_RELEVANT_CLASSES = {
    "person",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "cell phone",
}


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] Exists: {target}")
        return
    if target.exists() and target.stat().st_size == 0:
        target.unlink(missing_ok=True)
    print(f"[download] {url} -> {target}")
    tmp_target = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        with tmp_target.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=1024 * 1024)
    tmp_target.replace(target)


def extract_file(zip_path: Path, output_dir: Path) -> None:
    print(f"[extract] {zip_path} -> {output_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


def build_sample_manifest(
    instances_json_path: Path,
    output_manifest: Path,
    sample_count: int,
) -> None:
    payload = json.loads(instances_json_path.read_text(encoding="utf-8"))
    categories = {cat["id"]: cat["name"] for cat in payload["categories"]}
    keep_cat_ids = {cid for cid, name in categories.items() if name in RETAIL_RELEVANT_CLASSES}
    annotations = payload["annotations"]
    images = {img["id"]: img for img in payload["images"]}

    matched_image_ids: set[int] = set()
    for ann in annotations:
        if ann["category_id"] in keep_cat_ids:
            matched_image_ids.add(ann["image_id"])

    matched_images = [images[iid] for iid in matched_image_ids if iid in images]
    random.shuffle(matched_images)
    selected = matched_images[:sample_count]

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"[done] Wrote manifest with {len(selected)} images: {output_manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download retail-relevant validation data for detector testing."
    )
    parser.add_argument("--output", default="data/datasets/coco", help="Output dataset folder")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=300,
        help="Number of retail-relevant validation images to include in manifest",
    )
    args = parser.parse_args()

    out = Path(args.output).resolve()
    archives = out / "archives"
    download_file(COCO_ANNOTATIONS_URL, archives / "annotations_trainval2017.zip")
    download_file(COCO_VAL_IMAGES_URL, archives / "val2017.zip")

    annotations_root = out / "annotations"
    val_root = out / "val2017"
    if not (annotations_root / "instances_val2017.json").exists():
        extract_file(archives / "annotations_trainval2017.zip", out)
    if not val_root.exists() or not any(val_root.iterdir()):
        extract_file(archives / "val2017.zip", out)

    build_sample_manifest(
        instances_json_path=annotations_root / "instances_val2017.json",
        output_manifest=out / "retail_relevant_val_manifest.json",
        sample_count=max(1, args.sample_count),
    )
    print("[ok] Dataset assets ready.")


if __name__ == "__main__":
    main()
