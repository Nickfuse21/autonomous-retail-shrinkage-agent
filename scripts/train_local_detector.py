from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local YOLO retail detector on RTX 3050.")
    parser.add_argument(
        "--data",
        default="data/datasets/retail_yolo/dataset.yaml",
        help="Path to YOLO dataset.yaml",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument(
        "--device",
        default="0",
        help="YOLO device string. Use '0' for first GPU, or 'cpu'.",
    )
    parser.add_argument(
        "--project",
        default="runs/train",
        help="Output root for training runs",
    )
    parser.add_argument("--name", default="", help="Optional run name")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    cuda_available = torch.cuda.is_available()
    if args.device != "cpu" and not cuda_available:
        print("[warn] CUDA not available. Switching to CPU.")
        args.device = "cpu"

    run_name = args.name or f"retail-yolo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model = YOLO(args.model)

    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=run_name,
        pretrained=True,
        cache=True,
        amp=True,
        patience=12,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
    )

    summary = {
        "run_name": run_name,
        "device": args.device,
        "cuda_available": cuda_available,
        "results_dir": str(result.save_dir),
        "best_weights": str(Path(result.save_dir) / "weights" / "best.pt"),
        "last_weights": str(Path(result.save_dir) / "weights" / "last.pt"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
