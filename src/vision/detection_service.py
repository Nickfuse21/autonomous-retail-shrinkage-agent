from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any


@dataclass
class DetectionResult:
    detections: list[dict[str, Any]]
    model_ready: bool
    message: str
    device: str


class DetectionService:
    """Lazy-loaded object detector wrapper.

    Uses YOLOv8 models if available, but degrades gracefully when dependencies
    are not installed so the rest of the app continues to work.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._load_errors: dict[str, str] = {}
        self._device: str = "unknown"
        self._model_path = os.getenv("DETECTOR_MODEL_PATH", "yolov8s.pt")
        self._forced_device = os.getenv("DETECTOR_DEVICE")

    def detect(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.3,
        max_detections: int = 20,
        model_variant: str = "pretrained",
    ) -> DetectionResult:
        model_path = self._resolve_model_path(model_variant)
        model = self._ensure_model(model_path)
        if model is None:
            message = self._load_errors.get(model_path) or "Detector unavailable."
            return DetectionResult(
                detections=[],
                model_ready=False,
                message=message,
                device=self._device,
            )

        try:
            from PIL import Image
        except Exception:
            return DetectionResult(
                detections=[],
                model_ready=False,
                message="Pillow is required for frame decoding.",
                device=self._device,
            )

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            results = model.predict(
                image,
                conf=conf_threshold,
                verbose=False,
                max_det=max_detections,
                device=self._device,
            )
            parsed: list[dict[str, Any]] = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls_idx = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    label = result.names.get(cls_idx, str(cls_idx))
                    parsed.append(
                        {
                            "label": label,
                            "confidence": round(score, 4),
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                        }
                    )
            return DetectionResult(
                detections=parsed[:max_detections],
                model_ready=True,
                message="ok",
                device=self._device,
            )
        except Exception as exc:
            return DetectionResult(
                detections=[],
                model_ready=False,
                message=f"detection_failed: {exc.__class__.__name__}",
                device=self._device,
            )

    def _ensure_model(self, model_path: str) -> Any | None:
        if model_path in self._models:
            return self._models[model_path]
        if model_path in self._load_errors:
            return None
        try:
            import torch
            from ultralytics import YOLO

            if self._forced_device:
                self._device = self._forced_device
            else:
                self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            # YOLO downloads weights on first run; this may take some time once.
            loaded = YOLO(model_path)
            self._models[model_path] = loaded
            return loaded
        except Exception as exc:
            self._device = "cpu"
            self._load_errors[model_path] = (
                f"Could not load YOLO detector ({exc.__class__.__name__}). "
                "Install dependencies and retry."
            )
            return None

    def _resolve_model_path(self, model_variant: str) -> str:
        variant = model_variant.lower().strip()
        if variant == "custom":
            return self._model_path
        # Pretrained COCO model for broad object classes.
        return "yolov8s.pt"
