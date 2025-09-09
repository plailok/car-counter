"""Datatypes for detections (Step 1 will define Detection dataclass)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class Detection:
    """Single detection in absolute pixel coordinates (x1, y1, x2, y2).

    Attributes
    ----------
    x1, y1, x2, y2 : float
        Top-left (x1, y1) and bottom-right (x2, y2) corners in pixels.
        Expected invariant: x2 > x1, y2 > y1.
    score : float
        Confidence score in [0, 1].
    label : str
        Class label (default: "car").
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "car"

    angle: Optional[float] = None      # degrees, image coords, -90..90 (0 = горизонталь)
    r_width: Optional[float] = None    # width along principal axis (rotated box)
    r_height: Optional[float] = None   # height along principal axis

    # ---- geometry helpers -------------------------------------------------
    @property
    def width(self) -> float:
        """Box width (pixels)."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Box height (pixels)."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Box area (px^2)."""
        return self.width * self.height

    @property
    def aspect(self) -> float:
        """Aspect ratio width / height (returns np.inf if height == 0)."""
        return self.width / self.height if self.height > 0 else float("inf")

    # ---- conversions ------------------------------------------------------
    def to_xyxy(self) -> np.ndarray:
        """Return [x1, y1, x2, y2] as float32 array."""
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def to_csv_row(self) -> Dict[str, object]:
        """Return a dict row suitable for CSV export."""
        return {
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
            "score": float(self.score),
            "label": self.label,
        }

    def clip(self, image_w: int, image_h: int) -> "Detection":
        """Return a new detection clipped to image bounds [0, W)×[0, H)."""
        x1 = float(np.clip(self.x1, 0, image_w - 1))
        y1 = float(np.clip(self.y1, 0, image_h - 1))
        x2 = float(np.clip(self.x2, 0, image_w - 1))
        y2 = float(np.clip(self.y2, 0, image_h - 1))
        return Detection(x1, y1, x2, y2, self.score, self.label)

    # ---- factories --------------------------------------------------------
    @staticmethod
    def from_xyxy(box: Iterable[float], score: float, label: str = "car") -> "Detection":
        """Create Detection from [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = [float(v) for v in box]
        return Detection(x1, y1, x2, y2, float(score), label)

    @staticmethod
    def from_xywh(
        xywh: Iterable[float], score: float, label: str = "car"
    ) -> "Detection":
        """Create Detection from [x, y, w, h] (top-left + size)."""
        x, y, w, h = [float(v) for v in xywh]
        return Detection(x, y, x + w, y + h, float(score), label)

    @staticmethod
    def list_to_arrays(dets: List["Detection"]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Convert list of detections to (boxes[N,4], scores[N], labels[N])."""
        if len(dets) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
            )
        boxes = np.stack([d.to_xyxy() for d in dets]).astype(np.float32, copy=False)
        scores = np.array([d.score for d in dets], dtype=np.float32)
        labels = [d.label for d in dets]
        return boxes, scores, labels

    @staticmethod
    def arrays_to_list(
        boxes: np.ndarray, scores: np.ndarray, labels: Iterable[str] | None = None
    ) -> List["Detection"]:
        """Convert arrays back to a list of detections."""
        n = int(boxes.shape[0])
        if labels is None:
            labels = ["car"] * n
        return [
            Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), str(l))
            for b, s, l in zip(boxes, scores, labels)
        ]
