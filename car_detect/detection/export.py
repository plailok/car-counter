from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import csv
import json

from .types import Detection


def save_detections_csv(path: str | Path, dets: List[Detection]) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["x1","y1","x2","y2","w","h","score","label"])
        w.writeheader()
        for d in dets:
            w.writerow({
                "x1": round(d.x1, 3), "y1": round(d.y1, 3),
                "x2": round(d.x2, 3), "y2": round(d.y2, 3),
                "w":  round(d.width, 3), "h":  round(d.height, 3),
                "score": round(d.score, 6), "label": d.label,
            })


def save_detections_coco(
    path: str | Path,
    dets: List[Detection],
    image_size: Tuple[int, int],  # (width, height)
    categories: Dict[int, str] | None = None,
) -> None:
    """Минимальный COCO (один image)."""
    W, H = image_size
    categories = categories or {1: "car"}
    coco: Dict[str, Any] = {
        "images": [{"id": 1, "file_name": Path(path).stem, "width": W, "height": H}],
        "annotations": [],
        "categories": [{"id": cid, "name": name} for cid, name in categories.items()],
    }
    ann_id = 1
    for d in dets:
        bbox = [float(d.x1), float(d.y1), float(d.width), float(d.height)]
        coco["annotations"].append({
            "id": ann_id,
            "image_id": 1,
            "category_id": 1,   # один класс — "car"
            "bbox": bbox,       # [x,y,w,h]
            "area": float(d.width * d.height),
            "iscrowd": 0,
            "score": float(d.score),
        })
        ann_id += 1

    Path(path).write_text(json.dumps(coco, indent=2), encoding="utf-8")
