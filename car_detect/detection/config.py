from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import json
from pathlib import Path

from .template import TemplateParams
from .yolo import YOLOParams
from .tiler import TilerParams


@dataclass
class TemplateConfig:
    threshold: float = 0.70
    scales: Tuple[float, ...] = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
    nms_iou: float = 0.30
    base_sizes: Tuple[Tuple[int, int], ...] = ((120, 50), (50, 120))
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    peak_min_distance: int = 3
    relative_peak_distance: float = 0.40
    max_peaks_per_template: int = 400
    min_std: float = 14.0
    edge_grad_thresh: float = 40.0
    min_edge_density: float = 0.03

    def to_params(self) -> TemplateParams:
        return TemplateParams(
            threshold=self.threshold,
            scales=self.scales,
            nms_iou=self.nms_iou,
            base_sizes=self.base_sizes,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_grid=self.clahe_tile_grid,
            peak_min_distance=self.peak_min_distance,
            relative_peak_distance=self.relative_peak_distance,
            max_peaks_per_template=self.max_peaks_per_template,
            min_std=self.min_std,
            edge_grad_thresh=self.edge_grad_thresh,
            min_edge_density=self.min_edge_density,
        )


@dataclass
class YOLOConfig:
    weights: str = "yolov8n.pt"
    conf: float = 0.25
    iou: float = 0.50
    imgsz: int = 1280
    keep_classes: Optional[Tuple[int, ...]] = (2,)  # COCO: car=2
    width_range: Tuple[float, float] = (80.0, 170.0)
    height_range: Tuple[float, float] = (35.0, 90.0)
    aspect: float = 120.0 / 50.0
    aspect_tol: float = 0.8
    label: str = "car_yolo"

    def to_params(self) -> YOLOParams:
        return YOLOParams(
            weights=self.weights,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            keep_classes=self.keep_classes,
            width_range=self.width_range,
            height_range=self.height_range,
            aspect=self.aspect,
            aspect_tol=self.aspect_tol,
            label=self.label,
        )


@dataclass
class TilerConfig:
    enabled: bool = False
    tile_w: int = 640
    tile_h: int = 640
    overlap: int = 96
    nms_iou: float = 0.40
    cap_per_tile: int = 400
    use_ownership: bool = True

    def to_params(self) -> TilerParams:
        return TilerParams(
            tile_w=self.tile_w,
            tile_h=self.tile_h,
            overlap=self.overlap,
            nms_iou=self.nms_iou,
            cap_per_tile=self.cap_per_tile,
            use_ownership=self.use_ownership,
        )


@dataclass
class PipelineConfig:
    backend: str = "template"  # "template" | "yolo"
    template: TemplateConfig = field(default_factory=TemplateConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    tiler: TilerConfig = field(default_factory=TilerConfig)

    # ---- JSON I/O ----
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineConfig":
        tpl = TemplateConfig(**d.get("template", {}))
        yl = YOLOConfig(**d.get("yolo", {}))
        tl = TilerConfig(**d.get("tiler", {}))
        return PipelineConfig(backend=d.get("backend", "template"), template=tpl, yolo=yl, tiler=tl)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load_json(path: str | Path) -> "PipelineConfig":
        return PipelineConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---- Presets (примерные профили) ----

def preset_template_strict() -> PipelineConfig:
    pc = PipelineConfig()
    pc.backend = "template"
    pc.template.threshold = 0.75
    pc.template.scales = (0.9, 1.0, 1.1)
    pc.template.nms_iou = 0.30
    pc.tiler.enabled = False
    return pc


def preset_template_tiled() -> PipelineConfig:
    pc = PipelineConfig()
    pc.backend = "template"
    pc.template.threshold = 0.70
    pc.template.scales = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
    pc.template.nms_iou = 0.30
    pc.tiler.enabled = True
    pc.tiler.tile_w = 640
    pc.tiler.tile_h = 640
    pc.tiler.overlap = 96
    pc.tiler.nms_iou = 0.40
    return pc


def preset_yolo_default() -> PipelineConfig:
    pc = PipelineConfig()
    pc.backend = "yolo"
    pc.yolo.conf = 0.25
    pc.yolo.iou = 0.5
    pc.tiler.enabled = False
    return pc
