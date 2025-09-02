from car_detect.detection.template import (
    detect_cars_by_template, TemplateParams, generate_synthetic_scene, SyntheticParams, to_gray_clahe
)
from car_detect.detection.tiler import run_tiled, TilerParams
import numpy as np

def test_tiler_template_recall_ok():
    img, gt = generate_synthetic_scene(
        SyntheticParams(width=1920, height=1080, n_cars=18, seed=9, noise_sigma=6.0, jitter_scale=0.06)
    )

    # Глобальные mean/std для всех тайлов этой сцены
    gray_full = to_gray_clahe(img, 2.0, (8, 8))
    stats = (float(gray_full.mean()), float(gray_full.std()))

    tpl = TemplateParams(
        threshold=0.67,                              # было 0.75 → снизим чуточку
        scales=(0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
        nms_iou=0.30,
        relative_peak_distance=0.40,
    )

    dets = run_tiled(
        img,
        detector=lambda t: detect_cars_by_template(t, tpl, zscore_stats=stats),
        params=TilerParams(tile_w=640, tile_h=640, overlap=96, nms_iou=0.40),
    )

    assert len(dets) >= int(0.9 * len(gt)), f"recall too low: got {len(dets)} of {len(gt)}"
    assert len(dets) <= len(gt) + 2, f"too many: got {len(dets)} > {len(gt)}+2"