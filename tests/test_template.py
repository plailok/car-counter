import numpy as np

from car_detect.detection.template import (
    TemplateParams,
    detect_cars_by_template,
    generate_synthetic_scene,
    SyntheticParams,
)


def test_template_on_synthetic_exact_count():
    img, gt = generate_synthetic_scene(
        SyntheticParams(
            width=1280, height=720, n_cars=10, seed=7, noise_sigma=6.0, jitter_scale=0.06
        )
    )
    params = TemplateParams(
        threshold=0.72,
        scales=(0.9, 1.0, 1.1),
        nms_iou=0.4,
        base_sizes=((120, 50), (50, 120)),
        clahe_clip_limit=2.0,
        clahe_tile_grid=(8, 8),
        peak_min_distance=3,
    )
    dets = detect_cars_by_template(img, params)
    assert len(dets) == len(gt), f"Expected {len(gt)} cars, got {len(dets)}"

    # Optional: sanity check that scores are reasonable
    assert all(0.5 <= d.score <= 1.0 for d in dets)