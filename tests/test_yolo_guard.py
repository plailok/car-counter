import importlib.util
import numpy as np
import pytest

from car_detect.detection.yolo import detect_cars_by_yolo, YOLOParams, YoloNotAvailableError


def test_yolo_guard_when_not_installed():
    # Skip this test if ultralytics is actually installed.
    if importlib.util.find_spec("ultralytics") is not None:
        pytest.skip("Ultralytics installed; guard test not applicable.")

    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(YoloNotAvailableError) as ei:
        detect_cars_by_yolo(dummy, YOLOParams(weights="yolov8n.pt"))
    assert "pip install ultralytics" in str(ei.value)