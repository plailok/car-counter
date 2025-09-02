import numpy as np
import pytest

from car_detect.detection.nms import iou_xyxy, nms
from car_detect.detection.types import Detection


def test_iou_known_value():
    # Box A: (0,0)-(100,100) area = 10000
    # Box B: (50,50)-(150,150) area = 10000
    # Intersection = 50*50 = 2500; Union = 10000+10000-2500 = 17500
    # IoU = 2500/17500 = 1/7 ≈ 0.14285715
    a = np.array([[0, 0, 100, 100]], dtype=np.float32)
    b = np.array([[50, 50, 150, 150]], dtype=np.float32)
    iou = iou_xyxy(a, b)
    assert iou.shape == (1, 1)
    assert np.isclose(iou[0, 0], 1.0 / 7.0, atol=1e-6)


def test_iou_pairwise_shape_and_values():
    a = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
    iou = iou_xyxy(a, b)
    assert iou.shape == (2, 2)
    # First a vs first b = perfect overlap
    assert np.isclose(iou[0, 0], 1.0, atol=1e-6)
    # Second a (5x5) inside first b (10x10): IoU = 25 / 100 = 0.25
    assert np.isclose(iou[1, 0], 0.25, atol=1e-6)
    # First a vs second b: overlap (5..10)x(5..10)=25; union=100+100-25=175 => 25/175
    assert np.isclose(iou[0, 1], 25.0 / 175.0, atol=1e-6)


def test_nms_suppresses_overlapping():
    boxes = np.array(
        [
            [0, 0, 100, 100],       # A (high score)
            [10, 10, 110, 110],     # B (heavy overlap with A)
            [200, 200, 300, 300],   # C (far)
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    keep = nms(boxes, scores, iou_threshold=0.5)
    kept_boxes = boxes[keep]

    assert keep.tolist() == [0, 2], "Should keep A and C; B suppressed by overlap with A"
    assert kept_boxes.shape[0] == 2


def test_nms_tie_break_is_deterministic():
    # Equal scores; expect lower index first (deterministic)
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [0, 0, 10, 10],  # identical box
        ],
        dtype=np.float32,
    )
    scores = np.array([0.5, 0.5], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.3)
    # One of them must remain; with equal scores, index 0 wins.
    assert keep[0] == 0
    assert len(keep) == 1


def test_detection_helpers_roundtrip():
    dets = [
        Detection.from_xyxy([1, 2, 11, 12], 0.9, "car"),
        Detection.from_xywh([5, 5, 20, 10], 0.8, "car"),
    ]
    boxes, scores, labels = Detection.list_to_arrays(dets)
    dets2 = Detection.arrays_to_list(boxes, scores, labels)
    assert len(dets2) == len(dets)
    for d1, d2 in zip(dets, dets2):
        assert d1.to_xyxy().tolist() == d2.to_xyxy().tolist()
        assert pytest.approx(d1.score) == d2.score
        assert d1.label == d2.label