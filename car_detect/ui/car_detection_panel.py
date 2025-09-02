from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout


class CarDetectionPanel(QWidget):
    """Right-side control panel (Step 0 placeholder).

    Tooltips explain future controls to orient the user before they appear.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        lbl = QLabel(
            "Car Detection Controls (coming in Steps 5–7)\n"
            "• Template Matching params (threshold, scales, NMS)\n"
            "• YOLO params (conf, iou, size/ratio filters)\n"
            "• Tiling params (tile, overlap, global NMS)\n"
            "\nThis panel will emit detectRequested(cfg) later.",
            self,
        )
        lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lbl.setWordWrap(True)
        lbl.setToolTip(
            "This is a placeholder. In Step 6 we will add real controls with detailed tooltips."
        )
        layout.addWidget(lbl)
        layout.addStretch()