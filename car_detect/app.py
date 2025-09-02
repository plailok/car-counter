from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QDockWidget,
    QWidget,
    QVBoxLayout,
)

from .ui.car_detection_panel import CarDetectionPanel


class ImageView(QWidget):
    """Minimal image viewer for Step 0.

    This is a placeholder. In later steps, it will support overlays for detections.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label = QLabel(
            "Drop an image here or use File → Open image…\n(Step 0 placeholder)",
            self,
        )
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)

        # Enable drag & drop
        self.setAcceptDrops(True)

    def set_image(self, path: Path) -> None:
        """Load and display an image by path."""
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._label.setText(f"Failed to load image:\n{path}")
        else:
            self._label.setPixmap(pixmap)
            self._label.setScaledContents(True)
            self._label.setToolTip(str(path))

    # Drag & drop plumbing
    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        self.set_image(path)


class MainWindow(QMainWindow):
    """Main application window (Step 0)."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Car Counter — Step 0")
        self.resize(1200, 800)

        # Central image view
        self.image_view = ImageView(self)
        self.setCentralWidget(self.image_view)

        # Right dock with (future) controls
        self._create_right_dock()

        # Menu
        self._create_menu()

        # Status bar
        self.statusBar().showMessage("Ready (Step 0)")

    def _create_right_dock(self) -> None:
        dock = QDockWidget("Car Detection", self)
        dock.setObjectName("CarDetectionDock")
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)

        panel = CarDetectionPanel(self)
        dock.setWidget(panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open image…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open an image file")
        open_action.triggered.connect(self._open_image_dialog)
        file_menu.addAction(open_action)

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _open_image_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path_str:
            return
        self.image_view.set_image(Path(path_str))


def main() -> None:
    """Entry point."""
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()