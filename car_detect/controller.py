"""Controller placeholder (will be implemented in Step 7)."""

from __future__ import annotations
from typing import Any


class Controller:
    """Connects UI to detection backends (to be implemented in Step 7)."""

    def __init__(self) -> None:
        self._dummy = True

    def run_detection(self, cfg: Any) -> None:
        raise NotImplementedError("Will be implemented in Step 7.")