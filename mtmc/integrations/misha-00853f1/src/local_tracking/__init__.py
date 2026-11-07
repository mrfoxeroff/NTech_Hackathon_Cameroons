"""Per-camera observations; no cross-camera identity decisions."""

import os
from pathlib import Path

# Applies to both CLI and Python API; don't let upstream modify unrelated user settings.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".runtime" / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

__version__ = "0.1.0"
