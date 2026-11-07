"""Run the integrated misha local stage without installing another Python package."""

import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from local_tracking.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
