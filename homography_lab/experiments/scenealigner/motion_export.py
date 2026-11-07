"""Export only explicitly accepted calibrations; rejected hypotheses stay diagnostic."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
if __package__:
    from .motion_config import dump_json
else:
    from motion_config import dump_json


def exportable_calibrations(result):
    if not (
        result.get("automatic_deployment_ready") is True and result.get("absolute_location_verified") is True
    ):
        return {}
    if any(c.get("status") != "accepted" for c in result.get("components", [])):
        raise ValueError("Ready flag conflicts with unaccepted components")
    accepted = result.get("accepted_calibrations", {})
    supported = {camera for component in result.get("components", []) for camera in component["cameras"]}
    if set(accepted) - supported:
        raise ValueError("Accepted cameras require an accepted component")
    for camera, calibration in accepted.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", camera):
            raise ValueError("Unsafe camera identifier")
        h = np.asarray(calibration["homography"], float)
        if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
            raise ValueError("Invalid accepted homography")
    return accepted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = json.loads(args.joint.read_text(encoding="utf-8"))
    accepted = exportable_calibrations(result)
    args.output.mkdir(parents=True, exist_ok=False)
    for camera, calibration in accepted.items():
        dump_json(args.output / (camera + "-calibration.json"), calibration)
    dump_json(
        args.output / "manifest.json",
        {
            "exported_cameras": sorted(accepted),
            "status": "exported" if accepted else "no_accepted_calibrations",
            "diagnostic_candidates_exported": False,
        },
    )


if __name__ == "__main__":
    main()
