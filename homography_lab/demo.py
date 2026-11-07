"""Replay motion geometry from cached automatic proposals, without videos or neural models."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB / "experiments/scenealigner"))
import cv2
from motion_candidates import propagate_candidates, validate_bank
from motion_config import OFFICE, dump_json
from motion_export import exportable_calibrations
from motion_joint import solve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    bank = validate_bank(json.loads((OFFICE / "automatic-candidates.json").read_text(encoding="utf-8")))
    pairs = json.loads((LAB / "examples/motion/pairs.json").read_text(encoding="utf-8"))
    lenses = json.loads((OFFICE / "lenses.json").read_text(encoding="utf-8"))["cameras"]
    propagated = deepcopy(bank)
    propagate_candidates(propagated, pairs, lenses)
    result = solve(propagated, pairs, cv2.imread(str(OFFICE / "plan-walls.png"), 0))
    args.output.mkdir(parents=True, exist_ok=False)
    dump_json(args.output / "joint.json", result)
    dump_json(args.output / "accepted.json", exportable_calibrations(result))
    protocol = {
        "mode": "cached_neural_outputs_geometry_replay",
        "video_inference_executed": False,
        "neural_models_loaded": "torch" in sys.modules or "ultralytics" in sys.modules,
        "absolute_location_verified": result["absolute_location_verified"],
        "exported_cameras": sorted(exportable_calibrations(result)),
        "components": [
            {"cameras": c["cameras"], "status": c["status"], "reasons": c["reasons"]}
            for c in result["components"]
        ],
    }
    dump_json(args.output / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
