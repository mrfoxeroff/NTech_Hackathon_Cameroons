"""Post-run comparison: motion consistency is distinct from absolute floorplan accuracy."""

import argparse
import json
from pathlib import Path

from evaluate import evaluate_camera
from motion_joint import error_summary
from motion_pairs import balanced_matches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint", type=Path, required=True)
    p.add_argument("--baseline-candidates", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    joint = json.loads(args.joint.read_text())
    bank = json.loads(args.baseline_candidates.read_text())
    pairs = json.loads(args.pairs.read_text())
    baseline = {c: v["candidates"][0]["calibration"] for c, v in bank["cameras"].items() if v["candidates"]}
    report = {
        "evaluation_only": True,
        "manual_reference_used_for_inference": False,
        "pairs_are_proposals_not_person_ID_ground_truth": True,
        "reference_limitation": "Existing approximate manual JSON, not surveyed people locations.",
        "motion": {},
        "absolute_map": {},
    }
    for name, edge in pairs["edges"].items():
        rows = balanced_matches(
            [
                r
                for r in pairs["pairs"]
                if r["cameras"] == edge["cameras"]
                and r["split"] == "holdout"
                and all(v is not None for v in r["points"])
            ]
        )
        if all(c in joint["calibrations"] for c in edge["cameras"]):
            report["motion"][name] = {
                "independent_baseline": error_summary(rows, baseline),
                "joint_proposal": error_summary(rows, joint["calibrations"]),
                "relative_edge_accepted": edge["accepted"],
            }
    if args.reference:
        for camera, calibration in joint["calibrations"].items():
            path = args.reference / (camera + "_calibration.json")
            if path.exists():
                ref = json.loads(path.read_text())
                report["absolute_map"][camera] = {
                    "baseline": evaluate_camera(baseline[camera], ref),
                    "joint_proposal": evaluate_camera(calibration, ref),
                }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "motion": report["motion"],
                "absolute_medians": {
                    c: {k: v["median_error_px"] for k, v in d.items()}
                    for c, d in report["absolute_map"].items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
