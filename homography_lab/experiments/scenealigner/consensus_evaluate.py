"""Evaluate previously selected consensus hypotheses; reference coordinates are never used to fuse them."""

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate import evaluate_camera
from temporal_evaluate import accuracy_pass, transferred_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--consensus", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new output file")
    consensus = json.loads(args.consensus.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    output = {
        "evaluation_only": True,
        "reference_coordinates_used_for_consensus_selection": False,
        "source": str(args.consensus.resolve()),
        "trials": [],
    }
    for trial in consensus["trials"]:
        row = {key: trial[key] for key in ["subset", "threshold_fraction", "threshold_px", "status"]}
        if trial["clusters"]:
            chosen = trial["clusters"][0]
            transferred = transferred_reference(reference, chosen["input_affine"])
            score = evaluate_camera(chosen["calibration"], transferred)
            row.update(
                score=score,
                passes_reference_criteria=accuracy_pass(score),
                support_frames=chosen["support_frames"],
                input_frames=len(trial["input_frames"]),
                medoid_case=chosen["medoid_case"],
                medoid_rank=chosen["medoid_rank"],
            )
        else:
            row.update(passes_reference_criteria=False, score=None)
        output["trials"].append(row)
        print(
            row["subset"],
            row["threshold_fraction"],
            (row.get("score") or {}).get("median_error_px"),
            row["passes_reference_criteria"],
            row["status"],
        )
    default = consensus["default_threshold_fraction"]
    first = next(
        trial
        for trial in consensus["trials"]
        if trial["subset"] == "alternating_A" and trial["threshold_fraction"] == default
    )
    second = next(
        trial
        for trial in consensus["trials"]
        if trial["subset"] == "alternating_B" and trial["threshold_fraction"] == default
    )
    if first["clusters"] and second["clusters"]:
        a = np.asarray(first["clusters"][0]["signature"])
        b = np.asarray(second["clusters"][0]["signature"])
        output["disjoint_group_projection_rms_px"] = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
