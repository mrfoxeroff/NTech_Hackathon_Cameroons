"""Reconstruct detected overlap components independently, then search plan fragments."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def validate_groups(groups, names):
    flattened = [name for group in groups for name in group]
    if (
        not groups
        or any(not group for group in groups)
        or len(flattened) != len(set(flattened))
        or set(flattened) != set(names)
    ):
        raise ValueError("Groups must be a disjoint partition of all input camera names")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--floor-masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matcher", choices=["mutual", "topk"], default="topk")
    parser.add_argument("--reconstruction-only", action="store_true")
    args = parser.parse_args()
    graph = json.loads(args.overlap.read_text(encoding="utf-8"))
    groups = graph["groups"]
    images = {path.stem: path for path in args.images.glob("*.jpg")}
    validate_groups(groups, images)
    args.output.mkdir(parents=True, exist_ok=False)
    assignments = {
        "groups": groups,
        "overlap_report": str(args.overlap.resolve()),
        "manual_landmarks_used": False,
        "status": "running",
    }
    (args.output / "groups.json").write_text(json.dumps(assignments, indent=2) + "\n", encoding="utf-8")
    for index, names in enumerate(groups, 1):
        group = args.output / f"group_{index:02d}"
        views = group / "views"
        views.mkdir(parents=True)
        for name in names:
            shutil.copy2(images[name], views / images[name].name)
        print(f"Reconstructing {names}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(HERE / "run_experiment.py"),
                "--images",
                str(views),
                "--plan",
                str(args.plan),
                "--output",
                str(group / "reconstruction"),
                "--reconstruction-only",
            ],
            check=True,
        )
    if not args.reconstruction_only:
        subprocess.run(
            [
                sys.executable,
                str(HERE / "fragment_search.py"),
                "--groups",
                str(args.output),
                "--plan",
                str(args.plan),
                "--floor-masks",
                str(args.floor_masks),
                "--output",
                str(args.output / "fragment-search"),
                "--matcher",
                args.matcher,
            ],
            check=True,
        )
    assignments["status"] = "complete"
    (args.output / "groups.json").write_text(json.dumps(assignments, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
