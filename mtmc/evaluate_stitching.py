"""Reproducible evaluation on cached observations; labels never enter association."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from time import perf_counter

from associate import DEFAULTS, Observation, associate, export_tracks


def load_case(directory, case):
    reference = {(cam, ident): label for cam, ident, label, *times in case["labels"]}
    ranges = {(cam, ident): times or case["window_ms"] for cam, ident, _, *times in case["labels"]}
    raw = [json.loads(line) for line in (directory / case["observations"]).read_text().splitlines()]
    rows = [row for row in raw if (key := (row["camera_id"], row["local_id"])) in ranges
            and ranges[key][0] <= row["timestamp_ms"] < ranges[key][1]]
    return [Observation.parse(row) for row in rows], reference, len(raw) - len(rows)


def metrics(groups, reference):
    labels = [Counter(reference[o.camera, o.local_id] for o in group) for group in groups]
    counts = Counter(label for group in labels for label in group)
    return {"total_tracks": len(groups), "mixed_tracks": sum(len(group) > 1 for group in labels),
            "split_reference_tracks": sum(n > 1 for n in counts.values()),
            "extra_fragments": sum(n - 1 for n in counts.values()),
            "groups": [dict(group) for group in labels]}


def signature(groups):
    return [sorted((o.camera, str(o.local_id), o.frame) for o in group) for group in groups]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new one")
    baseline = json.loads(Path(__file__).with_name("camera_graph.pilot.json").read_text())
    improved = json.loads(args.config.read_text())
    manifest = json.loads(args.cases.read_text())
    variants = {"baseline": baseline,
                "local_only": baseline | {"stitch_local_fragments": True},
                "context_only": baseline | {"contextual_appearance": True},
                "without_joint": improved | {"joint_overlap_matching": False},
                "improved": improved}
    report = {"scope": "Reviewed observation ranges only; development set, not an independent test set.",
              "settings": variants, "cases": {}}
    args.output.mkdir(parents=True)
    for case in manifest["cases"]:
        observations, reference, excluded = load_case(args.cases.parent, case)
        result = {"reference_tracks": len(set(reference.values())), "observations": len(observations),
                  "excluded_observations": excluded, "variants": {}}
        for name, config in variants.items():
            start = perf_counter()
            groups = associate(observations, config)
            elapsed = (perf_counter() - start) * 1000
            assert sum(map(len, groups)) == len(observations), "Observation lost or duplicated"
            shuffled = observations.copy()
            random.Random(42).shuffle(shuffled)
            assert signature(groups) == signature(associate(shuffled, config)), "Input-order dependent IDs"
            result["variants"][name] = metrics(groups, reference) | {"elapsed_ms": elapsed}
            if name in ("baseline", "improved"):
                export_tracks(groups, args.output / case["name"] / name,
                              (DEFAULTS | config)["visibility_gap_ms"])
        report["cases"][case["name"]] = result
        print(case["name"], {name: (value["total_tracks"], value["mixed_tracks"])
                             for name, value in result["variants"].items()}, flush=True)
    report["totals"] = {name: {key: sum(case["variants"][name][key] for case in report["cases"].values())
                               for key in ("total_tracks", "mixed_tracks", "split_reference_tracks", "extra_fragments")}
                        for name in variants}
    report["reference_tracks"] = sum(case["reference_tracks"] for case in report["cases"].values())
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
