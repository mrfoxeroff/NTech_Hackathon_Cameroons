"""Repeat a fixed-group 2x2 study; reference JSONs are passed only to evaluation/reporting."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def validate_original_groups(groups, inputs):
    from grouped_experiment import validate_groups

    partition = json.loads((groups / "groups.json").read_text(encoding="utf-8"))["groups"]
    available = {path.stem: path for path in (inputs / "views").glob("*.jpg")}
    validate_groups(partition, available)
    for index, names in enumerate(partition, 1):
        manifest = json.loads(
            (groups / f"group_{index:02d}/reconstruction/manifest.json").read_text(encoding="utf-8")
        )
        if manifest["status"] != "complete" or sorted(
            Path(path).stem for path in manifest["images"]
        ) != sorted(names):
            raise ValueError("Original group reconstruction is incomplete or inconsistent")
        for image in manifest["images"]:
            path = Path(image)
            if (
                hashlib.sha256(path.read_bytes()).digest()
                != hashlib.sha256(available[path.stem].read_bytes()).digest()
            ):
                raise ValueError("Original reconstructions do not belong to these input images")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument(
        "--groups", type=Path, required=True, help="Existing original-image grouped reconstructions"
    )
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    validate_original_groups(args.groups, args.inputs)
    args.output.mkdir(parents=True, exist_ok=False)

    def run(script, *arguments):
        print("Stage:", script, flush=True)
        subprocess.run([sys.executable, str(HERE / script), *map(str, arguments)], check=True)

    radial, corrected = args.output / "inputs-radial", args.output / "inputs-corrected"
    semantic, groups = args.output / "semantics", args.output / "corrected-groups"
    for model, output in [("simple_radial", radial), ("simple_divisional", corrected)]:
        run("rectify_inputs.py", "--inputs", args.inputs, "--output", output, "--camera-model", model)
    run("semantic_inputs.py", "--inputs", args.inputs, "--rectified", corrected, "--output", semantic)
    run(
        "check_lens.py",
        "--inputs",
        args.inputs,
        "--semantics",
        semantic / "original",
        "--radial",
        radial / "lens.json",
        "--divisional",
        corrected / "lens.json",
        "--output",
        args.output / "straightness",
    )
    run(
        "grouped_experiment.py",
        "--images",
        corrected / "views",
        "--overlap",
        args.groups / "groups.json",
        "--plan",
        args.inputs / "plan.jpg",
        "--floor-masks",
        corrected / "floor",
        "--output",
        groups,
        "--reconstruction-only",
    )
    variants = []
    for name, label, use_lens, use_structure in [
        ("baseline", "Предыдущий вариант", False, False),
        ("lens", "Только коррекция", True, False),
        ("structure", "Только стены/двери", False, True),
        ("combined", "Коррекция + стены/двери", True, True),
    ]:
        output = args.output / name
        arguments = [
            "--groups",
            groups if use_lens else args.groups,
            "--plan",
            args.inputs / "plan.jpg",
            "--floor-masks",
            (corrected if use_lens else args.inputs) / "floor",
            "--output",
            output,
            "--matcher",
            "topk",
        ]
        if use_lens:
            arguments += ["--lens-json", corrected / "lens.json"]
        if use_structure:
            arguments += ["--structural-inputs", semantic / ("rectified" if use_lens else "original")]
        run("fragment_search.py", *arguments)
        variants.append(
            {
                "id": name,
                "label": label,
                "predictions": str((output / "predictions").resolve()),
                "evaluation": str((args.output / (name + "-evaluation.json")).resolve()),
            }
        )
    # No evaluation feedback can affect any inference run above.
    for variant in variants:
        run(
            "evaluate.py",
            "--predictions",
            variant["predictions"],
            "--reference",
            args.reference,
            "--output",
            variant["evaluation"],
        )
    config = {
        "original_inputs": str(args.inputs.resolve()),
        "corrected_inputs": str(corrected.resolve()),
        "semantics": str(semantic.resolve()),
        "corrected_semantic_mode": "rectified",
        "reference": str(args.reference.resolve()),
        "straightness": str((args.output / "straightness/straightness.json").resolve()),
        "variants": variants,
        "manual_landmarks_used_for_inference": False,
    }
    path = args.output / "experiment.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run("lens_structure_report.py", "--config", path)


if __name__ == "__main__":
    main()
