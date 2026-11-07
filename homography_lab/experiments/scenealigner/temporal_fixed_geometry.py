"""Replay all temporal images with fixed lens/gravity but new reconstruction and plan search."""

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import run_experiment as runtime
import torch
from lens_geometry import remap_grid
from PIL import Image
from temporal_inference import match_frame, read, reconstruct_view


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous-inference", type=Path, required=True)
    p.add_argument("--cached-inputs", type=Path, required=True)
    p.add_argument("--cached-reconstruction", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-structure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    previous = read(args.previous_inference / "protocol.json")
    if previous["status"] != "complete":
        raise ValueError("Original temporal experiment must finish first")
    camera = previous["camera"]
    cases = [
        {
            key: case[key]
            for key in [
                "id",
                "kind",
                "actual_seconds",
                "foreground_fraction",
                "original_image",
                "image_sha256",
            ]
        }
        for case in previous["cases"]
        if case["kind"] != "control_cached_search"
    ]
    lens = read(args.cached_inputs / "lens.json")["cameras"][camera]
    gravity = np.load(args.cached_reconstruction / "reconstruction.npz")["est_gravity_cam"]
    protocol = {
        "status": "running",
        "camera": camera,
        "manual_landmarks_used": False,
        "reference_JSONs_opened": False,
        "mode": "fixed_lens_and_gravity_new_reconstruction_and_matching",
        "lens": lens,
        "gravity": gravity.tolist(),
        "seed": 42,
        "cases": cases,
    }
    start = perf_counter()

    def save():
        protocol["elapsed_sec"] = perf_counter() - start
        runtime.save_json(args.output / "protocol.json", protocol)

    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    mx, my, _ = remap_grid(lens)
    for case in cases:
        folder = args.output / case["id"]
        for category in ["views", "floor", "wall", "door"]:
            (folder / "corrected" / category).mkdir(parents=True)
        image = np.asarray(Image.open(case["original_image"]).convert("RGB"))
        path = folder / "corrected/views" / (camera + ".jpg")
        Image.fromarray(cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderValue=(255, 255, 255))).save(
            path, quality=95
        )
        case["corrected_image"] = str(path.resolve())
        for category in ["floor", "wall", "door"]:
            mask = cv2.imread(
                str(args.previous_inference / case["id"] / "original" / category / (camera + ".png")), 0
            )
            cv2.imwrite(
                str(folder / "corrected" / category / (camera + ".png")),
                cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderValue=0),
            )
    del mx, my
    save()
    torch.manual_seed(42)
    model = runtime.load_pi3()
    for case in cases:
        recon = reconstruct_view(model, case["corrected_image"], gravity)
        np.savez_compressed(args.output / case["id"] / "reconstruction.npz", **recon)
        print("Fixed geometry reconstruction", case["id"], flush=True)
    del model, recon
    runtime.clean_cuda()
    from fragment_search import prepare_bank
    from scenealigner.inference import load_model

    torch.manual_seed(42)
    model = load_model()
    plan = Image.open(args.plan).convert("RGB")
    bank = prepare_bank(
        model, Image.open(args.plan_structure / "plan-structure.png").convert("RGB"), runtime.HERE / "cache"
    )
    walls = cv2.imread(str(args.plan_structure / "plan-walls.png"), 0)
    doors = read(args.plan_structure / "semantics.json")["plan_doors"]
    for case in cases:
        try:
            folder = args.output / case["id"]
            recon = dict(np.load(folder / "reconstruction.npz"))
            case.update(
                match_frame(
                    model,
                    bank,
                    recon,
                    plan,
                    Path(case["corrected_image"]),
                    folder / "corrected/floor",
                    folder / "corrected",
                    lens,
                    camera,
                    folder,
                    walls,
                    doors,
                    42,
                )
            )
            print("Fixed geometry search", case["id"], case["status"], flush=True)
        except Exception as error:
            case.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
    protocol["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
