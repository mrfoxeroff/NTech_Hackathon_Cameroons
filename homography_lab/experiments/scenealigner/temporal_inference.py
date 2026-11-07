"""Independent-frame recalibration and cached-input controls. No reference labels."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import run_experiment as runtime
import torch
from lens_geometry import remap_grid
from PIL import Image
from rectify_inputs import lens_from_prediction


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def snapshot_rng():
    return torch.get_rng_state(), torch.cuda.get_rng_state_all()


def restore_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])


def reconstruct_view(model, image_path, gravity):
    from pi3.utils.geometry import homogenize_points, se3_inverse
    from scenealigner.preprocess import load_and_preprocess_images

    tensor, sizes = load_and_preprocess_images([Path(image_path)], target_size=518)
    tensor = tensor.cuda()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        prediction = model(tensor[None])
    relative = se3_inverse(prediction["camera_poses"][:, 0:1]) @ prediction["camera_poses"]
    xyz = torch.einsum("bnij,bnhwj->bnhwi", relative, homogenize_points(prediction["local_points"]))[
        0, ..., :3
    ]
    poses = relative[0].float().cpu().numpy()
    return {
        "points_cam0": xyz.float().cpu().numpy(),
        "conf": torch.sigmoid(prediction["conf"][0, ..., 0]).float().cpu().numpy(),
        "extrinsics": poses,
        "est_gravity_cam": poses[0, :3, :3] @ np.asarray(gravity),
        "image_sizes": sizes,
        "preprocessed_images": (tensor * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy(),
    }


def match_frame(
    model,
    bank,
    recon,
    plan,
    image_path,
    floor_root,
    semantic_root,
    lens,
    camera,
    folder,
    plan_walls,
    plan_doors,
    seed=42,
):
    from fragment_search import search_one
    from scenealigner.preprocess import resize_plan

    cv2.setRNGSeed(seed)
    np.random.seed(seed)
    _, (scale, ox, oy) = resize_plan(plan)
    global_transform = np.array([[scale, 0, ox], [0, scale, oy], [0, 0, 1.0]])
    output = folder / "search"
    output.mkdir(parents=True)
    structure = {
        "semantics": semantic_root,
        "images": [str(image_path)],
        "floor_masks": floor_root,
        "plan_walls": plan_walls,
        "plan_doors": plan_doors,
    }
    result, pp, _ = search_one(model, bank, recon, plan, output, "topk", structure)
    record = {
        "search_status": result["status"],
        "search_ambiguous": result["ambiguous"],
        "search_score_gap": result.get("relative_score_gap"),
    }
    if result["candidates"]:
        best = result["candidates"][0]
        matrix = (global_transform @ np.array(best["density_to_original"]))[:2]
        prediction_root = folder / "predictions"
        prediction_root.mkdir()
        runtime.recover_homographies(
            recon,
            {"proj_params": pp, "M": matrix},
            [Path(image_path)],
            floor_root,
            plan.size,
            (scale, ox, oy),
            prediction_root,
            lens_models={camera: lens},
        )
        prediction_path = prediction_root / (camera + "-calibration.json")
        if prediction_path.is_file():
            prediction = read(prediction_path)
            prediction.update(
                method="SceneAligner_grouped_fragment_search",
                structural_features_used=True,
                search_ambiguous=result["ambiguous"],
                search_score=best["score"],
            )
            runtime.save_json(prediction_path, prediction)
            record["prediction"] = str(prediction_path.resolve())
    record["status"] = "complete" if record.get("prediction") else "no_prediction"
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-structure", type=Path, required=True)
    p.add_argument("--control-image", type=Path, required=True)
    p.add_argument("--cached-reconstruction", type=Path, required=True)
    p.add_argument("--cached-inputs", type=Path, required=True)
    p.add_argument("--cached-semantics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samples = read(args.samples)
    camera = samples["camera"]
    cases = [
        {
            "id": "control_median_fresh",
            "kind": "control_median",
            "image": str(args.control_image.resolve()),
            "actual_seconds": None,
            "foreground_fraction": None,
        }
    ] + samples["cases"]
    manifest = {
        "status": "running",
        "camera": camera,
        "manual_landmarks_used": False,
        "reference_JSONs_opened": False,
        "samples": str(args.samples.resolve()),
        "cases": cases,
        "seed_protocol": "42 per independently initialized model call; post-initialization RNG restored when reusing models",
        "search_seed": 42,
        "cached_search_control_seeds": [42, 43, 44],
        "stages": [],
    }
    start = perf_counter()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    def save():
        manifest["elapsed_sec"] = perf_counter() - start
        runtime.save_json(args.output / "protocol.json", manifest)

    def stage(name):
        manifest["stages"].append({"name": name, "elapsed_sec": perf_counter() - start})
        print(name, flush=True)
        save()

    def active():
        return [case for case in cases if not case.get("error")]

    def failed(case, error, stage_name):
        case.update(error=f"{type(error).__name__}: {error}", failed_stage=stage_name, status="failed")
        print("FAILED", case["id"], case["error"], flush=True)
        runtime.clean_cuda()
        save()

    for case in cases:
        folder = args.output / case["id"]
        for subdir in [
            "original/views",
            "original/floor",
            "original/wall",
            "original/door",
            "corrected/views",
            "corrected/floor",
            "corrected/wall",
            "corrected/door",
        ]:
            (folder / subdir).mkdir(parents=True)
        image = folder / "original/views" / (camera + ".jpg")
        shutil.copy2(case["image"], image)
        case["original_image"] = str(image.resolve())
        case["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    save()
    stage("Neural floor, wall and door masks for every selected image")
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    name = "nvidia/segformer-b0-finetuned-ade-512-512"
    cache = runtime.LAB / "neural_weights/huggingface"
    processor = AutoImageProcessor.from_pretrained(
        name, cache_dir=cache, local_files_only=True, use_fast=False
    )
    segmenter = (
        AutoModelForSemanticSegmentation.from_pretrained(
            name, cache_dir=cache, local_files_only=True, use_safetensors=True
        )
        .eval()
        .cuda()
    )
    labels = {label.lower().strip(): int(i) for i, label in segmenter.config.id2label.items()}
    categories = {
        "floor": [labels["floor"]],
        "wall": [labels[label] for label in ["wall", "column"] if label in labels],
        "door": [labels["door"]],
    }
    thresholds = {"floor": 0.6, "wall": 0.4, "door": 0.35}
    for case in active():
        try:
            image = Image.open(case["original_image"]).convert("RGB")
            with torch.inference_mode():
                probs = segmenter(**processor(images=image, return_tensors="pt").to("cuda")).logits.softmax(1)
                fractions = {}
                for category, indexes in categories.items():
                    probability = (
                        torch.nn.functional.interpolate(
                            probs[:, indexes].sum(1, keepdim=True),
                            size=(image.height, image.width),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                        .cpu()
                        .numpy()
                    )
                    mask = (probability >= thresholds[category]).astype(np.uint8) * 255
                    cv2.imwrite(
                        str(args.output / case["id"] / "original" / category / (camera + ".png")), mask
                    )
                    fractions[category] = float((mask > 0).mean())
            case["semantic_fractions"] = fractions
            print("Masks", case["id"], fractions, flush=True)
        except Exception as error:
            failed(case, error, "segmentation")
    del segmenter, processor, probs
    runtime.clean_cuda()
    stage("Independent lens estimation and consistent image/mask correction")
    from geocalib import GeoCalib

    torch.manual_seed(42)
    geo = GeoCalib(weights="distorted").eval().cuda()
    initialized_rng = snapshot_rng()
    for case in active():
        try:
            restore_rng(initialized_rng)
            image = np.asarray(Image.open(case["original_image"]).convert("RGB"))
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().cuda() / 255
            with torch.inference_mode():
                prediction = geo.calibrate(tensor, camera_model="simple_divisional")
            lens = lens_from_prediction(prediction, image.shape[1], image.shape[0], "simple_divisional")
            mx, my, valid = remap_grid(lens)
            corrected = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            folder = args.output / case["id"]
            path = folder / "corrected/views" / (camera + ".jpg")
            Image.fromarray(corrected).save(path, quality=95)
            for category in categories:
                mask = cv2.imread(str(folder / "original" / category / (camera + ".png")), 0)
                cv2.imwrite(
                    str(folder / "corrected" / category / (camera + ".png")),
                    cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderValue=0),
                )
            lens["valid_rectified_fraction"] = float(valid.mean())
            case["lens"] = lens
            case["corrected_image"] = str(path.resolve())
            runtime.save_json(
                folder / "lens.json", {"cameras": {camera: lens}, "manual_landmarks_used": False}
            )
            print("Lens", case["id"], lens["K"][0][0], lens["k1"], flush=True)
            save()
        except Exception as error:
            failed(case, error, "lens")
    # Same-image controls of the stochastic NMF stage; not used to choose a lens.
    lens_seed_checks = []
    control = np.asarray(Image.open(args.control_image).convert("RGB"))
    for seed in [43, 44]:
        torch.manual_seed(seed)
        tensor = torch.from_numpy(control.copy()).permute(2, 0, 1).float().cuda() / 255
        with torch.inference_mode():
            prediction = geo.calibrate(tensor, camera_model="simple_divisional")
        lens_seed_checks.append(
            {
                "seed": seed,
                "lens": lens_from_prediction(
                    prediction, control.shape[1], control.shape[0], "simple_divisional"
                ),
            }
        )
    runtime.save_json(args.output / "lens-seed-checks.json", lens_seed_checks)
    del geo, prediction, tensor
    runtime.clean_cuda()
    stage("Gravity estimation on corrected images")
    from scenealigner.preprocess import estimate_gravity

    torch.manual_seed(42)
    geo = GeoCalib(weights="pinhole").eval().cuda()
    initialized_rng = snapshot_rng()
    for case in active():
        try:
            restore_rng(initialized_rng)
            case["pinhole_gravity"] = estimate_gravity(Path(case["corrected_image"]), geo).tolist()
        except Exception as error:
            failed(case, error, "gravity")
    del geo
    runtime.clean_cuda()
    stage("Pi3 reconstruction, one view at a time with unchanged model weights")
    torch.manual_seed(42)
    model = runtime.load_pi3()
    for case in active():
        try:
            recon = reconstruct_view(model, case["corrected_image"], case["pinhole_gravity"])
            np.savez_compressed(args.output / case["id"] / "reconstruction.npz", **recon)
            del recon
            print("Reconstruction", case["id"], flush=True)
        except Exception as error:
            failed(case, error, "reconstruction")
    del model
    runtime.clean_cuda()
    stage("Plan-fragment search: per-frame recalibration and exact cached-input seed controls")
    from fragment_search import prepare_bank
    from scenealigner.inference import load_model

    torch.manual_seed(42)
    model = load_model()
    plan = Image.open(args.plan).convert("RGB")
    matching_plan = Image.open(args.plan_structure / "plan-structure.png").convert("RGB")
    bank = prepare_bank(model, matching_plan, runtime.HERE / "cache")
    plan_walls = cv2.imread(str(args.plan_structure / "plan-walls.png"), 0)
    plan_doors = read(args.plan_structure / "semantics.json")["plan_doors"]

    def match(case, recon, image_path, floor_root, semantic_root, lens, seed):
        case.update(
            match_frame(
                model,
                bank,
                recon,
                plan,
                image_path,
                floor_root,
                semantic_root,
                lens,
                camera,
                args.output / case["id"],
                plan_walls,
                plan_doors,
                seed,
            )
        )
        print("Completed search", case["id"], case["status"], "gap", case.get("search_score_gap"), flush=True)
        save()

    for case in active():
        try:
            folder = args.output / case["id"]
            recon = dict(np.load(folder / "reconstruction.npz"))
            match(
                case,
                recon,
                Path(case["corrected_image"]),
                folder / "corrected/floor",
                folder / "corrected",
                case["lens"],
                42,
            )
        except Exception as error:
            failed(case, error, "search")
    cached_recon = dict(np.load(args.cached_reconstruction / "reconstruction.npz"))
    cached_lens = read(args.cached_inputs / "lens.json")["cameras"][camera]
    for seed in [42, 43, 44]:
        case = {
            "id": f"control_cached_seed_{seed}",
            "kind": "control_cached_search",
            "seed": seed,
            "actual_seconds": None,
            "foreground_fraction": None,
        }
        cases.append(case)
        try:
            match(
                case,
                cached_recon,
                args.cached_inputs / "views" / (camera + ".jpg"),
                args.cached_inputs / "floor",
                args.cached_semantics,
                cached_lens,
                seed,
            )
        except Exception as error:
            failed(case, error, "cached_search")
    manifest["status"] = "complete"
    stage("Inference complete; reference-based evaluation must be run separately")


if __name__ == "__main__":
    main()
