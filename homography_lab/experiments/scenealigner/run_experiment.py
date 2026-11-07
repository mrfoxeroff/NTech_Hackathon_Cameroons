"""Isolated SceneAligner experiment. Reference JSONs are not opened here."""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
for name in ("SceneAligner", "Pi3", "GeoCalib", "dinov3"):
    sys.path.insert(0, str(LAB / "vendor" / name))
os.environ["HF_HUB_CACHE"] = str(HERE / "cache/hub")
os.environ["HF_HOME"] = str(HERE / "cache/hf")
os.environ["TORCH_HOME"] = str(HERE / "cache/torch")
os.environ["MPLCONFIGDIR"] = str(HERE / "cache/matplotlib")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def clean_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def load_pi3():
    from huggingface_hub import hf_hub_download
    from pi3.models.pi3 import Pi3

    path = hf_hub_download(
        "y-u-a-n-l-i/MegaDepth-X",
        "ckpts/pi3-FT_weight.pt",
        repo_type="dataset",
        revision="1c7dd2819846a4a51c10defae4f8c3f907af2b6e",
        local_files_only=True,
    )
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    # DINOv2 builds a scalar drop-path schedule with linspace(...).item().
    # Keep that tiny schedule concrete while allocating large parameters on meta.
    original_linspace = torch.linspace

    def scalar_schedule(*args, **kwargs):
        return original_linspace(*args, **(kwargs | {"device": "cpu"}))

    try:
        torch.linspace = scalar_schedule
        with torch.device("meta"):
            model = Pi3()
    finally:
        torch.linspace = original_linspace
    model.load_state_dict(state.get("model", state), strict=True, assign=True)
    model = model.eval().to(device="cuda", dtype=torch.float16)
    # The official forward explicitly disables autocast in these three heads.
    for name in ("point_head", "conf_head", "camera_head"):
        getattr(model, name).float()
    del state
    clean_cuda()
    return model


def recover_homographies(
    recon, alignment, image_paths, floor_masks, plan_size, plan_resize, output, lens_models=None
):
    """Fit floor homographies from predicted 3D points, not GitHub landmarks."""
    from density_geometry import get_density_coordinates

    pp, affine = alignment["proj_params"], alignment["M"]
    scale, ox, oy = plan_resize
    cameras = {}
    for i, image_path in enumerate(image_paths):
        name = image_path.stem
        with Image.open(image_path) as im:
            original_size = im.size
        width, height = map(int, recon["image_sizes"][i])
        full_h, full_w = recon["conf"][i].shape
        px, py = (full_w - width) // 2, (full_h - height) // 2
        mask_path = floor_masks / f"{name}.png"
        if not mask_path.is_file():
            cameras[name] = {"status": "no_floor_mask"}
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or list(mask.shape[::-1]) != list(original_size):
            raise ValueError("Floor mask resolution mismatch")
        selected = np.zeros((full_h, full_w), np.uint8)
        selected[py : py + height, px : px + width] = cv2.resize(
            mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
        good = (selected > 128) & np.isfinite(recon["conf"][i])
        if not good.any():
            cameras[name] = {"status": "no_floor_points"}
            continue
        cutoff = np.quantile(recon["conf"][i][good], 0.5)
        good &= recon["conf"][i] >= cutoff
        yy, xx = np.mgrid[:full_h, :full_w]
        good &= (xx % 8 == 0) & (yy % 8 == 0)
        xyz = recon["points_cam0"][i][good] @ pp["R_rect"].T
        density = get_density_coordinates(xyz, pp)
        plan = cv2.transform(density[:, None].astype(np.float32), affine)[:, 0]
        plan = (plan - [ox, oy]) / scale
        photo = np.column_stack(
            ((xx[good] - px) * original_size[0] / width, (yy[good] - py) * original_size[1] / height)
        )
        finite = np.isfinite(plan).all(1) & np.isfinite(photo).all(1)
        photo, plan = photo[finite], plan[finite]
        if len(photo) < 12:
            cameras[name] = {"status": "insufficient_floor_points", "points": len(photo)}
            continue
        # Prediction holdout, distinct from the separate GitHub-landmark evaluation.
        check = np.arange(len(photo)) % 4 == 0
        h, inliers = cv2.findHomography(photo[~check], plan[~check], cv2.RANSAC, 5.0)
        if h is None or np.linalg.matrix_rank(h) != 3:
            cameras[name] = {"status": "degenerate_predicted_floor"}
            continue
        pred = cv2.perspectiveTransform(photo[check, None].astype(np.float64), h)[:, 0]
        errors = np.linalg.norm(pred - plan[check], axis=1)
        quality = {
            "predicted_floor_points": len(photo),
            "fit_inlier_ratio": float(inliers.mean()),
            "predicted_floor_holdout_median_px": float(np.median(errors)),
            "predicted_floor_holdout_p95_px": float(np.quantile(errors, 0.95)),
            "independent_physical_accuracy_verified": False,
        }
        calibration = {
            "camera_id": name,
            "camera_size": list(original_size),
            "plan_size": list(plan_size),
            "homography": (h / h[2, 2]).tolist(),
            "status": "automatic_candidate",
            "method": "SceneAligner_Pi3_predicted_floor",
            "lens_correction_applied": False,
            "manual_landmarks_used": False,
            "quality": quality,
        }
        if lens_models is not None:
            if name not in lens_models or lens_models[name]["rectified_size"] != list(original_size):
                raise ValueError("Missing or mismatched camera lens transform")
            calibration.update(
                lens_correction_applied=True,
                homography_input_space="rectified_pixels",
                lens=lens_models[name],
                camera_size=lens_models[name]["size"],
            )
        save_json(output / f"{name}-calibration.json", calibration)
        cameras[name] = {"status": "automatic_candidate", "quality": quality}
    return cameras


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--floor-masks", type=Path)
    p.add_argument("--resolution", type=int, default=518)
    p.add_argument(
        "--reconstruction-only",
        action="store_true",
        help="Save an independent group reconstruction for fragment search",
    )
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new output directory")
    if args.resolution < 224 or args.resolution % 14:
        p.error("Resolution must be a multiple of 14, at least 224")
    images = sorted(x for x in args.images.iterdir() if x.suffix.lower() in (".jpg", ".png", ".jpeg"))
    if not images:
        p.error("No input images")
    args.output.mkdir(parents=True)
    manifest = {
        "status": "running",
        "manual_landmarks_used": False,
        "resolution": args.resolution,
        "images": [str(x.resolve()) for x in images],
        "plan": str(args.plan.resolve()),
        "steps": [],
    }
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    np.random.seed(42)
    torch.manual_seed(42)
    start = perf_counter()

    def stage(name):
        manifest["steps"].append({"name": name, "elapsed_sec": perf_counter() - start})
        save_json(args.output / "manifest.json", manifest)
        print(name, flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in the isolated environment")
        manifest["gpu"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
        from geocalib import GeoCalib
        from pi3.utils.geometry import homogenize_points, se3_inverse
        from scenealigner.inference import align_to_floorplan, get_floorplan_aligned_scene, load_model
        from scenealigner.preprocess import (
            compute_robust_gravity_medoid,
            estimate_gravity,
            load_and_preprocess_images,
            load_and_resize_plan,
        )
        from scenealigner.visualization import draw_alignment, draw_scene_cameras

        stage("Estimating gravity with GeoCalib")
        geo = GeoCalib(weights="pinhole").eval().to("cuda")
        gravities = [estimate_gravity(path, geo) for path in images]
        del geo
        clean_cuda()
        stage("Reconstructing with Pi3 (mixed precision)")
        model = load_pi3()
        tensor, image_sizes = load_and_preprocess_images(images, target_size=args.resolution)
        tensor = tensor.to("cuda")
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            prediction = model(tensor[None])
        relative = se3_inverse(prediction["camera_poses"][:, 0:1]) @ prediction["camera_poses"]
        xyz = torch.einsum("bnij,bnhwj->bnhwi", relative, homogenize_points(prediction["local_points"]))[
            0, ..., :3
        ]
        poses = relative[0].float().cpu().numpy()
        g0 = [poses[i, :3, :3] @ g for i, g in enumerate(gravities)]
        recon = {
            "points_cam0": xyz.float().cpu().numpy(),
            "conf": torch.sigmoid(prediction["conf"][0, ..., 0]).float().cpu().numpy(),
            "extrinsics": poses,
            "est_gravity_cam": compute_robust_gravity_medoid(np.stack(g0)),
            "image_sizes": image_sizes,
            "preprocessed_images": (tensor * 255)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy(),
        }
        np.savez_compressed(args.output / "reconstruction.npz", **recon)
        del model, tensor, prediction, relative, xyz
        clean_cuda()
        if args.reconstruction_only:
            manifest["status"] = "complete"
            stage("Completed group reconstruction; fragment search is separate")
            return
        stage("Aligning density map with the plan")
        model = load_model()
        plan, resize = load_and_resize_plan(args.plan)
        alignment = align_to_floorplan(recon, plan, model)
        alignment["density_img"].save(args.output / "density.png")
        overlay = draw_alignment(
            np.array(alignment["density_img"]), np.array(plan), alignment["M"], alignment["src_px"]
        )
        Image.fromarray(overlay).save(args.output / "alignment.png")
        scene = get_floorplan_aligned_scene(recon, alignment)
        np.savez_compressed(args.output / "scene.npz", **scene)
        cameras = np.array(plan).copy()
        draw_scene_cameras(cameras, scene["cam_to_world"], (0, 150, 230))
        Image.fromarray(cameras).save(args.output / "cameras.png")
        save_json(
            args.output / "alignment.json",
            {
                "M": alignment["M"].tolist(),
                "source_points": alignment["src_px"].tolist(),
                "plan_points": alignment["dst_px"].tolist(),
                "plan_resize": list(resize),
            },
        )
        if args.floor_masks:
            with Image.open(args.plan) as original:
                manifest["cameras"] = recover_homographies(
                    recon, alignment, images, args.floor_masks, original.size, resize, args.output
                )
        manifest["status"] = "complete"
        stage("Completed automatic inference; independent evaluation is separate")
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["elapsed_sec"] = perf_counter() - start
        manifest["peak_cuda_allocated_bytes"] = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        save_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
