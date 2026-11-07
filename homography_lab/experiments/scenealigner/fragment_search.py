"""Multi-scale neural plan-fragment search. No reference JSON is read here."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import run_experiment as runtime
import torch
import torch.nn.functional as F
from PIL import Image


def windows(width, height):
    """Generic overlapping windows; no office coordinates or room IDs."""
    found = {(0, 0, width, height)}
    for fraction in [0.65, 0.45, 0.30, 0.20]:
        for aspect in [1.0, 0.6, 1.6] if fraction in [0.65, 0.45] else [1.0]:
            side = max(width, height) * fraction
            w = min(width, round(side * np.sqrt(aspect)))
            h = min(height, round(side / np.sqrt(aspect)))
            xs = (
                np.linspace(0, width - w, max(1, int(np.ceil((width - w) / (0.7 * w))) + 1))
                .round()
                .astype(int)
            )
            ys = (
                np.linspace(0, height - h, max(1, int(np.ceil((height - h) / (0.7 * h))) + 1))
                .round()
                .astype(int)
            )
            found.update((int(x), int(y), w, h) for y in ys for x in xs)
    return sorted(found, key=lambda box: (-box[2] * box[3], box))


def rotation_matrix(turns, size=512):
    return [
        np.eye(3),
        np.array([[0, -1, size - 1], [1, 0, 0], [0, 0, 1]]),
        np.array([[-1, 0, size - 1], [0, -1, size - 1], [0, 0, 1]]),
        np.array([[0, 1, 0], [-1, 0, size - 1], [0, 0, 1]]),
    ][turns]


def compose_to_original(affine, crop_transform, rotation):
    return np.linalg.inv(crop_transform) @ np.vstack((affine, [0, 0, 1])) @ rotation


def fit_candidate(source, target, similarities):
    """Reject collapsed/collinear matches instead of averaging multimodal positions."""
    if len(source) < 8:
        return None
    affine, mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=8.0, maxIters=3000, confidence=0.995
    )
    if affine is None or not np.isfinite(affine).all():
        return None
    good = mask.ravel().astype(bool)
    residuals = np.linalg.norm(
        cv2.transform(source[:, None].astype(np.float32), affine)[:, 0] - target, axis=1
    )
    unique_good = np.zeros(len(source), bool)
    seen_source, seen_target = set(), set()
    for index in np.argsort(residuals):
        if not good[index]:
            continue
        akey, bkey = tuple(np.round(source[index], 2)), tuple(np.round(target[index], 2))
        if akey not in seen_source and bkey not in seen_target:
            unique_good[index] = True
            seen_source.add(akey)
            seen_target.add(bkey)
    good = unique_good
    ratio = float(good.sum() / len(np.unique(source, axis=0)))
    if good.sum() < 6 or ratio < 0.3:
        return None
    a, b = source[good].astype(np.float32), target[good].astype(np.float32)
    coverage = [float(cv2.contourArea(cv2.convexHull(p)) / 512**2) for p in [a, b]]
    scale = float(np.sqrt(abs(np.linalg.det(affine[:, :2]))))
    if min(coverage) < 0.015 or not 0.08 <= scale <= 5:
        return None
    residual = np.linalg.norm(cv2.transform(a[:, None], affine)[:, 0] - b, axis=1)
    score = float(np.sqrt(good.sum()) * np.sqrt(coverage[0]) * ratio * np.mean(similarities[good]))
    return {
        "affine": affine,
        "inliers": good,
        "score": score,
        "count": int(good.sum()),
        "ratio": ratio,
        "coverage": coverage,
        "scale": scale,
        "median_residual": float(np.median(residual)),
    }


@torch.inference_mode()
def descriptors(model, image):
    from scenealigner.inference import norm_transform

    tokens = model.forward_features(norm_transform(image)[None].cuda())["x_norm_patchtokens"][0]
    side = int(np.sqrt(len(tokens)))
    yy, xx = np.mgrid[:side, :side]
    centers = (np.column_stack((xx.ravel(), yy.ravel())) + 0.5) * 512 / side
    dark = cv2.resize(
        (np.asarray(image.convert("L")) < 220).astype(np.float32), (side, side), interpolation=cv2.INTER_AREA
    ).ravel()
    keep = dark >= 0.025
    return F.normalize(tokens[keep].float(), dim=-1), centers[keep].astype(np.float32)


def prepare_bank(model, plan, cache):
    from scenealigner.preprocess import resize_plan

    identity = hashlib.sha256(plan.tobytes() + str(plan.size).encode() + b"fragment-bank-v1").hexdigest()
    path = cache / f"fragment-bank-{identity[:16]}.pt"
    if path.is_file():
        bank = torch.load(path, map_location="cpu", weights_only=True)
        if bank["identity"] != identity:
            raise ValueError("Plan cache identity mismatch")
        return bank["items"]
    items = []
    for index, (x, y, w, h) in enumerate(windows(*plan.size)):
        crop = plan.crop((x, y, x + w, y + h))
        if (np.asarray(crop.convert("L")) < 180).mean() < 0.03:
            continue
        image, (scale, ox, oy) = resize_plan(crop)
        feature, centers = descriptors(model, image)
        if len(centers) < 12:
            continue
        transform = [[scale, 0.0, ox - scale * x], [0.0, scale, oy - scale * y], [0.0, 0.0, 1.0]]
        items.append(
            {
                "box": [x, y, w, h],
                "transform": transform,
                "features": feature.cpu().half(),
                "centers": torch.tensor(centers),
            }
        )
        if len(items) % 20 == 0:
            print(f"Plan fragments encoded: {len(items)}", flush=True)
    torch.save({"identity": identity, "items": items}, path)
    return items


def distinct_candidates(candidates, maximum=10):
    chosen = []
    center = np.array([256.0, 256.0, 1.0])
    for candidate in sorted(candidates, key=lambda item: -item["score"]):
        m = np.asarray(candidate["density_to_original"])
        position = (m @ center)[:2]
        scale = np.sqrt(abs(np.linalg.det(m[:2, :2])))
        angle = np.arctan2(m[1, 0], m[0, 0])
        duplicate = False
        for other in chosen:
            previous = np.asarray(other["density_to_original"])
            previous_scale = np.sqrt(abs(np.linalg.det(previous[:2, :2])))
            previous_angle = np.arctan2(previous[1, 0], previous[0, 0])
            delta_angle = abs(np.angle(np.exp(1j * (angle - previous_angle))))
            duplicate |= (
                np.linalg.norm(position - (previous @ center)[:2]) < 30
                and 0.7 < scale / previous_scale < 1.4
                and delta_angle < np.deg2rad(20)
            )
        if duplicate:
            continue
        chosen.append(candidate)
        if len(chosen) == maximum:
            break
    return chosen


def search_one(model, bank, recon, plan, output, matcher="mutual", structural=None):
    from scenealigner.inference import reconstruct_projection_geometry, render_density_map

    pp, points, _ = reconstruct_projection_geometry(
        recon["points_cam0"], recon["conf"], recon["est_gravity_cam"]
    )
    density = render_density_map(points, pp)
    wall_image, door_points, floors = None, None, None
    if structural:
        from structural_features import structural_density
        from structural_matching import floor_probes

        density, wall_image, door_points, diagnostic = structural_density(
            recon, pp, structural["images"], structural["semantics"]
        )
        floors = floor_probes(recon, pp, structural["images"], structural["floor_masks"])
        runtime.save_json(output / "structural-features.json", diagnostic)
        Image.fromarray(wall_image).save(output / "wall-density.png")
    density.save(output / "density.png")
    candidates = []
    for turns in range(4):
        rotated = Image.fromarray(np.rot90(np.asarray(density), -turns).copy())
        features, centers = descriptors(model, rotated)
        rotation = rotation_matrix(turns)
        if len(centers) < 8:
            continue
        for item in bank:
            target_features = F.normalize(item["features"].float().cuda(), dim=-1)
            sims = features @ target_features.T
            if matcher == "topk":
                values, indexes = sims.topk(3, dim=1)
                reverse = sims.topk(min(3, len(features)), dim=0).indices
                query = torch.arange(len(features), device="cuda")[:, None]
                mutual = (reverse[:, indexes] == query[None]).any(0)
                keep = mutual & (values >= 0.35)
                queries = query.expand_as(indexes)[keep]
                destinations = indexes[keep]
                source = centers[queries.cpu().numpy()]
                target = item["centers"][destinations.cpu()].numpy()
                confidence = values[keep].cpu().numpy()
            else:
                values, indexes = sims.topk(2, dim=1)
                backward = sims.argmax(0)
                query = torch.arange(len(features), device="cuda")
                mutual = backward[indexes[:, 0]] == query
                keep = mutual & (values[:, 0] >= 0.35) & (values[:, 0] - values[:, 1] >= 0.005)
                source = centers[keep.cpu().numpy()]
                target = item["centers"][indexes[keep, 0].cpu()].numpy()
                confidence = values[keep, 0].cpu().numpy()
            fit = fit_candidate(source, target, confidence)
            if fit is None:
                continue
            matrix = compose_to_original(fit["affine"], np.asarray(item["transform"]), rotation)
            probes = np.column_stack((centers, np.ones(len(centers)))) @ np.linalg.inv(rotation).T
            projected = probes @ matrix.T
            inside = (
                (projected[:, 0] >= 0)
                & (projected[:, 0] < plan.width)
                & (projected[:, 1] >= 0)
                & (projected[:, 1] < plan.height)
            )
            if inside.mean() < 0.8:
                continue
            good = fit["inliers"]
            src_original = (np.column_stack((source[good], np.ones(good.sum()))) @ np.linalg.inv(rotation).T)[
                :, :2
            ]
            dst_original = (
                np.column_stack((target[good], np.ones(good.sum()))) @ np.linalg.inv(item["transform"]).T
            )[:, :2]
            candidates.append(
                {
                    "score": fit["score"],
                    "fragment_box": item["box"],
                    "density_rotation_quarters": turns,
                    "density_to_original": matrix.tolist(),
                    "inliers": fit["count"],
                    "inlier_ratio": fit["ratio"],
                    "coverage": fit["coverage"],
                    "median_residual_fragment_px": fit["median_residual"],
                    "source_points": src_original.tolist(),
                    "plan_points": dst_original.tolist(),
                }
            )
        print(f"Rotation {turns * 90}: {len(candidates)} geometrically supported candidates", flush=True)
    nn_candidate_count = len(candidates)
    if structural:
        from structural_matching import rescore_and_propose

        candidates = rescore_and_propose(
            wall_image, door_points, floors, structural["plan_walls"], structural["plan_doors"], candidates
        )
        print(
            f"Wall/door geometry: {nn_candidate_count} neural candidates -> {len(candidates)} retained hypotheses",
            flush=True,
        )
    runtime.save_json(output / "raw-candidates.json", candidates)
    selected = distinct_candidates(candidates)
    result = {
        "method": "SceneAligner_multiscale_mutual_patch_RANSAC",
        "manual_landmarks_used": False,
        "matcher": matcher,
        "structural_features_used": bool(structural),
        "nn_candidates_before_structure": nn_candidate_count,
        "candidates_before_nms": len(candidates),
        "candidates": selected,
        "physically_verified": False,
        "status": "no_supported_candidate" if not selected else "automatic_candidate",
    }
    if len(selected) >= 2:
        result["relative_score_gap"] = (selected[0]["score"] - selected[1]["score"]) / max(
            selected[0]["score"], 1e-9
        )
        result["ambiguous"] = result["relative_score_gap"] < 0.15
    else:
        result["ambiguous"] = True  # no competing hypothesis does not establish certainty
    runtime.save_json(output / "search.json", result)
    return result, pp, density


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups", type=Path, required=True, help="Root with group_*/reconstruction/manifest.json"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--floor-masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matcher", choices=["mutual", "topk"], default="mutual")
    parser.add_argument("--lens-json", type=Path)
    parser.add_argument(
        "--structural-inputs", type=Path, help="Semantic wall/door directory; parent has drawing structure"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from scenealigner.inference import load_model
    from scenealigner.preprocess import resize_plan
    from scenealigner.visualization import draw_alignment

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    start = perf_counter()
    plan = Image.open(args.plan).convert("RGB")
    matching_plan = plan
    plan_walls, plan_doors = None, None
    if args.structural_inputs:
        matching_plan = Image.open(args.structural_inputs.parent / "plan-structure.png").convert("RGB")
        if matching_plan.size != plan.size:
            raise ValueError("Structural plan resolution mismatch")
        plan_walls = cv2.imread(str(args.structural_inputs.parent / "plan-walls.png"), 0)
        plan_doors = json.loads(
            (args.structural_inputs.parent / "semantics.json").read_text(encoding="utf-8")
        )["plan_doors"]
    lenses = json.loads(args.lens_json.read_text(encoding="utf-8"))["cameras"] if args.lens_json else None
    plan_512, (scale, ox, oy) = resize_plan(plan)
    global_transform = np.array([[scale, 0, ox], [0, scale, oy], [0, 0, 1.0]])
    model = load_model()
    bank = prepare_bank(model, matching_plan, runtime.HERE / "cache")
    print(f"Searching {len(bank)} plan fragments", flush=True)
    summary = {
        "manual_landmarks_used": False,
        "reference_JSONs_opened": False,
        "fragments": len(bank),
        "groups": {},
    }
    predictions = args.output / "predictions"
    predictions.mkdir()
    for group in sorted(args.groups.glob("group_*")):
        if not (group / "reconstruction/manifest.json").is_file():
            continue
        print("Searching", group.name, flush=True)
        output = args.output / group.name
        output.mkdir()
        manifest = json.loads((group / "reconstruction/manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            raise ValueError("Incomplete group reconstruction")
        recon = dict(np.load(group / "reconstruction/reconstruction.npz"))
        structure = (
            {
                "semantics": args.structural_inputs,
                "images": manifest["images"],
                "floor_masks": args.floor_masks,
                "plan_walls": plan_walls,
                "plan_doors": plan_doors,
            }
            if args.structural_inputs
            else None
        )
        result, pp, density = search_one(model, bank, recon, plan, output, args.matcher, structure)
        summary["groups"][group.name] = {k: v for k, v in result.items() if k != "candidates"}
        if result["candidates"]:
            best = result["candidates"][0]
            matrix = (global_transform @ np.asarray(best["density_to_original"]))[:2]
            alignment = {"proj_params": pp, "M": matrix}
            overlay = draw_alignment(
                np.array(density), np.array(plan_512), matrix, np.asarray(best["source_points"])
            )
            Image.fromarray(overlay).save(output / "alignment.png")
            cameras = runtime.recover_homographies(
                recon,
                alignment,
                [Path(x) for x in manifest["images"]],
                args.floor_masks,
                plan.size,
                (scale, ox, oy),
                predictions,
                lens_models=lenses,
            )
            summary["groups"][group.name]["cameras"] = cameras
            for name, camera in cameras.items():
                path = predictions / f"{name}-calibration.json"
                if not path.is_file():
                    continue
                value = json.loads(path.read_text(encoding="utf-8"))
                value.update(
                    method="SceneAligner_grouped_fragment_search",
                    group=group.name,
                    search_ambiguous=result["ambiguous"],
                    search_score=best["score"],
                    structural_features_used=bool(structure),
                )
                runtime.save_json(path, value)
        summary["elapsed_sec"] = perf_counter() - start
        runtime.save_json(args.output / "summary.json", summary)
    if not summary["groups"]:
        raise ValueError("No complete group reconstructions")
    print("Fragment search completed", flush=True)


if __name__ == "__main__":
    main()
