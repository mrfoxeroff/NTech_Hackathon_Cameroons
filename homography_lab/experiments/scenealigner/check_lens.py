"""Plumb-line diagnostics on automatically traced wall/door edges, not plan points."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from lens_geometry import undistort_points
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar


def edge_chains(image, semantic):
    scale = min(1.0, 1280 / image.shape[1])
    reduced = cv2.resize(image, None, fx=scale, fy=scale)
    mask = cv2.resize(semantic, reduced.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY), (3, 3), 0), 60, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    chains = []
    for contour in contours:
        points = contour[::3, 0].astype(float)
        if len(points) < 40:
            continue
        smooth = gaussian_filter1d(points, 1.2, axis=0, mode="wrap")
        left, right = smooth - np.roll(smooth, 4, axis=0), np.roll(smooth, -4, axis=0) - smooth
        cosine = np.sum(left * right, axis=1) / (
            np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-9
        )
        straight = cosine >= np.cos(np.deg2rad(25))
        if (~straight).any():
            pivot = np.flatnonzero(~straight)[0]
            points, straight = np.roll(points, -pivot, axis=0), np.roll(straight, -pivot)
        padded = np.r_[False, straight, False].astype(int)
        starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
        for start, end in zip(starts, ends):
            chain = points[start:end]
            if len(chain) < 32 or np.linalg.norm(chain[-1] - chain[0]) < 0.10 * np.hypot(*reduced.shape[:2]):
                continue
            x, y = np.rint(chain).astype(int).T
            if (mask[y, x] > 0).mean() < 0.65:
                continue
            chains.append(chain / scale)
    unique = {}
    for chain in sorted(chains, key=len, reverse=True):
        center = np.mean(chain, axis=0)
        if any(
            np.linalg.norm(center - np.mean(other, axis=0)) < 25
            and abs(
                abs(
                    np.dot(
                        (chain[-1] - chain[0]) / np.linalg.norm(chain[-1] - chain[0]),
                        (other[-1] - other[0]) / np.linalg.norm(other[-1] - other[0]),
                    )
                )
                - 1
            )
            < 0.015
            for other in unique.values()
        ):
            continue
        unique[len(unique)] = chain
    return list(unique.values())


def residuals(chains, lens=None):
    rms, relative = [], []
    for chain in chains:
        if lens:
            chain, valid = undistort_points(chain, lens)
            if not valid.all():
                rms.append(float("inf"))
                relative.append(float("inf"))
                continue
        centered = chain - np.mean(chain, axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        value = float(np.sqrt(np.mean((centered @ vectors[1]) ** 2)))
        span = np.ptp(centered @ vectors[0])
        rms.append(value)
        relative.append(value / max(span, 1))
    return np.asarray(rms), np.asarray(relative)


def metric(chains, lens=None):
    rms, relative = residuals(chains, lens)
    finite = np.isfinite(relative)
    return {
        "chains": len(chains),
        "invalid_chains": int((~finite).sum()),
        "median_rms_pixels": float(np.median(rms)) if len(rms) and finite.all() else None,
        "median_relative_deviation": float(np.median(relative)) if len(rms) and finite.all() else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--semantics", type=Path, required=True)
    p.add_argument("--radial", type=Path, required=True)
    p.add_argument("--divisional", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    radial = json.loads(args.radial.read_text(encoding="utf-8"))["cameras"]
    divisional = json.loads(args.divisional.read_text(encoding="utf-8"))["cameras"]
    report = {
        "manual_landmarks_used": False,
        "interpretation": "Potential straight edges, not surveyed lines; held-out chains test straightness, not floorplan localization.",
        "cameras": {},
    }
    for path in sorted((args.inputs / "views").glob("*.jpg")):
        name = path.stem
        image = cv2.imread(str(path))
        mask = cv2.bitwise_or(
            cv2.imread(str(args.semantics / "wall" / (name + ".png")), 0),
            cv2.imread(str(args.semantics / "door" / (name + ".png")), 0),
        )
        chains = edge_chains(image, mask)
        train = [c for i, c in enumerate(chains) if i % 3 != 0]
        checks = chains[::3]
        result = {
            "training_chains": len(train),
            "check_chains": len(checks),
            "checks": {
                "original": metric(checks),
                "neural_radial": metric(checks, radial[name]),
                "neural_divisional": metric(checks, divisional[name]),
            },
        }
        preview = image.copy()
        for i, chain in enumerate(chains):
            cv2.polylines(
                preview,
                [np.rint(chain).astype(int)],
                False,
                (255, 200, 40) if i % 3 == 0 else (70, 230, 80),
                3,
            )
        cv2.imwrite(str(args.output / (name + "-chains.jpg")), preview)
        if len(train) >= 5 and len(checks) >= 3:
            lens = dict(divisional[name])
            intrinsic = np.array(lens["K"])
            w, h = lens["size"]
            max_r2 = (w / 2 / intrinsic[0, 0]) ** 2 + (h / 2 / intrinsic[1, 1]) ** 2

            def objective(k):
                trial = lens | {"k1": float(k)}
                _, values = residuals(train, trial)
                return float(np.mean(np.log1p((values / 0.002) ** 2))) if np.isfinite(values).all() else 1e6

            optimized = minimize_scalar(
                objective, bounds=(-0.8 / max_r2, 0.5 / max_r2), method="bounded", options={"xatol": 1e-5}
            )
            lens["k1"] = float(optimized.x)
            # Straightness is independent of the output's uniform focal scale.
            lens["estimation_source"] = "automatic_wall_edge_chains_train_split"
            result["plumbline_lens"] = lens
            result["checks"]["plumbline_divisional"] = metric(checks, lens)
        report["cameras"][name] = result
        print(name, result["check_chains"], result["checks"], flush=True)
    (args.output / "straightness.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
