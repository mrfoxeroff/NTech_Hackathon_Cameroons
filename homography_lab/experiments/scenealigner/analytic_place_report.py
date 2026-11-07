"""Render completed inference and evaluation, never part of candidate selection."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lens_geometry import project_calibration


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--background", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output-prefix", default="comparison")
    args = p.parse_args()
    if (args.run / (args.output_prefix + ".png")).exists():
        raise FileExistsError("Use a fresh report path")
    rows = [
        ("Границы пола", "search-adaptive-rays", "evaluation-adaptive-rays.json", "boundary_only"),
        (
            "Границы + видимость",
            "search-adaptive-rays",
            "evaluation-adaptive-rays.json",
            "boundary_and_visibility",
        ),
        (
            "Дверь + границы + видимость",
            "search-doors-adaptive-rays",
            "evaluation-doors-adaptive-rays.json",
            "doors_boundary_and_visibility",
        ),
        (
            "Дополнительно: размер двери",
            "search-doors-size",
            "evaluation-doors-size.json",
            "doors_boundary_and_visibility",
        ),
    ]
    reference = read(args.reference)
    use = np.asarray(reference["inliers"], bool)
    source = np.asarray(reference["camera_points"])[use]
    target = np.asarray(reference["plan_points"])[use]
    plan = cv2.cvtColor(cv2.imread(str(args.plan)), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 4, figsize=(17, 7), constrained_layout=True)
    for axis, (title, folder, evaluation, mode) in zip(axes, rows):
        data = read(args.run / folder / "search.json")["modes"][mode]
        result = read(args.run / evaluation)["modes"][mode]["best_hypothesis"]
        projected, valid = project_calibration(source, data["candidates"][0]["calibration"])
        axis.imshow(plan)
        axis.scatter(target[:, 0], target[:, 1], c="#00a54a", marker="x", s=45, label="Контрольные JSON")
        axis.scatter(projected[valid, 0], projected[valid, 1], c="#d62728", s=22, label="Лучший кандидат")
        for a, b in zip(target[valid], projected[valid]):
            axis.plot([a[0], b[0]], [a[1], b[1]], color="#d62728", alpha=0.4, linewidth=0.7)
        axis.set_title(f"{title}\nМедиана {result['median_error_px']:.1f} px; решение: отказ", fontsize=11)
        axis.set_xlim(0, plan.shape[1])
        axis.set_ylim(plan.shape[0], 0)
        axis.axis("off")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Камера 2: автоматический выбор места остаётся неоднозначным\nКонтрольные точки используются только при оценке",
        fontsize=13,
    )
    fig.savefig(args.run / (args.output_prefix + ".png"), dpi=145)
    plt.close(fig)
    image = cv2.imread(str(args.background))
    geometry = dict(np.load(args.run / "features/geometry.npz"))
    for x, y in geometry["source_boundary_pixels"].astype(int):
        cv2.circle(image, (x, y), 4, (0, 220, 0), -1)
    doors = read(args.run / "doors/doors.json")
    for x, y in np.asarray(doors["stable_image_anchors"]).astype(int):
        cv2.circle(image, (x, y), 14, (0, 0, 255), -1)
    image = cv2.resize(image, (1296, 972))
    cv2.imwrite(str(args.run / "automatic-features.jpg"), image)


if __name__ == "__main__":
    main()
