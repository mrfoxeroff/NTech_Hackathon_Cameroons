"""Post-hoc evaluation and visual comparison; never selects inference candidates."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from evaluate import evaluate_camera
from PIL import Image

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / "cache/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--search-dir", default="fragment-search")
    parser.add_argument("--evaluation", default="evaluation.json")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    root = args.run
    old = read(HERE / "scenealigner-five-reference-evaluation.json")["cameras"]
    current = read(root / args.evaluation)
    groups = read(root / "groups.json")["groups"]
    search_root = root / args.search_dir
    summary = read(search_root / "summary.json")
    suffix = "-" + args.suffix if args.suffix else ""
    plan = Image.open(HERE / "inputs/office-five/plan.jpg")
    oracle = {
        "diagnostic_only": True,
        "references_used_for_candidate_selection": True,
        "inference_candidates_unchanged": True,
        "not_an_automatic_result": True,
        "cameras": {},
    }
    rows = []
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    fig.suptitle(
        "Привязка: общий прогон → группы и поиск фрагментов\nЗелёные кресты — JSON, оранжевые — автоматическое предсказание",
        fontsize=14,
    )
    for index, name in enumerate(current["expected_cameras"]):
        reference = read(args.reference / f"{name}_calibration.json")
        src, dst = np.array(reference["camera_points"]), np.array(reference["plan_points"])
        use = np.asarray(reference["inliers"], bool)
        pred_path = search_root / f"predictions/{name}-calibration.json"
        best = read(pred_path) if pred_path.is_file() else None
        new = current["cameras"].get(name)
        group_index = next(i for i, group in enumerate(groups, 1) if name in group)
        search = read(search_root / f"group_{group_index:02d}/search.json")
        oracle_error = None
        if best and search["candidates"]:
            selected = np.array(search["candidates"][0]["density_to_original"])
            results = []
            for rank, candidate in enumerate(search["candidates"], 1):
                hypothesis = dict(best)
                hypothesis["homography"] = (
                    np.array(candidate["density_to_original"])
                    @ np.linalg.inv(selected)
                    @ np.array(best["homography"])
                ).tolist()
                score = evaluate_camera(hypothesis, reference)
                results.append(
                    {
                        "rank": rank,
                        "median_error_px": score["median_error_px"],
                        "p95_error_px": score["p95_error_px"],
                    }
                )
            oracle["cameras"][name] = results
            oracle_error = min(r["median_error_px"] for r in results if r["median_error_px"] is not None)
        new_text = (
            f"{new['median_error_px']:.1f}" if new and new["median_error_px"] is not None else "нет привязки"
        )
        oracle_text = f"{oracle_error:.1f}" if oracle_error is not None else "—"
        rows.append(f"| {name} | {old[name]['median_error_px']:.1f} | {new_text} | {oracle_text} |")
        for column, prediction in enumerate(
            [read(HERE / f"runs/office-five-518/{name}-calibration.json"), best]
        ):
            ax = axes[index, column]
            ax.imshow(plan)
            ax.scatter(*dst[use].T, marker="x", color="#00884f", s=35, zorder=4)
            if prediction:
                xy = np.column_stack((src, np.ones(len(src)))) @ np.asarray(prediction["homography"]).T
                xy = xy[:, :2] / xy[:, 2:3]
                for a, b in zip(dst[use], xy[use]):
                    ax.plot([a[0], b[0]], [a[1], b[1]], color="#ed8523", alpha=0.65, lw=0.8)
                ax.scatter(*xy[use].T, color="#ed8523", s=20, zorder=5)
            ax.set_xlim(0, plan.width)
            ax.set_ylim(plan.height, 0)
            ax.axis("off")
            value = (
                f"{old[name]['median_error_px']:.1f} px"
                if column == 0
                else new_text + " px"
                if best
                else "Нет привязки"
            )
            ax.set_title(f"{name}: " + ("до — " if column == 0 else "после — ") + value)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(root / f"before-after{suffix}.png", dpi=130, facecolor="white")
    plt.close(fig)
    (root / f"oracle-top10-diagnostic{suffix}.json").write_text(
        json.dumps(oracle, indent=2) + "\n", encoding="utf-8"
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for index, ax in enumerate(axes.ravel()):
        if index >= len(groups):
            ax.axis("off")
            continue
        search = read(search_root / f"group_{index + 1:02d}/search.json")
        ax.imshow(plan)
        density = np.asarray(Image.open(search_root / f"group_{index + 1:02d}/density.png").convert("L"))
        ys, xs = np.where(density < 220)
        outline = np.array(
            [
                [xs.min(), ys.min(), 1],
                [xs.max(), ys.min(), 1],
                [xs.max(), ys.max(), 1],
                [xs.min(), ys.max(), 1],
                [xs.min(), ys.min(), 1],
            ]
        )
        for rank, candidate in enumerate(search["candidates"][:3], 1):
            xy = outline @ np.asarray(candidate["density_to_original"]).T
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                label=f"#{rank}: {candidate['score']:.3f}",
                lw=2,
                linestyle="-" if rank == 1 else "--",
            )
        label = (
            " — нет гипотезы"
            if not search["candidates"]
            else " — неоднозначно"
            if search["ambiguous"]
            else " — кандидат, не проверен"
        )
        ax.set_title(", ".join(groups[index]) + label)
        ax.set_xlim(0, plan.width)
        ax.set_ylim(plan.height, 0)
        ax.axis("off")
        if search["candidates"]:
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Гипотезы фрагментов, ранжированные без точек JSON", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(root / f"fragment-candidates{suffix}.png", dpi=130, facecolor="white")
    plt.close(fig)
    confirmed = ", ".join("[" + ", ".join(group) + "]" for group in groups)
    text = f"""# Группы камер и поиск фрагментов

Проверенные алгоритмом группы: **{confirmed}**.
Это компоненты графа подтверждённых совпадений, а не утверждение, что остальные камеры физически не пересекаются.
Камеры 1 и 2 визуально показывают один теннисный стол, но надёжное геометрическое ребро для камеры 2 не получено;
она оставлена отдельно. Номера комнат, положения камер и точки JSON не используются при группировке и поиске.

## Изменения

- LoFTR сопоставляет полные изображения и увеличенные перекрывающиеся участки; проверяются эпиполярная геометрия и отложенные совпадения.
- Второй метод, SIFT на исходном разрешении, проверяет предложенную геометрию; ложные совпадения пола и дверей отбрасываются.
  Статистический показатель основан на приближённой модели случайных перестановок; это не гарантия отсутствия ложных связей.
- Каждая группа реконструируется отдельно, поэтому удалённым камерам не навязывается общее взаимное положение.
- План автоматически разбит на {summary["fragments"]} перекрывающихся фрагментов нескольких масштабов и форм.
  Для каждого проверяются четыре поворота вида сверху. Используются реальные взаимные максимумы сходства признаков,
  а не усреднение координат нескольких возможных мест. Вырожденные привязки отбрасываются.
- Сохраняются до 10 разных гипотез, их геометрические показатели и признак неоднозначности. Ранжирование не читает JSON.
- Отсутствующие предсказания учитываются явно, поэтому отказ от привязки не выглядит улучшением точности.

## Проверка на имеющихся JSON

Ошибки в пикселях схемы 612×658; используются те же исходные флаги `inliers`, что и в прежнем прогоне.
JSON — существующая ручная разметка, не независимая геодезическая истина.

| Камера | До, медиана | Автоматический top-1, медиана | Лучший из top-10 с подсказкой JSON, только диагностика |
|---|---:|---:|---:|
{chr(10).join(rows)}

Покрытие: {len(current["cameras"])}/{len(current["expected_cameras"])} камер. Нет предсказания: {", ".join(current["missing_predictions"]) or "нет"}.
Последний столбец использует контрольные точки для выбора из кандидатов и **не является автоматическим результатом**.
Он показывает, содержит ли поиск подходящий вариант; автоматический top-1 им не заменяется.

![Сравнение](before-after{suffix}.png)

![Кандидаты](fragment-candidates{suffix}.png)

## Ограничения

Внутренняя согласованность совпадений и разрыв между оценками кандидатов не гарантируют правильное место на плане.
Искажение широкоугольных камер не исправлено. Автоматическое ребро камеры 2 с игровой зоной не подтверждено.
Пороги и способ поиска изменялись по визуальным и геометрическим диагностическим результатам, без контрольных координат;
итоговая проверка относится к этому офису и не доказывает обобщение на другое помещение.
Исходный трекинг и его калибровки не изменены. Все матрицы остаются экспериментальными кандидатами.

Оценка: `{args.evaluation}`. Группы: `groups.json`. Полные списки: `{args.search_dir}/group_*/search.json`.
Повторный запуск описан в основном `README.md` эксперимента.
"""
    (root / f"RESULTS{suffix}.md").write_text(text, encoding="utf-8")
    assessment = {
        "status": "not_ready_for_tracking",
        "automatic_calibrations_safe_to_integrate": False,
        "manual_landmarks_used_for_inference": False,
        "groups": groups,
        "predicted_cameras": sorted(current["cameras"]),
        "missing_predictions": current["missing_predictions"],
        "evaluation": args.evaluation,
        "search": args.search_dir,
        "reason": "Reference errors remain too large; candidate score gaps do not establish physical accuracy.",
    }
    (root / f"assessment{suffix}.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    print("Saved grouped report, before-after.png and fragment-candidates.png")


if __name__ == "__main__":
    main()
