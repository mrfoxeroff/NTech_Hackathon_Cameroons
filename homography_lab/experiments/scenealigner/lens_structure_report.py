"""Render the complete 2x2 study after inference; no candidate selection here."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from lens_geometry import project_calibration, undistort_points
from PIL import Image

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / "cache/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    config = read(args.config)
    output = args.config.parent
    variants = config["variants"]
    reports = {variant["id"]: read(variant["evaluation"]) for variant in variants}
    names = reports["combined"]["expected_cameras"]
    values = np.full((len(names), len(variants)), np.nan)
    rows = []
    for i, name in enumerate(names):
        cells = []
        for j, variant in enumerate(variants):
            score = reports[variant["id"]]["cameras"].get(name)
            value = score.get("median_error_px") if score else None
            if value is not None:
                values[i, j] = value
            cells.append(f"{value:.1f}" if value is not None else "нет привязки")
        rows.append("| " + name + " | " + " | ".join(cells) + " |")
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = plt.get_cmap("magma_r").copy()
    palette.set_bad("#dddddd")
    heat = ax.imshow(np.ma.masked_invalid(values), cmap=palette, vmin=0, vmax=650, aspect="auto")
    for i in range(len(names)):
        for j in range(len(variants)):
            ax.text(
                j,
                i,
                f"{values[i, j]:.1f}" if np.isfinite(values[i, j]) else "—",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if values[i, j] > 350 else "black",
            )
    short_labels = {
        "baseline": "Предыдущий\nвариант",
        "lens": "Коррекция\nобъектива",
        "structure": "Стены /\nдвери",
        "combined": "Коррекция +\nстены / двери",
    }
    ax.set_xticks(range(len(variants)), [short_labels[v["id"]] for v in variants])
    ax.set_yticks(range(len(names)), names)
    ax.set_title(
        "Медианная ошибка привязки, px схемы 612×658\nПоказаны все камеры; прочерк — отсутствие предсказания"
    )
    fig.colorbar(heat, ax=ax, label="px")
    fig.tight_layout()
    fig.savefig(output / "comparison-2x2.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    straightness = read(config["straightness"])
    edge_rows = []
    for name in names:
        diagnostic = straightness["cameras"][name]
        before = diagnostic["checks"]["original"]["median_relative_deviation"]
        after = diagnostic["checks"]["neural_divisional"]["median_relative_deviation"]
        improvement = (1 - after / before) * 100 if before and after is not None else None
        edge_rows.append(
            f"| {name} | {diagnostic['check_chains']} | {before:.5f} | {after:.5f} | {improvement:.1f}% |"
        )
    semantics = read(Path(config["semantics"]) / "semantics.json")
    combined = next(v for v in variants if v["id"] == "combined")
    cam2 = reports["combined"]["cameras"].get("cam_2")
    if cam2:
        prediction = read(Path(combined["predictions"]) / "cam_2-calibration.json")
        reference = read(Path(config["reference"]) / "cam_2_calibration.json")
        original = Image.open(Path(config["original_inputs"]) / "views/cam_2.jpg")
        corrected = np.asarray(
            Image.open(Path(config["corrected_inputs"]) / "views/cam_2.jpg").convert("RGB")
        ).copy()
        wall = cv2.imread(
            str(Path(config["semantics"]) / config["corrected_semantic_mode"] / "wall/cam_2.png"), 0
        )
        contours, _ = cv2.findContours(wall, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(corrected, contours, -1, (240, 80, 50), 4)
        plan = Image.open(Path(config["original_inputs"]) / "plan.jpg")
        source, target = np.asarray(reference["camera_points"]), np.asarray(reference["plan_points"])
        keep = np.asarray(reference["inliers"], bool)
        projected, valid = project_calibration(source, prediction)
        rectified_points, rv = undistort_points(source, prediction["lens"])
        fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
        axes[0].imshow(original)
        axes[1].imshow(corrected)
        for ax, points in [(axes[0], source), (axes[1], rectified_points)]:
            ax.scatter(*points[keep].T, marker="x", c="#00e68a", s=40)
            for index in np.flatnonzero(keep):
                ax.annotate(
                    str(index + 1),
                    points[index],
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="white",
                    fontsize=9,
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
                )
            ax.axis("off")
        axes[0].set_title("Камера 2: исходный кадр")
        axes[1].set_title("Коррекция и границы нейросетевой маски стен")
        ax = axes[2]
        ax.imshow(plan)
        ax.scatter(*target[keep].T, marker="x", c="#00884f", s=45, label="JSON, только проверка")
        ax.scatter(*projected[keep & valid].T, c="#ed8523", s=28, label="Автоматическая проекция")
        for index in np.flatnonzero(keep & valid):
            ax.plot(
                [target[index, 0], projected[index, 0]],
                [target[index, 1], projected[index, 1]],
                color="#ed8523",
                lw=1,
            )
        ax.set_xlim(0, plan.width)
        ax.set_ylim(plan.height, 0)
        ax.axis("off")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8)
        ax.set_title(f"Медиана {cam2['median_error_px']:.1f} px; P95 {cam2['p95_error_px']:.1f} px")
        fig.suptitle(
            "Контрольные точки добавлены после расчёта; для подгонки и выбора гипотез не использовались",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        fig.savefig(output / "camera-2-result.png", dpi=140, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    assessment = {
        "status": "partial_localization_improvement_not_ready_for_tracking",
        "manual_landmarks_used_for_inference": False,
        "original_tracking_changed": False,
        "automatic_calibrations_safe_to_integrate": False,
        "camera_2_existing_reference": cam2,
        "variants": variants,
        "all_reference_inlier_projections_valid": all(
            score.get("unprojectable_reference_inliers", 0) == 0
            for report in reports.values()
            for score in report["cameras"].values()
        ),
    }
    (output / "assessment.json").write_text(
        json.dumps(assessment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    before_cam2 = reports["baseline"]["cameras"].get("cam_2", {})
    local_success = (
        cam2
        and cam2["median_error_px"] is not None
        and cam2["median_error_px"] < before_cam2.get("median_error_px", float("inf"))
    )
    if not cam2 or cam2["median_error_px"] is None:
        raise ValueError("Review the interpretation: camera 2 has no complete projection in this repeat")
    headline = (
        "Локальное улучшение для камеры 2; готовой автоматической калибровки всех камер пока нет."
        if local_success
        else "Общая готовность не подтверждена; сравните все режимы в таблице."
    )
    reference_count = cam2["evaluated_reference_inliers"]
    within10 = round(cam2["fraction_within_10px"] * reference_count)
    total_count = sum(
        score["evaluated_reference_inliers"] for score in reports["combined"]["cameras"].values()
    )
    valid_count = sum(
        score["evaluated_reference_inliers"] - score.get("unprojectable_reference_inliers", 0)
        for score in reports["combined"]["cameras"].values()
    )
    no_door_names = ", ".join(
        name for name, record in semantics["cameras"].items() if record["door_fraction"] == 0
    )
    text = f"""# Коррекция объектива и сопоставление по стенам/дверям

**{headline}**
Сочетание коррекции и структурного поиска дало медиану {cam2["median_error_px"]:.2f} px и P95 {cam2["p95_error_px"]:.2f} px
по {reference_count} исходным контрольным инлайерам камеры 2. В пределах 10 px — {within10} из {reference_count}.
Это сравнение с имеющимися ручными JSON, не с независимой геодезической истиной и не проверка на новом помещении.

## Что проверено

Четыре режима используют одинаковые группы `[cam_1, cam_3]`, `[cam_2]`, `[cam_4]`, `[cam_5]` и одинаковые исходные соответствия для оценки.
Камеры не перегруппировывались после коррекции, чтобы отделить влияние искажений от изменения состава групп.
В 48 исходных точках учтены 40 заранее отмеченных инлайеров; исходные флаги исключений не изменялись.
JSON не читались при оценке параметров объектива, выделении стен/дверей, расчёте матриц и ранжировании кандидатов.
Нельзя автоматически выбрать лучший режим для каждой камеры по этой таблице и назвать такой выбор работой без контрольной разметки.

| Камера | Предыдущий вариант | Только коррекция | Только стены/двери | Коррекция + стены/двери |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Ошибки — медианы в пикселях схемы 612×658. Прочерк/«нет привязки» означает отказ, а не нулевую ошибку.
В комбинированном варианте удалось спроецировать {valid_count} из {total_count} исходных контрольных инлайеров.
Непроецируемые точки не удаляются молча: в списке ошибок они сохранены как `null`, а число таких инлайеров указано отдельно.

![Сравнение всех режимов](comparison-2x2.png)

![Камера 2](camera-2-result.png)

## Коррекция объектива

GeoCalib с официальными весами `distorted` оценивает параметры только по изображению.
Проверены `simple_radial` и предназначенная авторами для сильных искажений `simple_divisional`.
У первой модели обратное преобразование неприменимо к части рамки кадра; это обнаружено проверкой области обратимости.
Вторая обратима на всех дискретных пробах рамки для наших пяти кадров. При выводе остаются белые области без исходных пикселей,
а расширение поля зрения ограничено; это не восстановление скрытых деталей изображения.

Проверка кривизны использовала автоматически выделенные длинные рёбра рядом с нейросетевыми масками стен/дверей.
Это предполагаемые физически прямые линии, не ручная разметка. Отдельные цепочки отложены для проверки.
Ниже относительное отклонение от прямой, нормированное на длину цепочки: оно не улучшается только от уменьшения картинки.

| Камера | Проверочных цепочек | До | После divisional | Снижение отклонения |
|---|---:|---:|---:|---:|
{chr(10).join(edge_rows)}

Всего 3–9 проверочных цепочек на камеру; это диагностическое свидетельство, не полная метрическая калибровка.
Отдельно рассчитана подгонка одного коэффициента по обучающей части цепочек (`plumbline_divisional` в `straightness.json`);
в итоговую привязку этот дополнительный вариант не подмешивался.

## Стены и дверные проёмы

SegFormer выделяет стены/колонны и двери. Те же маски переносятся коррекцией в исправленные кадры без повторной сегментации —
так изменение геометрии не смешивается с изменением ответа сегментатора.
Нормали поверхности Pi3 отбрасывают горизонтальные поверхности, ошибочно похожие на стены.
На схеме автоматически подавляются мелкие линии мебели/штриховки, выделяются стены,
а дверные кандидаты ищутся по разрыву стены, дуге открывания и створке с учётом толщины стены.
Найдено {len(semantics["plan_doors"])} кандидатов; они не объявляются подтверждёнными дверями.

Поиск сочетает DINOv3 по фрагментам схемы с гипотезами поворота/масштаба/сдвига по автоматически полученным углам стен и дверям.
Гипотезы проверяются расстоянием до стен схемы, попаданием видимого пола в свободное пространство и согласованностью дверных кандидатов.
Часть ближайших гипотез уточняется численно. Ручных опорных координат нет.

Сегментатор дверей не дал масок выше порога на камерах: **{no_door_names or "таких камер нет"}**.
В текущем прогоне успех камеры 2 относится к стенам и коррекции, а не к распознанным дверям.
В группах с камерами 3–5 дверные кандидаты есть, но устойчивое правильное место на схеме они не определили.
В исходной версии обнаружен и исправлен случай, когда дверная точка вне анализируемой карты плотности влияла на выбор гипотезы;
финальная таблица использует согласованную область поиска.

## Что осталось

- Общего улучшения на всех камерах нет: похожие стены и коридоры допускают несколько правдоподобных положений.
- У камеры 2 разрыв внутренних оценок лучших гипотез мал: правильный top-1 в этом офисе не делает поиск надёжным в общем случае.
- Нужно отдельно проверить более сильное распознавание дверей и их последовательности, а также устойчивость к выбору кадров.
- Для дальнейшей геометрии полезно сравнить предсказанный 3D-пол с аналитической проекцией на плоскость по внутренним параметрам и вертикали.
- Нужна проверка на другом помещении и независимых контрольных данных.

Матрицы оставлены экспериментальными; существующий трекинг и его калибровки не менялись.
**Гомография исправленного изображения не применяется напрямую к исходным пикселям:** сначала выполняется обратное искажение,
затем гомография. Этот порядок реализован и проверен в `lens_geometry.py`; модель объектива сохранена вместе с каждой новой матрицей.

## Файлы и повторение

Все пути и режимы записаны в `experiment.json`. Параметры объектива — `{Path(config["corrected_inputs"]) / "lens.json"}`.
Проверка рёбер — `{config["straightness"]}`. Оценки и предсказания каждого режима указаны в конфигурации.
Полный новый прогон выполняет `lens_structure_experiment.py`; инструкция — в `LENS_STRUCTURE.md` каталога эксперимента.
"""
    (output / "RESULTS.md").write_text(text, encoding="utf-8")
    print("Saved RESULTS.md, comparison-2x2.png, camera-2-result.png and assessment.json")


if __name__ == "__main__":
    main()
