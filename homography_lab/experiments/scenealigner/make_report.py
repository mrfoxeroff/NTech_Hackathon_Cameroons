"""Render measured results; calibration references are used only for assessment."""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / "cache/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    report = read(HERE / "scenealigner-five-reference-evaluation.json")
    manifest = read(HERE / "runs/office-five-518/manifest.json")
    reference_provenance = read(HERE / "reference/provenance.json")
    diagnosis = read(HERE / "geometry-five-diagnostic.json")
    baseline = read(HERE / "c3po-reference-evaluation.json")
    pair = read(HERE / "scenealigner-reference-evaluation.json")
    total = sum(m["evaluated_reference_inliers"] for m in report["cameras"].values())
    within = sum(
        round(m["fraction_within_10px"] * m["evaluated_reference_inliers"])
        for m in report["cameras"].values()
    )
    if within != 0:
        raise ValueError(
            "Named experiment results changed: review the interpretation before regenerating this failure report"
        )
    plan = Image.open(HERE / "inputs/office-five/plan.jpg")
    fig, axes = plt.subplots(5, 2, figsize=(14, 20), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.suptitle(
        "Пять камер: автоматическая привязка не прошла проверку\n"
        "Зелёные кресты — JSON; оранжевые точки — предсказание; серые — исключённые точки JSON",
        fontsize=14,
    )
    for row, (name, metric) in enumerate(report["cameras"].items()):
        reference = read(HERE / f"reference/{name}_calibration.json")
        prediction = read(HERE / f"runs/office-five-518/{name}-calibration.json")
        source = np.array(reference["camera_points"])
        target = np.array(reference["plan_points"])
        use = np.asarray(reference["inliers"], bool)
        projected = np.column_stack((source, np.ones(len(source)))) @ np.asarray(prediction["homography"]).T
        xy = projected[:, :2] / projected[:, 2:3]
        left, right = axes[row]
        left.imshow(Image.open(HERE / f"inputs/office-five/views/{name}.jpg"))
        left.scatter(*source[use].T, c="#00ff9a", marker="x", s=40)
        left.scatter(*source[~use].T, c="gray", marker="x", s=30)
        for index, point in enumerate(source):
            left.annotate(
                str(index + 1),
                point,
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=9,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
            )
        left.set_title(f"Камера {row + 1}: контрольные точки, без подбора по ним")
        left.axis("off")
        right.imshow(plan)
        right.scatter(*target[use].T, c="#00884f", marker="x", s=45, zorder=4)
        right.scatter(*target[~use].T, c="gray", marker="x", s=25, zorder=4)
        for index in np.flatnonzero(use):
            end = xy[index]
            right.plot(
                [target[index, 0], end[0]], [target[index, 1], end[1]], color="#f17720", alpha=0.65, lw=1
            )
            right.annotate(
                str(index + 1),
                target[index],
                fontsize=8,
                color="#005e33",
                xytext=(3, 3),
                textcoords="offset points",
            )
        right.scatter(*xy[use].T, c="#f17720", s=24, zorder=5)
        outside = int(((xy[use] < 0).any(1) | (xy[use] >= plan.size).any(1)).sum())
        right.set_xlim(0, plan.width)
        right.set_ylim(plan.height, 0)
        right.set_title(
            f"Медиана {metric['median_error_px']:.1f} px; ≤10 px: 0/{use.sum()}"
            + (f"; вне плана: {outside}" if outside else "")
        )
        right.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.956))
    fig.savefig(HERE / "comparison-five.png", dpi=140, facecolor="white")
    plt.close(fig)

    rows = []
    for name, score in report["cameras"].items():
        c3po = baseline["cameras"].get(name)
        two = pair["cameras"].get(name)
        c3po_text = format(c3po["median_error_px"], ".1f") if c3po else "—"
        two_text = format(two["median_error_px"], ".1f") if two else "—"
        rows.append(
            f"| {name} | {c3po_text} | {two_text} | {score['median_error_px']:.1f} | "
            f"{score['p95_error_px']:.1f} | 0/{score['evaluated_reference_inliers']} |"
        )
    text = f"""# Результат эксперимента на пяти камерах

**Автоматическая привязка к схеме пока не работает с достаточной точностью.**
Все пять камер не прошли сравнение с существующими точками из GitHub: ни одна из 40 учтённых точек не попала в радиус 10 пикселей.
Существующий проект и его калибровки не изменены. Экспериментальные матрицы не подключены к трекингу.

## Что проверено

Из пяти MKV подготовлены фоновые кадры: для неподвижных камер используется медиана семи кадров по длительности записи.
SegFormer выделяет пол. GeoCalib оценивает направление вертикали, Pi3 восстанавливает общую 3D-геометрию,
SceneAligner с адаптированным DINOv3 сопоставляет её вид сверху со схемой. Из предсказанных точек пола восстанавливаются гомографии.
Нейросети используются с готовыми весами; обучения в этом эксперименте нет.

Ручные точки, номера комнат и позиции камер не передаются в автоматический запуск. JSON открываются отдельной оценкой после предсказания.
Для отдельной диагностики геометрии точки JSON явно используются при подгонке преобразования — это контроль с подсказкой, а не автоматический результат.

## Измерения

Ошибки в пикселях исходной схемы 612×658. Медиана и P95 считаются по точкам, отмеченным `inliers` в исходных JSON.
Исходно 48 точек, учтено 40, исключено 8 по уже существующим флагам; исключённые ошибки также сохранены в JSON оценки.
Это сравнение с имеющейся ручной разметкой, не с геодезической истиной. Переводить пиксели в метры без масштаба плана нельзя.

| Камера | C3Po, медиана | SceneAligner, 2 камеры, медиана | SceneAligner, 5 камер, медиана | 5 камер, P95 | В пределах 10 px |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Добавление камер уменьшило ошибки первых двух в этом прогоне, но ни один вариант ещё не пригоден для отображения людей на плане.
Для C3Po камеры 3–5 здесь не запускались; прочерк означает отсутствие измерения.

![Сравнение](comparison-five.png)

## Где возникает ошибка

При диагностической подгонке масштаба, поворота и сдвига по JSON камеры 1 проверка на точках другой камеры даёт:

- камера 2: медиана {diagnosis["cross_camera_holdout"]["fit_cam_1_test_cam_2"]["median_px"]:.1f} px;
- камера 3: {diagnosis["cross_camera_holdout"]["fit_cam_1_test_cam_3"]["median_px"]:.1f} px;
- камера 4: {diagnosis["cross_camera_holdout"]["fit_cam_1_test_cam_4"]["median_px"]:.1f} px;
- камера 5: {diagnosis["cross_camera_holdout"]["fit_cam_1_test_cam_5"]["median_px"]:.1f} px.

Это указывает на две разные проблемы: неверное сопоставление реконструкции со схемой и несогласованные взаимные положения удалённых камер.
Причина несогласованности окончательно не установлена; слабое перекрытие — вероятное объяснение.
Проверка SIFT с геометрическим фильтром не подтвердила ни одного ребра при заранее заданном строгом пороге 20 инлайеров.
Больше всего инлайеров у 1–3 (19); отсутствие достаточных совпадений не доказывает, что перекрытия нет.

Внутренняя ошибка гомографии по другим предсказаниям той же сети очень мала (менее пикселя),
но это лишь согласованность предсказаний. Она не подтверждает правильное место на схеме.
Искажения объектива в этом прогоне не исправлялись; их вклад отдельно не оценён.

## Ресурсы

На {manifest["gpu"]} прогон пяти фоновых кадров занял {manifest["elapsed_sec"]:.1f} с,
пик выделенной памяти тензоров PyTorch — {manifest["peak_cuda_allocated_bytes"] / 1024**3:.2f} ГиБ.
Это время одного запуска моделей и сохранения результатов; подготовка фонов и масок в него не входит.
Это не обработка всех кадров пяти двадцатиминутных роликов и не замер скорости трекинга.
Проверен также запуск на двух изображениях официального примера; это проверка запуска, не воспроизведение метрик статьи.

## Дальнейший вектор

1. Надёжно определять перекрывающиеся ракурсы нейросетевым сопоставлением и геометрической проверкой; не соединять несвязанные камеры в одну реконструкцию.
2. Отдельно улучшать поиск фрагмента плана: несколько гипотез комнат/коридоров, двери и стыки стен, оценка нескольких поворотов и масштабов.
3. Для связывания удалённых групп попробовать проход с подвижной камерой через коридор, если такие данные можно получить.
   Дополнительные кадры одной неподвижной камеры нового ракурса не дают.
4. Проверить модель искажения широкоугольных камер и устойчивость результата к выбору фоновых кадров.
5. Если готовые веса не подходят — собрать данные «изображения + план + независимые соответствия» для адаптации и проверки;
   отделять помещения обучения от помещений теста. Пяти JSON текущего офиса недостаточно для обучения универсальной модели.
6. Подключать к трекингу только прошедшие независимую проверку привязки; для неуверенного результата оставлять положение неизвестным.

## Происхождение

Контрольная разметка и схема взяты из `main` репозитория `mrfoxeroff/NTech_Hackathon_Cameroons`,
коммит `{reference_provenance["commit"]}`. Хеши файлов: [reference/provenance.json](reference/provenance.json).
Ревизии весов и локальные пути: [model-files.json](model-files.json).
Зависимости: [requirements-lock.txt](requirements-lock.txt).
Код и команды: [README.md](README.md).
"""
    (HERE / "RESULTS.md").write_text(text, encoding="utf-8")
    summary = {
        "status": "rejected_against_existing_reference",
        "manual_landmarks_used_for_inference": False,
        "reference_points_within_10px": within,
        "evaluated_points": total,
        "reference_commit": reference_provenance["commit"],
        "production_code_changed": False,
        "automatic_calibrations_safe_to_integrate": False,
        "cameras": report["cameras"],
    }
    (HERE / "assessment.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Created RESULTS.md, comparison-five.png and assessment.json")


if __name__ == "__main__":
    main()
