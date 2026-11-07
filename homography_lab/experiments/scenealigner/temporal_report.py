"""Temporal reliability report. Reads completed evaluations, never selects a model."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from lens_geometry import project_calibration
from PIL import Image

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / "cache/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--floor-mask", type=Path, required=True)
    parser.add_argument("--frozen-calibration", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    full = read(root / "evaluation.json")
    fixed = read(root / "fixed-camera-evaluation.json")
    samples = read(root / "samples.json")
    registration = read(root / "registration.json")
    protocol = read(root / "inference/protocol.json")
    fixed_protocol = read(root / "fixed-camera-inference/protocol.json")
    cases = sorted(samples["cases"], key=lambda case: case["actual_seconds"])
    full_cases = {case["id"]: case for case in protocol["cases"]}
    fixed_cases = {case["id"]: case for case in fixed_protocol["cases"]}

    def value(record, key="fresh_score", metric="median_error_px"):
        return (record.get(key) or {}).get(metric)

    def display(number, digits=1):
        return f"{number:.{digits}f}" if number is not None else "нет"

    times = np.array([case["actual_seconds"] / 60 for case in cases])
    series = [
        (
            "Сохранена вся привязка",
            "#008861",
            [value(full["cases"][case["id"]], "frozen_score") for case in cases],
        ),
        (
            "Фиксированы параметры камеры, новый поиск",
            "#da8620",
            [value(fixed["cases"][case["id"]]) for case in cases],
        ),
        ("Полный пересчёт", "#bc3d49", [value(full["cases"][case["id"]]) for case in cases]),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, color, values in series:
        ax.plot(
            times,
            [v if v is not None else np.nan for v in values],
            marker="o",
            lw=1.5,
            label=label,
            color=color,
        )
    ax.axhline(
        10, color="gray", linestyle="--", lw=1, label="10 px по медиане; дополнительно требуется P95 ≤20 px"
    )
    ax.set_yscale("log")
    ax.set_ylim(4, 650)
    ax.set_xlabel("Минуты от начала записи")
    ax.set_ylabel("Медианная ошибка по прежним JSON, px схемы; логарифмическая шкала")
    ax.set_title(
        "Камера 2: надёжность на 10 новых кадрах\nТочки проверки перенесены регистрацией неподвижной сцены; новой ручной разметки нет"
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(root / "temporal-errors.png", dpi=140, facecolor="white")
    plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, case in zip(axes.ravel(), cases):
        ax.imshow(Image.open(case["image"]))
        seconds = case["actual_seconds"]
        ax.set_title(
            f"{int(seconds // 60):02d}:{int(seconds % 60):02d} — "
            + ("стресс-кадр" if case["kind"] == "foreground_stress" else "равномерная выборка"),
            fontsize=10,
        )
    fig.suptitle("Все выбранные кадры, включая людей и открытые двери", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(root / "sample-contact.png", dpi=120, facecolor="white")
    plt.close(fig)
    floor = cv2.imread(str(args.floor_mask), 0)
    y, x = np.unravel_index(
        np.argmax(cv2.distanceTransform((floor > 128).astype(np.uint8), cv2.DIST_L2, 5)), floor.shape
    )
    probe = np.array([[float(x), float(y)]])
    frozen = read(args.frozen_calibration)
    stable, _ = project_calibration(probe, frozen)
    plan = Image.open(args.plan)
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    for ax, dataset, title in [
        (axes[0], full_cases, "Полный пересчёт"),
        (axes[1], fixed_cases, "Фиксированы параметры камеры, поиск новый"),
    ]:
        ax.imshow(plan)
        ax.scatter(*stable[0], c="#008861", s=90, marker="*", label="Сохранённая привязка", zorder=10)
        for order, case in enumerate(cases, 1):
            item = dataset[case["id"]]
            if not item.get("prediction"):
                continue
            point, valid = project_calibration(probe, read(item["prediction"]))
            if valid[0]:
                ax.plot(
                    [stable[0, 0], point[0, 0]], [stable[0, 1], point[0, 1]], c="#bc3d49", alpha=0.5, lw=0.8
                )
                ax.scatter(*point[0], c="#bc3d49", s=25)
                ax.annotate(str(order), point[0], xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlim(0, plan.width)
        ax.set_ylim(plan.height, 0)
        ax.axis("off")
        ax.set_anchor("N")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.suptitle(
        "Один пиксель пола при неподвижной камере получает разные места на плане\nПиксель автоматически выбран из маски пола; это не положение реального человека",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(root / "location-jumps.png", dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    rows = []
    for case in cases:
        name = case["id"]
        seconds = case["actual_seconds"]
        a, b = full["cases"][name], fixed["cases"][name]
        rows.append(
            f"| {int(seconds // 60):02d}:{int(seconds % 60):02d} | {'стресс' if case['kind'] == 'foreground_stress' else 'равномерно'} | "
            f"{display(value(a, 'frozen_score'), 2)} | {display(value(b))} | {display(value(a))} | {display(value(a, metric='p95_error_px'))} |"
        )
    displacements = [
        probe["registration"]["median_corner_displacement_original_px"]
        for probe in registration["probes"]
        if probe["registration"].get("accepted")
    ]
    frozen_values = [
        value(record, "frozen_score")
        for record in full["cases"].values()
        if value(record, "frozen_score") is not None
    ]
    controls = []
    for name, record in full["controls"].items():
        controls.append(
            f"| {name} | {display(value(record), 3)} | {display(value(record, metric='p95_error_px'), 3)} |"
        )
    for name, record in fixed["controls"].items():
        controls.append(
            f"| Те же исходные данные, фиксированы линза и вертикаль | {display(value(record), 3)} | {display(value(record, metric='p95_error_px'), 3)} |"
        )
    summary = full["summary"]
    consensus_section = ""
    consensus_assessment = {}
    consensus_paths = [
        ("Полный пересчёт", "consensus-full-v2"),
        ("Фиксированы линза и вертикаль", "consensus-fixed-camera-v2"),
    ]
    if all((root / folder / "evaluation.json").is_file() for _, folder in consensus_paths):
        consensus_rows = []
        passing, trials_count = 0, 0
        split_disagreement = []
        for label, folder in consensus_paths:
            evaluated = read(root / folder / "evaluation.json")
            passing += sum(trial["passes_reference_criteria"] for trial in evaluated["trials"])
            trials_count += len(evaluated["trials"])
            split_disagreement.append(f"{label}: {evaluated['disjoint_group_projection_rms_px']:.1f} px")
            for trial in evaluated["trials"]:
                if trial["threshold_fraction"] != 0.02:
                    continue
                score = trial["score"] or {}
                consensus_rows.append(
                    f"| {label} | {trial['subset']} | {trial.get('support_frames', 0)}/{trial.get('input_frames', 0)} | "
                    f"{display(score.get('median_error_px'))} | {display(score.get('p95_error_px'))} |"
                )
                if trial["subset"] == "all_frames":
                    consensus_assessment[folder] = {
                        "support_frames": trial["support_frames"],
                        "frames": trial["input_frames"],
                        "median_error_px": score.get("median_error_px"),
                        "p95_error_px": score.get("p95_error_px"),
                        "passes_reference_criteria": trial["passes_reference_criteria"],
                    }
        consensus_section = f"""## Продолжение: согласование гипотез между кадрами

Добавлен отдельный прототип без контрольных координат: 16 пространственно разнесённых проб выбираются из нейросетевой маски пола.
Гипотезы разных кадров сравниваются по проекциям этих проб. Один кадр даёт не более одного голоса в кластере;
дубликаты кандидатов одного изображения поддержку не увеличивают. Выбирается медоид — существующая матрица с её моделью линзы,
а не среднее коэффициентов несовместимых гомографий. Регистрация кадра применяется перед обратным искажением.

До сравнения с JSON заданы радиусы 1%, 2% и 4% диагонали схемы; основной — 2% (около 18 px).
Проверены все 10 кадров и два непересекающихся поднабора по 5 кадров, чередующихся по времени.

| Исходные гипотезы | Поднабор | Поддержка победителя | Медиана, px | P95, px |
|---|---|---:|---:|---:|
{chr(10).join(consensus_rows)}

Из {trials_count} вариантов разбиения и радиуса {passing} прошли критерии точности по JSON.
Это варианты на одной выборке, не независимые испытания на {trials_count} объектах.
Для объединения с фиксированной камерой поддержка {consensus_assessment["consensus-fixed-camera-v2"]["support_frames"]} из {len(cases)} кадров
дала медианную ошибку {consensus_assessment["consensus-fixed-camera-v2"]["median_error_px"]:.1f} px.
Результаты двух отдельных групп также расходятся: RMS проекций автоматически выбранных проб — {"; ".join(split_disagreement)}.
**Согласие между кадрами само по себе не подтверждает правильное место на схеме.**

Результаты: `consensus-full-v2/` и `consensus-fixed-camera-v2/`.
Контрольные JSON открывались только последующим `consensus_evaluate.py`; выбранные гипотезы после проверки не заменялись.

"""
    assessment = {
        "status": "automatic_relocalization_not_reliable_on_temporal_samples",
        "camera": samples["camera"],
        "new_frames": len(cases),
        "full_recalibration_pass_count": summary["fresh_pass_count"],
        "fixed_camera_new_search_pass_count": fixed["summary"]["fresh_pass_count"],
        "registered_frozen_calibration_pass_count": summary["frozen_pass_count"],
        "frozen_result_is_not_independent_person_position_ground_truth": True,
        "automatic_initialization_ready_for_tracking": False,
        "original_tracking_changed": False,
        "reference_labels_used_for_inference": False,
        "temporal_consensus": consensus_assessment,
    }
    (root / "assessment.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    median_only_passes = sum(
        value(record) is not None and value(record) <= 10 for record in full["cases"].values()
    )
    text = f"""# Надёжность камеры 2 на других кадрах

**Автоматический повторный поиск пока ненадёжен. Сохранённая привязка не показывает дрейфа при наблюдаемой неподвижности камеры.**

Проверено {len(cases)} новых моментов из {samples["duration_seconds"] / 60:.1f}-минутной записи: 8 равномерных и 2 стресс-кадра.
Критерий точности по прежним JSON: медиана ≤10 px, P95 ≤20 px, все исходные инлайеры проецируются.
Результаты:

- Полный пересчёт параметров объектива, вертикали, реконструкции и поиска: **{summary["fresh_pass_count"]}/{len(cases)}**.
- Линза и вертикаль зафиксированы, но реконструкция и поиск пересчитываются: **{fixed["summary"]["fresh_pass_count"]}/{len(cases)}**.
- Прежняя матрица и параметры камеры сохранены, проверен перенос по стабильной сцене: **{summary["frozen_pass_count"]}/{len(cases)}**.
- Все {summary["ambiguous_searches"]} новых полных поисков помечены собственным алгоритмом как неоднозначные.

Среди полных пересчётов медиану ≤10 px дали {median_only_passes} кадров; оба порога прошли {summary["fresh_pass_count"]}.
Это измерения на одной записи одной камеры, не оценка вероятности успеха на всех объектах.

## Что означает проверка сохранённой привязки

Регистрация строилась по SIFT-признакам изображения, без точек плана. Часть совпадений отложена для проверки;
исходные контрольные точки не использовались для оценки перемещения камеры.
Все {summary["accepted_camera_stability_probes"]}/{summary["camera_stability_probes"]} проверочных кадров по длительности записи прошли проверку малого движения.
Максимальное оценённое смещение углов изображения — {max(displacements):.3f} исходного пикселя; это оценка регистрации, не точное измерение механического движения.

При переносе старых контрольных точек этой регистрацией медианная ошибка замороженной привязки лежит в диапазоне
**{min(frozen_values):.4f}–{max(frozen_values):.4f} px**. Значение близко к прежним 7.4 px ожидаемо для неподвижной камеры.
**Это не десять новых независимых ручных разметок и не измерение ошибки положения ног людей.**
Окрестности контрольных точек дополнительно сравнивались по NCC, включая изменение видимости;
точки с плохим совпадением не удалялись из ошибки ради улучшения результата.

## Все новые кадры

Ошибки в пикселях схемы 612×658. Контрольные точки — прежние JSON, перенесённые регистрацией изображения.

| Момент от начала | Выборка | Сохранена вся привязка, медиана | Фиксирована камера, новый поиск, медиана | Полный пересчёт, медиана | Полный пересчёт, P95 |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

![Ошибки по времени](temporal-errors.png)

![Перемещения на схеме](location-jumps.png)

## Контрольные повторы

У GeoCalib есть случайная NMF-инициализация даже при оценке. Модели повторно используются ради экономии памяти/времени,
но для независимых вызовов восстанавливается состояние RNG после их инициализации с seed=42.
В исходном удачном пакетном прогоне камера 2 шла после камеры 1, поэтому состояние случайного этапа было другим.
В этом тесте порядок кадров не определяет начальную случайность каждого вызова.

| Повтор | Медиана, px | P95, px |
|---|---:|---:|
{chr(10).join(controls)}

При тех же закэшированных изображениях, масках и 3D-геометрии поиск воспроизводится для проверенных seed 42, 43, 44.
Полная новая оценка геометрии даже на прежнем медианном изображении может изменить выбранное место.
Фиксация параметров камеры восстанавливает контрольный результат на прежнем кадре,
но на новых кадрах сама по себе устойчивость поиска не обеспечивает.

## Где требуется доработка

Только для диагностики, с подсказкой контрольных JSON, подходящий вариант есть среди top-10 в
{summary["oracle_any_pass_count_diagnostic"]}/{len(cases)} полных пересчётов и
{fixed["summary"]["oracle_any_pass_count_diagnostic"]}/{len(cases)} пересчётов с фиксированной геометрией камеры.
Этот выбор не использовался в автоматическом результате. Проблема есть и в ранжировании, и в формировании кандидатов/геометрии.

Практический режим для этой неподвижной камеры — сохранять найденную и проверенную калибровку и следить за её смещением,
а не заново выбирать место на схеме на каждом кадре. Это хранение автоматически рассчитанной матрицы, а не ручное задание опорных точек.
Надёжная автоматическая первоначальная привязка ещё не решена: нужен отбор по серии кадров и дополнительным структурным признакам,
с явным отказом при неоднозначности. Простое голосование может закрепить одинаковую ошибочную комнату.

{consensus_section}
## Протокол и ограничения

- Время равномерных кадров задано долями длительности до оценки: {samples["uniform_fractions"]}.
- Стресс-кадры выбирались по изменению в фиксированной маске пола, без контрольных координат и результатов моделей.
  Первоначальное ограничение 15 секунд от равномерных кадров исключало короткие события;
  до нейросетевого прогона оно заменено запретом дубликата в пределах 1 секунды. Изменение сохранено в `samples-initial-selection.json`.
- Все 10 выбранных кадров учтены, включая ошибки и неоднозначные решения. Контроли не включены в знаменатель.
- Оптика, маски и структура плана оценивались без контрольных координат; JSON открывались отдельной оценкой после расчётов.
- Двери этой камеры по-прежнему не выделяются выше принятого порога сегментатора.
- Проверка относится к новым кадрам той же записи, не к другой комнате, камере, погоде или освещению.
- Исходный трекинг и его калибровки не изменены; автоматический пересчёт не подключён к нему.

## Файлы

`samples.json` — кадры, времена и хеши; `registration.json` — независимая проверка стабильности камеры;
`inference/protocol.json` — полный пересчёт; `fixed-camera-inference/protocol.json` — контроль с фиксированной камерой;
`evaluation.json`, `fixed-camera-evaluation.json` — все ошибки, включая диагностический top-10;
`inference/lens-seed-checks.json` — отдельные повторы оценки объектива;
`assessment.json` — итог непригодности автоматической инициализации;
`sample-contact.png` — все выбранные кадры.
Повторение описано в `TEMPORAL_VALIDATION.md` каталога эксперимента.
"""
    (root / "RESULTS.md").write_text(text, encoding="utf-8")
    print("Saved RESULTS.md, assessment.json, temporal-errors.png, location-jumps.png, sample-contact.png")


if __name__ == "__main__":
    main()
