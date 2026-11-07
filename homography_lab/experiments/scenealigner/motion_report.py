"""Visual review of proposed motion calibration; never used for fitting or selection."""

import argparse
import html
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lens_geometry import project_calibration
from motion_pairs import balanced_matches


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.run
    frames = read(root / "frames/frames.json")
    observations = [
        json.loads(line)
        for line in (root / "motion-filtered/observations.jsonl").read_text().splitlines()
        if line
    ]
    pairs = read(root / "geometry-associated-final/pairs.json")
    joint = read(root / "joint-final/joint.json")
    bank = read(root / "baseline-candidates.json")
    cameras = list(frames["config"]["cameras"])
    density = {}
    for row in observations:
        if row["split"] == "train" and row["foot"] is not None:
            second = round(row["timestamp_sec"] * 2) / 2
            density[second] = density.get(second, 0) + 1
    timestamp = max(density, key=density.get) if density else frames["frames"][0]["timestamp_sec"]
    fig, axes = plt.subplots(1, len(cameras), figsize=(15, 5), constrained_layout=True)
    by_frame = {}
    for row in observations:
        by_frame.setdefault(row["frame_id"], []).append(row)
    for axis, camera in zip(np.atleast_1d(axes), cameras):
        frame = min(
            [f for f in frames["frames"] if f["camera"] == camera],
            key=lambda f: abs(f["timestamp_sec"] - timestamp),
        )
        image = cv2.imread(frame["image"])
        for row in by_frame.get(frame["id"], []):
            box = np.array(row["box"]).astype(int)
            color = (0, 210, 0) if row["foot"] is not None else (0, 130, 255)
            cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, 4)
            cv2.putText(
                image,
                "local " + str(row["local_id"]),
                tuple(box[:2]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
            if row["foot"] is not None:
                cv2.circle(image, tuple(np.rint(row["foot"]).astype(int)), 13, color, -1)
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.axis("off")
        axis.set_title(camera)
    fig.suptitle(
        f"Автоматические наблюдения, {timestamp:.2f} с по экранным часам\nЗелёный: предложена опора стоп; оранжевый: только тело. ID локальные.",
        fontsize=12,
    )
    fig.savefig(args.output / "observations-contact.png", dpi=140)
    plt.close(fig)
    primary = next(
        (e for e in pairs["edges"].values() if e.get("train_supported")), next(iter(pairs["edges"].values()))
    )
    ca, cb = primary["cameras"]
    holdout = balanced_matches(
        [
            r
            for r in pairs["pairs"]
            if r["cameras"] == [ca, cb]
            and r["split"] == "holdout"
            and all(v is not None for v in r["points"])
        ]
    )
    plan = cv2.cvtColor(cv2.imread(str(args.plan)), cv2.COLOR_BGR2RGB)
    before = {c: v["candidates"][0]["calibration"] for c, v in bank["cameras"].items() if v["candidates"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), constrained_layout=True)
    for axis, title, calibrations in [
        (axes[0], "Независимые автоматические гипотезы", before),
        (axes[1], "Совместная гипотеза: не принята", joint["calibrations"]),
    ]:
        axis.imshow(plan)
        if holdout:
            a, va = project_calibration(np.array([r["points"][0] for r in holdout]), calibrations[ca])
            b, vb = project_calibration(np.array([r["points"][1] for r in holdout]), calibrations[cb])
            for x, y, valid in zip(a, b, va & vb):
                if valid:
                    axis.plot([x[0], y[0]], [x[1], y[1]], c="black", alpha=0.4, lw=0.7)
            axis.scatter(a[va, 0], a[va, 1], c="#e33158", s=24, label=ca)
            axis.scatter(b[vb, 0], b[vb, 1], c="#0785c1", s=24, label=cb)
        axis.set_title(title, fontsize=11)
        axis.set_xlim(0, plan.shape[1])
        axis.set_ylim(plan.shape[0], 0)
        axis.axis("off")
        axis.legend(loc="lower left")
    fig.suptitle(
        f"{len(holdout)} пар из других эпизодов: проверка межкамерного согласования\nСближение точек не подтверждает правильность места на плане.",
        fontsize=12,
    )
    fig.savefig(args.output / "motion-before-after.png", dpi=150)
    plt.close(fig)
    if (root / "evaluation-final.json").exists():
        evaluation = read(root / "evaluation-final.json")
        current = evaluation["absolute_map"]
        active = [c for c in cameras if c in current]
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        xs = np.arange(len(active))
        width = 0.35
        axis.bar(
            xs - width / 2,
            [current[c]["baseline"]["median_error_px"] for c in active],
            width,
            label="Независимые гипотезы",
        )
        axis.bar(
            xs + width / 2,
            [current[c]["joint_proposal"]["median_error_px"] for c in active],
            width,
            label="Совместная гипотеза",
        )
        axis.set_xticks(xs, active)
        axis.set_ylabel("Медиана ошибки, пиксели схемы")
        axis.legend(fontsize=8)
        axis.set_title(
            "Отдельная оценка абсолютного места по старым JSON\nКонтрольные точки не участвовали в подгонке"
        )
        fig.savefig(args.output / "absolute-map-errors.png", dpi=150)
        plt.close(fig)
        pair_name = ca + "-" + cb
        metric = evaluation["motion"].get(pair_name, {})
        old = metric.get("independent_baseline", {})
        new = metric.get("joint_proposal", {})

        def number(value):
            return "—" if value is None else f"{value:.1f}"

        table = "".join(
            f"<tr><td>{html.escape(c)}</td><td>{number(current[c]['baseline']['median_error_px'])}</td>"
            f"<td>{number(current[c]['joint_proposal']['median_error_px'])}</td></tr>"
            for c in active
        )
        page = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Совместная привязка через движение</title><style>
body{{font:17px/1.55 system-ui,sans-serif;max-width:1120px;margin:36px auto;padding:0 22px;color:#17212d;background:#f8fafc}}
h1{{font-size:30px;line-height:1.2}}h2{{font-size:22px}}img{{width:100%;height:auto;background:white;border:1px solid #dbe2ea;border-radius:10px;margin:12px 0}}
.notice{{padding:18px 22px;background:#fff3cf;border-left:5px solid #bd8600;border-radius:8px}}table{{border-collapse:collapse;background:white;width:100%}}
td,th{{text-align:left;padding:10px 16px;border-bottom:1px solid #dde4ed}}.metric{{font-size:28px;font-weight:700}}small{{color:#526274}}
</style><h1>Совместная привязка камер через движение людей</h1>
<div class="notice"><b>Расчёт выполнен, гипотеза не принята для рабочего трекинга.</b><br>
Камеры удаётся согласовать между собой, но правильное место на плане остаётся неоднозначным. Дополнительная проверка относительной геометрии также не пройдена.</div>
<h2>Межкамерное согласование {html.escape(ca)} — {html.escape(cb)}</h2>
<div class="metric">{number(old.get("median_px"))} → {number(new.get("median_px"))} пикселя схемы</div>
<p>Медианное расхождение {new.get("pairs", 0)} автоматически предложенных пар из других эпизодов.
Это согласованность проекций, а не измеренная ошибка положения людей и не подтверждение их личностей.</p>
<img src="motion-before-after.png" alt="Сравнение независимой и совместной проекции">
<h2>Правильность места на плане — отдельная проверка</h2>
<table><tr><th>Камера</th><th>До, медиана ошибки</th><th>Совместная гипотеза</th></tr>{table}</table>
<p>Пиксели схемы по старым контрольным JSON. Для камеры 2 точность ухудшилась; приближение точек друг к другу не означает правильную привязку к помещению.</p>
<img src="absolute-map-errors.png" alt="Ошибки абсолютного места">
<h2>Что получила система из кадров</h2><img src="observations-contact.png" alt="Наблюдения тела и предложенные опоры стоп">
<p>В пилоте обработаны камеры 1–3. Для камеры 3 надёжной связи по движению недостаточно.
Записи 4–5 в этом прогоне не обрабатывались. Готовый трекинг и его ID не изменены.</p>
<small>Полный протокол: ../protocol.json. Численные результаты: ../evaluation-final.json. Контрольные JSON не передавались в расчёт привязки.</small></html>"""
        (args.output / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
