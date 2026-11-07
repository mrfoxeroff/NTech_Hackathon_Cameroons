# Внешние модели лаборатории

Веса и видеозаписи не добавляются в Git. Для примера `demo.py` они не требуются.
Полная обработка использует отдельное окружение из `requirements-motion.txt`.

YOLO Pose и SegFormer можно получить явной командой:

```powershell
./homography_lab/.venv/Scripts/python.exe homography_lab/prepare_motion_models.py --download
```

Скрипт закрепляет ревизию SegFormer и проверяет SHA-256 YOLO перед установкой файла.
Для полной офлайн-работы перенесите подготовленные кэши и три ReID-файла на целевой компьютер.
Скрипт расчёта сам не скачивает модели.

| Файл / модель | Происхождение / контроль |
|---|---|
| `yolo11n-pose.pt` | Официальный `ultralytics/assets`, выпуск `v8.3.0`; SHA-256 `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` |
| SegFormer B0 ADE20K | `nvidia/segformer-b0-finetuned-ade-512-512`, ревизия `489d5cd81a0b59fab9b7ea758d3548ebe99677da` |
| `osnet_x0_25_msmt17.pth` | Существующий ReID-ресурс проекта; SHA-256 `cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18` |
| `person-reidentification-retail-0288.xml` | Intel Open Model Zoo; SHA-256 `30f452b2ed2763f7bf429eea23f8f243ad0e7f3cdd7226db6888f1660efb3424` |
| `person-reidentification-retail-0288.bin` | Intel Open Model Zoo; SHA-256 `6496b8d51741703c9e9c7f6474cb96c60de1856d7b4fd77235e48cea146066a5` |

ReID-файлы передаются с ресурсами основного проекта, как описано в корневом README и `model-assets.json`.
Архитектура OSNet включена в `experiments/scenealigner/vendor_sources` с лицензией MIT.
OSNet и Intel используются только для признаков внешности; сопоставление и калибровка реализованы в лаборатории.

Параметры уже подготовленного офиса и автоматические предложения мест находятся в `assets/office`.
Это результаты прежнего автоматического расчёта, не замена весов и не ручная разметка опор.
Для другой сцены нужно получить новый банк автоматических гипотез.
