# Демо камер 1–2 и HTTP API

Подготовка сеанса запускается из корня репозитория в Windows PowerShell.
Используйте окружение проекта с PyTorch и CUDA, если доступна GPU.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[demo,dev]"
.\.venv\Scripts\python.exe -m tools.run_web_demo --start 188 --end 200 --output outputs_demo
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Откройте `http://127.0.0.1:8000`. Если готовый `outputs_demo/session` уже есть,
достаточно запустить сервер. Для другого каталога задайте перед запуском
`$env:NTECH_SESSION_DIR = 'outputs_web_verified/session'`.

Для подготовки нужны исходные `data/video/camera_1.mkv` и
`data/video/camera_2.mkv`, фон `data/samples/camera_1.jpg`,
`weights/person-office-v2.pt` и пара `person-reidentification-retail-0288.xml` /
`.bin` в `mtmc/weights`. Видео, фон и веса передаются отдельно от Git.
Контрольные суммы и источник модели ReID приведены в
[model-assets.json](../model-assets.json) и [каталоге Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0288/model.yml).
План `office.jpg` и калибровки `cam_1_calibration.json` и
`cam_2_calibration.json` входят в репозиторий.

`--start` и `--end` задают секунды на общей шкале экранных часов. Подготовщик
сверяет SHA-256 обоих исходников с аудитом Dima и применяет привязку по участкам.
По умолчанию демо обрабатывает каждый второй исходный кадр. `--stride 1` включает
все кадры. Для нового запуска укажите новый каталог `--output`.

## Откуда поступают данные

1. Локальный этап пишет `local/office_01/camera_N/frames.jsonl`, кропы и маски.
2. Адаптер Dima читает завершённые локальные результаты, вычисляет признаки ReID
   и пишет `global/observations.jsonl`, `assignments.jsonl` и `tracks.json`.
   Наблюдения без локального ID остаются в `unassigned.jsonl`.
3. Экспорт связывает назначения по `(camera_id, local_id, frame_index)`,
   проецирует точки опоры на план по гомографиям Fedor и кодирует MP4 с глобальными
   ID. Метки MP4 вычисляются из той же общей шкалы, что и траектории.
4. Сервер читает готовый каталог `session/` с `session.json`, планом и MP4.
   Интерфейс сам запрашивает данные по HTTP. Готовый каталог можно перенести на
   другой компьютер для показа.

Чтобы повторить только экспорт из готового результата Dima, выполните
`.\.venv\Scripts\python.exe -m tools.export_web_session outputs_demo/global outputs_reexport/session`.

## Контракт HTTP

| Запрос | Ответ |
|---|---|
| `GET /api/session` | JSON сеанса версии `1.0` |
| `GET /media/map.png` | План в пикселях калибровки |
| `GET /media/camera_1.mp4`, `GET /media/camera_2.mp4` | H.264 MP4 с поддержкой HTTP Range |
| `GET /docs` | Описание маршрутов FastAPI |

`start_time`, `end_time`, интервалы видимости и `map_path[].t` выражены в
секундах относительно `timeline_origin`. Значение `cameras[camera].start` задаёт
общий момент начала MP4, поэтому позиция плеера равна `t - camera.start`.
`coverage` указывает интервалы с видео. В пропуске слот скрывает кадр.

`tracks_summary` индексируется строковым глобальным ID. В каждом треке
`camera_segments` определяет видимость в момент времени, а `map_path` содержит
`t`, `x`, `y`, камеру и статус точки. `x` и `y` измерены в пикселях плана
от верхнего левого угла. Интерфейс рисует оценки и разрывы пунктиром.
Точки с неизвестной опорой или за границами плана пропускаются.

При отсутствии готового сеанса API возвращает 503, а страница показывает
команду подготовки. Веб-сервер выдаёт только файлы, перечисленные в сеансе.

## Проверка и область работы

На реальных записях 188–200 с обработаны 301 кадр двух камер и 872 подтверждённых
локальных наблюдения. Алгоритм сформировал четыре глобальных ID, каждый
появился в обеих камерах. Это результат алгоритма на фрагменте.
В браузере проверены загрузка плана, перемотка обоих MP4, воспроизведение
и выбор ID. Путь подготовки целиком также запущен на интервале 191–195 с.

Демо использует основной алгоритм Dima с Intel ReID. Двухкамерное уточнение
с резервным OSNet доступно в отдельном сценарии
[Dima](../mtmc/IDENTITY_REPAIR.md). Для подключения камер 3–5 потребуется
добавить их проверенные связи и привязку часов в подготовщик.
