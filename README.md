# Трекинг людей по камерам

Проект синхронизирует записи пяти камер офиса, присваивает людям сквозные ID и показывает их движение на плане. В плеере можно выбрать время, камеры или трек из журнала. План можно приблизить кнопками или колёсиком и перетащить мышью или пальцем.

## Открыть готовое демо

На Windows x64 распакуйте `NTech-Demo-Windows-x64.zip`, запустите `start-demo.bat` и откройте `http://127.0.0.1:8765` в Chrome или Edge. Архив с записями передаётся отдельно от репозитория. Python уже входит в архив, для просмотра модель и видеокарта не нужны.

Если у вас есть только готовая папка `session`, положите её в `outputs_all/session` и запустите плеер из корня проекта. Нужен Python 3.12 или новее.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install fastapi==0.162.0 uvicorn==0.54.0
.\.venv\Scripts\python.exe app.py
```

Откройте `http://127.0.0.1:8000`.

## Обработать исходные записи

Для обработки нужны файлы, которые передаются вместе с данными проекта.

| Что | Куда положить |
| --- | --- |
| Пять исходных записей | `data/video/camera_1.mkv` … `data/video/camera_5.mkv` |
| Эталонный кадр для фильтра игрового стола | `data/samples/camera_1.jpg` |
| Дообученные веса сегментации людей | `weights/person-office-v2.pt` |
| Веса для сопоставления людей между камерами | `mtmc/weights/person-reidentification-retail-0288.xml` и `.bin` |

План `office.jpg`, калибровки камер и привязка часов входят в репозиторий. Конфигурация [configs/cameras.yaml](configs/cameras.yaml) уже использует дообученную модель и фильтры сцены. Обучать модель для запуска не нужно. Контрольные суммы весов приведены в [model-assets.json](model-assets.json).

Команды ниже устанавливают обработчик и собирают новый сеанс из исходных видео. Для GPU сначала установите сборку PyTorch под свою видеокарту. Полный прогон удобнее выполнить заранее и перенести готовый сеанс на компьютер для показа.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[demo,dev]"
.\.venv\Scripts\python.exe -m tools.run_web_demo --output outputs_new
$env:NTECH_SESSION_DIR = 'outputs_new/session'
.\.venv\Scripts\python.exe app.py
```

Для короткой проверки добавьте `--cameras 1 2 --end 15` к команде обработки. Для каждого прогона выбирайте новый каталог `--output`.

## Где что находится

| Путь | Назначение |
| --- | --- |
| [src/local_tracking](src/local_tracking) | Сегментация, ByteTrack и фильтры мебели, экранов и зеркал |
| [mtmc](mtmc) | Сопоставление треков между камерами по времени, внешности и координатам |
| [tools/run_web_demo.py](tools/run_web_demo.py) | Обработка записей и сборка сеанса для плеера |
| [app.py](app.py), [viewer.js](viewer.js) | HTTP API и интерфейс просмотра |

[Формат сеанса и HTTP API](docs/web_demo.md) · [Данные локального трекера](docs/data_contract.md) · [Дообучение и проверка ложных срабатываний](docs/mirror_finetuning_report.md)

## Тесты

В окружении с установленным `.[demo,dev]` выполните:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -c mtmc/pytest.ini mtmc -q
```
