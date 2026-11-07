# NTech Hackathon — сквозной трекинг людей

Ветка `Dima`: локальная детекция и сегментация людей, трекинг внутри камеры,
ReID-признаки и собственное межкамерное сопоставление с глобальными ID.
Поддерживаются проверка часов записей, устранение локальных дублей и уточнение
сопоставлений при перекрытиях людей.

## Состав проекта

| Каталог | Назначение |
|---|---|
| `mtmc/` | Алгоритмы, команды обработки, конфигурации и тесты |
| `mtmc/local_tracking/` | Локальный трекер, адаптированный из ветки `misha` |
| `mtmc/integrations/` | Снимок исходного кода `misha`, тесты и сведения о происхождении |
| `mtmc/sync-check/` | Привязка часов наших двух записей и компактные отчёты |
| `calibration-camera-1/` | Инструменты и результаты разметки камеры и схемы |
| `office.jpg`, `office_camera.png` | Схема офиса без камер и с камерами |

Видео, веса нейросетей, виртуальное окружение, маски, кропы и промежуточные
выгрузки хранятся отдельно и исключены из Git. В репозитории находятся код,
конфигурации, разметка, тесты и документация. Исторические отчёты могут
ссылаться на локальные результаты, которые нужно получить повторным запуском.

## Установка

Проверено на Windows и Python 3.10. Обучение моделей не требуется.
Команды выполняются из корня репозитория:

```powershell
py -3.10 -m venv mtmc/.venv
& './mtmc/.venv/Scripts/python.exe' -m pip install -r requirements.txt
& './mtmc/.venv/Scripts/python.exe' -m pytest -c mtmc/pytest.ini mtmc -q
```

Версии зависимостей записаны из рабочего окружения. Чистая установка через
интернет отдельно не проверялась; вариант PyTorch CPU/CUDA выбирается под машину.

Создайте `mtmc/weights/` и поместите туда следующие файлы:

| Файл | Источник |
|---|---|
| `yolo11n-seg.pt` | [Ultralytics YOLO11](https://huggingface.co/Ultralytics/YOLO11/tree/main) |
| `person-reidentification-retail-0288.xml` и `.bin` | [Open Model Zoo, модель 0288](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-reidentification-retail-0288), вариант FP32 |
| `osnet_x0_25_msmt17.pth` | [OSNet, веса MSMT17](https://huggingface.co/kaiyangzhou/osnet/tree/main) |
| `torchreid_osnet.py` | [Архитектура Torchreid OSNet](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py), сохранить под указанным именем |

Контрольные суммы использованных локальных файлов: [model-assets.json](../model-assets.json).
Условия использования внешнего кода и моделей задаются их источниками.

## Запуск

Поместите `camera_new_1.mkv` и `camera_new_2.mkv` в корень репозитория.
Для этих конкретных файлов сохранены проверенные опоры экранных часов:
`mtmc/local_stage.synced.yaml` и `mtmc/sync-check/alignment.json`.
Для других видео нужно настроить камеры и измерить синхронизацию заново.

Полный путь для четырёх демонстрационных эпизодов:

```powershell
& './mtmc/.venv/Scripts/python.exe' mtmc/process_synced_demo.py `
  --alignment mtmc/sync-check/alignment.json --output mtmc/demo-run

& './mtmc/.venv/Scripts/python.exe' mtmc/render_synced.py `
  --alignment mtmc/sync-check/alignment.json `
  --demo mtmc/demo-run/demo.json --output mtmc/demo-run/result.mp4
```

Для произвольной записи: настроить YAML по образцу
`mtmc/local_stage.example.yaml`, запустить `mtmc/local_stage.py process`, затем
передать локальные результаты в `mtmc/import_local_tracking.py --associate`.
Проверенный пример и описание форматов: [интеграция локального этапа](INTEGRATION_MISHA.md).
Код основного межкамерного алгоритма: [associate.py](associate.py).
Геометрическое уточнение двух камер и резервный OSNet:
[refine_cross_camera.py](refine_cross_camera.py).

Результат содержит `assignments.jsonl`, `tracks.json` и `tracks.csv`.
Глобальный ID назначается наблюдению по ключу камера / локальный ID / кадр.
Неподтверждённые наблюдения после уточнения имеют `global_id: null` и отображаются
серым `ID ?`. Подтверждённый выход из обзора всех камер завершает трек.

## Проверенные результаты и ограничения

139 автоматических тестов. На четырёх контрольных эпизодах получены
4 / 2 / 1 / 1 трек вместо прежних 14 / 2 / 1 / 2. Все 1665 входных наблюдений
сохранены, 7 оставлены без подтверждённого ID. Повторное уточнение даёт одинаковые
назначения ID. Это 83 секунды выбранных сцен, а не независимая оценка всей записи.

Геометрическое уточнение сейчас рассчитано на две неподвижные камеры;
соответствие верхних точек рамок не является калибровкой пола.
Синхронизация основана на экранных часах, аппаратная кадровая точность не подтверждена.
На полных записях последнее исправление ID ещё не проверено.

Подробности: [описание алгоритма](README.md),
[исправление ID](IDENTITY_REPAIR.md),
[проверка синхронизации](sync-check/README.md).
