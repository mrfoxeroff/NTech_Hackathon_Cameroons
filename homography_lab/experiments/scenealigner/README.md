# Автоматическая привязка камер к плану: отдельный эксперимент

## Совместная привязка через движение — реализованный MVP

Добавлен самостоятельный пайплайн: позы/видимые стопы, локальные треки тел, признаки внешности,
разрешение неоднозначных соответствий по относительной геометрии, перенос гипотез и совместный выбор.
Пилот 1–3: 1876 кадров. Для 1–2 межкамерное расхождение восьми предложенных пар других эпизодов
уменьшилось с 176.8 до 4.7 px схемы, но абсолютная привязка и строгая относительная проверка не подтверждены.
Все компоненты остаются с отказом; рабочий трекинг не менялся.

Просмотр: `runs/motion-joint-final/review-final/index.html`.
Результаты: `runs/motion-joint-final/RESULTS.md`. Запуск: `MOTION_CALIBRATION.md`.
71 тест эксперимента и 22 базовых теста проходят.

## Новый поиск места: физическая проекция, двери, проверка обзора

Добавлены проекция плоскости пола без нейросетевой глубины, агрегирование пяти кадров,
автоматические опоры дверей Grounding DINO, контроль масштаба по ширине двери,
проверка пересечения лучей со стенами и отказ от экспорта при неоднозначном месте.
Пилот проверен на камере 2, включая пять дополнительных кадров. **Выигрыша в точности выбора комнаты нет**:
лучшие гипотезы неправильные, четыре режима возвращают отказ. Исходный трекинг не менялся.
51 тест эксперимента проходит.

Результаты: `runs/analytic-place-v1/RESULTS.md`. Запуск и новый вектор развития:
`PLACE_SELECTION.md` — совместный выбор положений камер через надёжные наблюдения стоп и переходы между помещениями.

## Надёжность на других кадрах

Проверены 10 новых моментов камеры 2 и 60 кадров на стабильность её положения.
Автоматический повторный поиск прошёл критерии медиана ≤10 px/P95 ≤20 px только на **1/10** кадров;
фиксация линзы/вертикали при новом поиске тоже дала **1/10**.
Сохранённая целиком привязка не показывает дрейфа по проверке регистрации неподвижной сцены.
Это не новая разметка положения людей. Выигрыш 7.4 px на исходном кадре не означает надёжную автоматическую инициализацию.
Результаты: `runs/camera-2-temporal/RESULTS.md`. Протокол и повторение: `TEMPORAL_VALIDATION.md`.
Дополнительное согласование гипотез по кадрам также не дало правильной привязки: поддержка 9/10 закрепила ошибку около 58 px.
Проверен 41 тест эксперимента.

## Последняя проверка: искажения, стены, двери

Проверены четыре режима на всех пяти камерах. Сочетание коррекции и структурного поиска дало для камеры 2
**495.7 → 7.4 px по медиане, P95 19.7 px** по существующим контрольным JSON, без подгонки по этим точкам.
Для остальных камер устойчивого выигрыша нет; готовый трекинг не изменялся.
Полный отчёт: `runs/lens-structure-study/RESULTS.md`. Повторяемый запуск и файлы: `LENS_STRUCTURE.md`.
Проверены 29 тестов эксперимента и согласованность зависимостей.

## Обновление: группы и поиск фрагментов

Новый прогон: `runs/office-grouped-fragments/RESULTS-final.md`.
Подтверждено перекрытие 1–3; группы для раздельной реконструкции: `[cam_1, cam_3]`, `[cam_2]`, `[cam_4]`, `[cam_5]`.
Камера 2 визуально видит ту же игровую зону, но автоматическая геометрическая проверка не подтвердила её связь.
Одиночная группа означает отсутствие надёжного ребра, а не доказанное отсутствие перекрытия.

Добавлены сопоставление LoFTR по увеличенным участкам, проверка SIFT на исходном разрешении и поиск по 106 фрагментам схемы,
четырём поворотам, нескольким возможным соответствиям каждой детали. Вырожденные преобразования отклоняются;
разные положения, масштабы и повороты сохраняются как отдельные гипотезы. В поиске нет координат из контрольных JSON.

Медианные ошибки: камера 1 — **183.4 → 41.5 px**, камера 3 — **115.2 → 92.2 px**;
камера 2 — **167.3 → 495.7 px**, камера 5 — **170.6 → 308.7 px**; камера 4 — отказ от привязки.
Это частичное улучшение, общего улучшения точности на всех пяти камерах нет. Нельзя подключать этот результат к трекингу.
Маленькая разница между оценками кандидатов показывает неоднозначность, большая разница не гарантирует правильность.

Основные результаты:

- `runs/office-grouped-fragments/RESULTS-final.md` — сравнение с исходным прогоном;
- `runs/office-grouped-fragments/before-after-final.png` — контрольные точки и автоматические проекции;
- `runs/office-grouped-fragments/fragment-candidates-final.png` — разные гипотезы фрагментов;
- `runs/office-grouped-fragments/fragment-search-final/` — полный поиск, сырые кандидаты и матрицы;
- `runs/office-five-overlap-verified.json` — граф с геометрическими доказательствами и неподтверждёнными рёбрами.

Сначала задайте PowerShell-переменные из раздела ниже. Веса LoFTR уже сохранены в `cache`.
После этого новый полный прогон можно повторить так (используйте новые имена файлов/каталогов):

```powershell
& $scenePython "$experimentDir\neural_overlap.py" `
  --images "$experimentDir\inputs\office-five\views" `
  --weights "$experimentDir\cache\loftr_outdoor.ckpt" `
  --variant outdoor --confidence .2 --tiles --augment-sift `
  --output "$experimentDir\runs\overlap-repeat"

& $scenePython "$experimentDir\verify_overlap.py" `
  --images "$experimentDir\inputs\office-five\views" `
  --network-report "$experimentDir\runs\overlap-repeat\overlap.json" `
  --output "$experimentDir\runs\overlap-verified-repeat.json"

& $scenePython "$experimentDir\grouped_experiment.py" `
  --images "$experimentDir\inputs\office-five\views" `
  --overlap "$experimentDir\runs\overlap-verified-repeat.json" `
  --plan "$experimentDir\inputs\office-five\plan.jpg" `
  --floor-masks "$experimentDir\inputs\office-five\floor" `
  --matcher topk --output "$experimentDir\runs\grouped-repeat"

& $scenePython "$experimentDir\evaluate.py" `
  --predictions "$experimentDir\runs\grouped-repeat\fragment-search\predictions" `
  --reference "$experimentDir\reference" `
  --output "$experimentDir\runs\grouped-repeat\evaluation.json"

& $scenePython "$experimentDir\grouped_report.py" `
  --run "$experimentDir\runs\grouped-repeat" --reference "$experimentDir\reference"
```

Для повторного поиска на уже вычисленных групповых реконструкциях достаточно запустить `fragment_search.py`
с `--groups "$experimentDir\runs\office-grouped-fragments"`, входным планом/масками, новым `--output` и `--matcher topk`.
Новый каталог поиска можно передать генератору отчёта через `--search-dir`, файл оценки — через `--evaluation`, имя отчёта — через `--suffix`.

Официальные веса LoFTR загружены с адресов, указанных в установленном Kornia:
`https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt` и
`https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt`.
SHA256 indoor: `be9ff88b323ec27889114719f668ae41aff7034b56a4c4acbd46b8b180b87ed3`;
outdoor: `21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec`.
Веса загружаются через `weights_only=True`, без изменения пакета Kornia.
В этом обновлении прошли **13 тестов**: в том числе на ложные совпадения, коллинеарность, сохранение отдельных камер,
переход между координатами фрагмента и исходной схемы, сохранение разных поворотов и явный учёт отсутствующих предсказаний.

Ниже сохранено описание исходного эксперимента и окружения.

**Текущий результат: нейросети запустились на всех пяти камерах, но автоматическая привязка не прошла сравнение с существующими точками.**
Результат и иллюстрация находятся в `RESULTS.md` и `comparison-five.png`. Проверка пригодности — `assessment.json`;
для текущего прогона `automatic_calibrations_safe_to_integrate: false`.

Исходный трекинг и checkout `NTech_Hackathon_Cameroons` не изменяются. Скрипты не публикуют результаты в GitHub.
Это эксперимент с фоновыми изображениями, а не повторная обработка всех кадров видео.

## Как устроен запуск

1. `prepare_inputs.py backgrounds` извлекает семь кадров из каждой записи, проверяет неподвижность камеры и строит медианный фон.
   Ранее подготовленные кадры переиспользуются только при совпадении хешей видео и изображения с исходной сессией.
2. `prepare_inputs.py masks` запускает SegFormer, выделяет пол с вероятностью ≥0.6.
3. `run_experiment.py` последовательно запускает GeoCalib → Pi3 → SceneAligner, освобождая GPU между моделями.
   Предсказанные 3D-точки пола переводятся в план и используются для оценки гомографий.
4. `evaluate.py` сравнивает результат с JSON, полученными из GitHub. Эти точки не подаются моделям и не участвуют в подгонке автоматических гомографий.
5. `diagnose.py` — отдельная диагностика **с подгонкой по контрольным JSON**. Она помогает разделить ошибки реконструкции и привязки к плану.
   Её численные результаты нельзя выдавать за точность полностью автоматического решения.
6. `check_overlap.py` ищет геометрически согласованные SIFT-совпадения между камерами. Это предварительная диагностика перекрытия, не готовый модуль объединения реконструкций.

Точки на стенах, мебели и столешнице нельзя использовать как точки пола. Положение человека следует проецировать по контакту ног с полом.
Сеть может создать согласованную, но неверно расположенную привязку: маленькая ошибка на её собственных предсказаниях этого не обнаруживает.

## Повторить в текущем окружении

PowerShell. Все необходимые веса уже скачаны. Подготовка фонов использует прежнее окружение с PyAV;
модели работают в отдельном окружении Python 3.12 с CUDA. Существующее окружение трекинга не обновлялось.

```powershell
Set-Location -LiteralPath "D:\AI's_directory"
$experimentDir = "D:\AI's_directory\homography_lab\experiments\scenealigner"
$scenePython = "D:\AI's_directory\homography_lab\.venv-scene\Scripts\python.exe"
$videoPython = "D:\AI's_directory\mtmc\.venv\Scripts\python.exe"
```

Существующие подготовленные изображения можно использовать повторно:

```powershell
& $scenePython "$experimentDir\run_experiment.py" `
  --images "$experimentDir\inputs\office-five\views" `
  --plan "$experimentDir\inputs\office-five\plan.jpg" `
  --floor-masks "$experimentDir\inputs\office-five\floor" `
  --output "$experimentDir\runs\office-five-repeat"

& $scenePython "$experimentDir\evaluate.py" `
  --predictions "$experimentDir\runs\office-five-repeat" `
  --reference "$experimentDir\reference" `
  --output "$experimentDir\office-five-repeat-evaluation.json"
```

Для повторной подготовки фоновых кадров и масок:

```powershell
& $videoPython "$experimentDir\prepare_inputs.py" backgrounds `
  --output "$experimentDir\inputs\office-five-repeat"
& $scenePython "$experimentDir\prepare_inputs.py" masks `
  --output "$experimentDir\inputs\office-five-repeat"
```

Затем передайте новый каталог `views`, `floor` и его `plan.jpg` в запуск моделей.
Для нового эксперимента используйте новое имя выходного каталога: скрипты отказываются перезаписывать прогоны.
Простое повторение на тех же кадрах не даёт независимой проверки обобщения.

Отдельная диагностика и проверки:

```powershell
& $scenePython "$experimentDir\diagnose.py" `
  --run "$experimentDir\runs\office-five-repeat" `
  --reference "$experimentDir\reference" `
  --output "$experimentDir\office-five-repeat-diagnostic.json"

& $scenePython "$experimentDir\check_overlap.py" `
  --images "$experimentDir\inputs\office-five\views" `
  --output "$experimentDir\runs\office-five-overlap-repeat"

& $scenePython -m pytest -q -c "$experimentDir\pytest.ini" $experimentDir
& $scenePython -m pip check
```

Сборка отчёта по сохранённым именованным прогонам:

```powershell
& $scenePython "$experimentDir\make_report.py"
```

Эта команда пересоздаёт только отчёт, рисунок и итоговую оценку, не матрицы и не контрольные JSON.
Она читает исходные именованные прогоны `office-pair-518` и `office-five-518`; для нового сравнения их пути в генераторе нужно явно обновить.

## Что сохранено

- `reference/`: пять исходных калибровочных JSON из `main`, происхождение и хеши.
- `inputs/office-five/`: фоновые изображения, маски, план, хеши исходных видео, времена выбранных кадров 3–5.
  Данные фоновых кадров 1–2 также сохранены в исходной сессии `homography_lab/runs/office-01/session.json`.
- `runs/office-five-518/`: реконструкция, вид плотности сверху, совмещение со схемой, позиции камер и пять кандидатных гомографий.
- `scenealigner-five-reference-evaluation.json`: ошибки на каждом контрольном соответствии, включая исключённые точки.
- `geometry-five-diagnostic.json`: диагностика с явно указанным использованием контрольных точек при подгонке.
- `runs/office-five-overlap/overlap.json`: геометрически проверенные совпадения между камерами.
- `assessment.json`: итог «не подключать к трекингу» для текущего результата.
- `requirements-lock.txt`: точные версии установленных зависимостей.

`status: complete` в манифесте означает, что вычисления завершились, а `automatic_candidate` означает неподтверждённую гипотезу.
Оба поля не означают, что привязка правильная. Оценка по имеющимся JSON также не заменяет проверку на независимом помещении.

## Источники и версии

Использованы официальные реализации:

| Компонент | Исходный код | Зафиксированный коммит |
|---|---|---|
| SceneAligner | https://github.com/Cornell-VAILab/SceneAligner | `65f91bec1202ada2806a80f7f67cb18ee52c9012` |
| Pi3 | https://github.com/yyfz/Pi3 | `6acdeb033d337e90b9df7a77b0c8929231083f16` |
| GeoCalib | https://github.com/cvg/GeoCalib | `3ee7ec3411160fea5f3ac32377e53137e9119824` |
| DINOv3 | https://github.com/facebookresearch/dinov3 | `83318a8767dc346d5e11a70521860e0da2e1099b` |

Ревизии файлов моделей записаны в `model-files.json` в путях `snapshots/<revision>`.
GeoCalib использует официальный `v1.0/geocalib-pinhole.tar`; SegFormer —
`nvidia/segformer-b0-finetuned-ade-512-512`, ревизия `489d5cd81a0b59fab9b7ea758d3548ebe99677da`.
Веса и внешние репозитории не менялись. В нашем запуске Pi3 используется экономная загрузка и mixed precision для RTX 2060;
численная эквивалентность с авторским запуском на другой GPU отдельно не измерялась.
Лицензии компонентов описаны авторами в `homography_lab/vendor/SceneAligner/THIRD_PARTY.md`;
у весов Pi3 есть ограничение на некоммерческое использование.

## Ограничения

Нет гарантии совпадения по всей комнате, поправки на искажение объектива, контроля физического масштаба,
надёжного объединения ракурсов без перекрытия и обученной проверки семантики найденных соответствий.
Текущие скрипты не являются готовой заменой существующей калибровки. Направления доработки и фактические измерения — в `RESULTS.md`.

## Выполненные проверки

Четыре теста эксперимента прошли: оценка не меняет матрицу, исключения JSON учитываются явно,
несовпадающие размеры отклоняются, гомография восстанавливается из синтетических 3D-точек без контрольного JSON.
22 теста существующего `homography_lab` также прошли. `pip check` нового окружения не обнаружил конфликтов.
Проверены хеши пяти JSON и схемы, происхождение повторно использованных кадров 1–2,
пять выходных калибровок и отсутствие изменений внешнего кода. Коммиты и состояние внешних реализаций — в `source-versions.json`.
Эти проверки подтверждают корректность запуска и измерения; они не подтверждают пригодность нейросетевой привязки.
