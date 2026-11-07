# Надёжность камеры 2 на новых кадрах

**Повторный автоматический поиск ненадёжен: критерии точности прошёл 1 из 10 новых кадров.**
Фиксация линзы и вертикали при новом поиске также дала 1/10. Сохранённая целиком привязка не показывает дрейфа:
по 60 проверочным кадрам камера стабильна, а перенос старых контрольных точек даёт прежнюю медиану около 7.38 px.
Это не новая ручная разметка положения людей и не подтверждение качества на другом помещении.

Результаты: `runs/camera-2-temporal/RESULTS.md`.
Графики: `temporal-errors.png`, `location-jumps.png`; все кадры — `sample-contact.png` в том же каталоге.

## Что проверено

- 8 времён заданы равномерной сеткой до сравнения с контролем: примерно 01:00, 03:15, 05:30, 07:45, 10:00, 12:15, 15:00, 18:30.
- Ещё 2 кадра, 03:10 и 10:10, выбраны по наибольшим изменениям в зоне пола на скане через каждые 20 секунд.
  При выборе не использовались контрольные координаты или результаты привязки.
- Старая матрица проверяется при переносе контрольных точек независимо оценённой регистрацией сцены.
  Регистрация SIFT/RANSAC использует отложенную часть совпадений; контрольные точки не участвуют в подгонке перемещения камеры.
- Заново рассчитаны сегментация, линза, вертикаль, Pi3 и сопоставление с планом на всех 10 кадрах.
- Отдельно на тех же 10 изображениях сохранены линза и вертикаль, но Pi3 и поиск рассчитаны заново.
- Проверены исходный медианный кадр и повторный поиск с полностью закэшированными входами при seed 42, 43, 44.
- Все случаи, отказы и ошибки входят в итог. Контрольные повторы исключены из знаменателя новых кадров.
  Выбор лучшего из top-10 с подсказкой JSON сохранён только как диагностика, не подменяет автоматический top-1.

Критерий по прежним контрольным JSON: медиана ≤10 px, P95 ≤20 px, все исходные инлайеры проецируются.
Для полного пересчёта только 18:30 прошёл оба порога. На 10:00 медиана 7.40 px, но P95 21.28 px — критерий не пройден.
Все 10 полных новых поисков помечены как неоднозначные.

## Интерпретация

При неподвижной камере следует хранить найденную и проверенную матрицу вместе с параметрами линзы и проверять смещение сцены.
Это не ручное задание опорных точек. Ненадёжна пока автоматическая первоначальная привязка и её пересчёт по отдельным кадрам.
Фиксации одних внутренних параметров недостаточно: нестабильны также реконструкция/извлечение признаков и выбор фрагмента схемы.

Дополнительно реализовано согласование top-10 гипотез между кадрами. Пробы автоматически выбраны из маски пола,
учитывается не более одного голоса от кадра, итог выбирается медоидом без усреднения гомографий.
На основной настройке поддержка 9 из 10 кадров дала неверную привязку с медианой 57.8 px.
Проверены два поднабора по 5 кадров и три заранее заданных радиуса; ни один из 18 вариантов на этой выборке не прошёл критерии точности.
Результаты находятся в `runs/camera-2-temporal/consensus-fixed-camera-v2/` и `consensus-full-v2/`.
Это продолжение проверки, а не исправленная автоматическая инициализация.

В GeoCalib найдена случайная NMF-инициализация в режиме оценки. При повторном использовании модели для каждого изображения
восстанавливалось её состояние генераторов после инициализации с seed=42 — как для независимого вызова модели.
Исходный успешный пакетный расчёт камеры 2 использовал другое состояние после обработки камеры 1.
Поиск на неизменных закэшированных данных воспроизвёл результат 7.3815 px для всех трёх проверенных seed.
Поэтому проблема не объясняется только случайностью самого финального поиска.

## Повторение

Нужны существующие веса и окружения на этом компьютере. Результат пишется в новый каталог.
Первый этап использует PyAV из старого окружения; остальные — отдельное CUDA-окружение.

```powershell
Set-Location -LiteralPath "D:\AI's_directory"
$experimentDir = "D:\AI's_directory\homography_lab\experiments\scenealigner"
$scenePython = "D:\AI's_directory\homography_lab\.venv-scene\Scripts\python.exe"
$videoPython = "D:\AI's_directory\mtmc\.venv\Scripts\python.exe"
$temporalRun = "$experimentDir\runs\camera-2-temporal-repeat"
$frozenCamera = "$experimentDir\runs\office-divisional-groups\search-lens-structure-v2\predictions\cam_2-calibration.json"

& $videoPython "$experimentDir\temporal_frames.py" `
  --video "D:\AI's_directory\camera_new_2.mkv" `
  --background "$experimentDir\inputs\office-five\views\cam_2.jpg" `
  --floor-mask "$experimentDir\inputs\office-five\floor\cam_2.png" --output $temporalRun

& $scenePython "$experimentDir\temporal_registration.py" `
  --samples "$temporalRun\samples.json" `
  --background "$experimentDir\inputs\office-five\views\cam_2.jpg" `
  --output "$temporalRun\registration.json"

& $scenePython "$experimentDir\temporal_inference.py" `
  --samples "$temporalRun\samples.json" --plan "$experimentDir\inputs\office-five\plan.jpg" `
  --plan-structure "$experimentDir\inputs\office-five-structural-v2" `
  --control-image "$experimentDir\inputs\office-five\views\cam_2.jpg" `
  --cached-reconstruction "$experimentDir\runs\office-divisional-groups\group_02\reconstruction" `
  --cached-inputs "$experimentDir\inputs\office-five-divisional" `
  --cached-semantics "$experimentDir\inputs\office-five-structural-v2\divisional" `
  --output "$temporalRun\inference"

& $scenePython "$experimentDir\temporal_fixed_geometry.py" `
  --previous-inference "$temporalRun\inference" `
  --cached-inputs "$experimentDir\inputs\office-five-divisional" `
  --cached-reconstruction "$experimentDir\runs\office-divisional-groups\group_02\reconstruction" `
  --plan "$experimentDir\inputs\office-five\plan.jpg" `
  --plan-structure "$experimentDir\inputs\office-five-structural-v2" `
  --output "$temporalRun\fixed-camera-inference"

foreach ($temporalMode in @("inference", "fixed-camera-inference")) {
  $evaluationName = if ($temporalMode -eq "inference") { "evaluation.json" } else { "fixed-camera-evaluation.json" }
  & $scenePython "$experimentDir\temporal_evaluate.py" `
    --inference "$temporalRun\$temporalMode" --samples "$temporalRun\samples.json" `
    --registration "$temporalRun\registration.json" `
    --reference "$experimentDir\reference\cam_2_calibration.json" `
    --frozen-calibration $frozenCamera `
    --background "$experimentDir\inputs\office-five\views\cam_2.jpg" `
    --output "$temporalRun\$evaluationName"
}

& $scenePython "$experimentDir\temporal_report.py" --root $temporalRun `
  --plan "$experimentDir\inputs\office-five\plan.jpg" `
  --floor-mask "$experimentDir\inputs\office-five\floor\cam_2.png" `
  --frozen-calibration $frozenCamera
```

Чтобы перестроить только существующий отчёт, передайте в последнюю команду
`--root "$experimentDir\runs\camera-2-temporal"`.
Модели при этом не запускаются, результаты расчёта не меняются.

Для повторения согласования гипотез выполните для каждого режима (например, `fixed-camera-inference`):

```powershell
& $scenePython "$experimentDir\temporal_consensus.py" `
  --inference "$temporalRun\fixed-camera-inference" --registration "$temporalRun\registration.json" `
  --floor-mask "$experimentDir\inputs\office-five\floor\cam_2.png" `
  --output "$temporalRun\consensus-fixed-camera-v2"

& $scenePython "$experimentDir\consensus_evaluate.py" `
  --consensus "$temporalRun\consensus-fixed-camera-v2\consensus.json" `
  --reference "$experimentDir\reference\cam_2_calibration.json" `
  --output "$temporalRun\consensus-fixed-camera-v2\evaluation.json"
```

Для полного пересчёта замените `fixed-camera-inference` на `inference`, а выходной каталог — на `consensus-full-v2`.
После обоих расчётов перестройте отчёт последней командой из основного блока.

## Проверки и ограничения

Прошёл **41 тест** эксперимента, включая новые проверки малого смещения камеры, отказа при движении,
сохранности контрольных координат, учёта неудач/контролей и недопустимости выдавать результат top-10 с подсказкой за автоматический.
Дополнительно проверены один голос от кадра, границы кластеров и выбор проб из маски пола без опорных координат.
Контрольный повтор с фиксированной камерой воспроизводит исходный результат с погрешностью намного меньше 0.001 px.
Хеш исходного видео совпадает с видео прежнего эксперимента; контрольные JSON не изменены.

Проверка не измеряет истинные положения людей: новая ручная/инструментальная разметка отсутствует.
Коррелированные кадры одной записи не являются проверкой обобщения на другие объекты.
Исходный трекинг, ветка репозитория и его рабочие калибровки не менялись.
