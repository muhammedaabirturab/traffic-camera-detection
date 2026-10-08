# Architecture

```
React + Vite dashboard  ──HTTP/JSON──▶  FastAPI backend  ──▶  Ultralytics YOLO models
 (frontend/)                              (app/main.py)         COCO · rider · (helmet)
                                               │
                    ┌──────────────────────────┼───────────────────────────┐
                    ▼                          ▼                           ▼
             analysis/ (pipeline)       rules/ (rule engine)        storage/ (SQLite history)
```

## Pipeline

```
Image / Video
  → preprocessing        decode, validate, downscale (app/detection/preprocessing.py)
  → YOLO detection       COCO + rider (+ helmet) models (app/detection/yolo_detector.py)
  → object identification  canonical labels, duplicate filtering (association.dedupe)
  → tracking (video)     ByteTrack ids + IoU tracker for rider-model-only bikes (tracker.py)
  → relationship analysis  which person sits on which two-wheeler; helmet in head region (association.py)
  → rule engine          candidate violations -> rule-database records (violation_detector.py, rule_engine.py)
  → violation classification + confidence evaluation  (bands, min thresholds, "insufficient evidence")
  → evidence generation  annotated frame + crop (analysis/evidence_generator.py)
  → dashboard            overlays, cards, history, Traffic Intelligence Score
```

The three concerns are deliberately separate:

| Layer | Question it answers | Code |
|---|---|---|
| Object detection | *What is where?* | `yolo_detector.py` |
| Traffic-rule inference | *Do these objects stand in a relationship that matters?* | `association.py`, `violation_detector.py` |
| Violation classification | *Which configured rule, how confident, enough evidence?* | `rule_engine.py` + `traffic_rules.json` |

## Folder layout

```
app/
  main.py                FastAPI app + routes (create_app() is injectable for tests)
  config.py              all settings (env / .env)
  model_info.py          Model page data, reads real metrics file
  detection/             yolo_detector, association, violation_detector, tracker, preprocessing, types
  rules/                 traffic_rules.json (editable) + rule_engine.py
  analysis/              pipeline (one frame), image_analyzer, video_analyzer, jobs, summary, evidence_generator
  storage/history.py     SQLite history
  utils/                 visualization (OpenCV drawing), video_io (H.264 writer), helpers, logging
  models/                weights + README (see there)
frontend/                React dashboard (pages/, components/)
scripts/                 prepare_dataset, train, validate, predict, export_model
tests/                   rules, detection, API tests
docs/                    this folder
```

## Runtime data (git-ignored `data/`)

`data/media/<analysis-id>/` holds `original.jpg`, `annotated.jpg`, `evidence_N.jpg` (and `processed.mp4` for videos);
`data/trafficguard.db` holds the history (summary columns + the full JSON result).

## Video jobs

`POST /api/analyze/video` stores the upload, queues a background job (one at a time, because inference is the
bottleneck) and returns `{"status": "queued", "job_id": ...}`. The UI polls `GET /api/jobs/{id}` for progress and
then loads the finished result from `GET /api/history/{id}`.

## Error handling

Domain errors (`InvalidInputError`, `InferenceError`, missing models) are mapped to short JSON messages
(`{"status": "error", "code": ..., "message": ...}`); unexpected exceptions are logged server-side and returned as a
generic message, so stack traces never reach the UI. If the backend is down the UI shows how to start it.
