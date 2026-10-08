# TrafficGuard AI

**Intelligent Traffic Violation Detection System** — YOLO-based analysis of Indian traffic-camera images and video.

TrafficGuard AI detects vehicles and riders with YOLO, works out which people are sitting on which two-wheeler,
applies a configurable rule database, and reports **AI-detected possible violations** with a confidence score, an
evidence image and a plain-language explanation. Every result is labelled *"Requires human verification"*.

> **Educational project.** It is not an enforcement tool, its output is not legal proof, and the legal references
> in `app/rules/traffic_rules.json` must be verified against current official sources.

<!-- Screenshots: add your own after running the app (see "Screenshots" below). -->

---

## Features

| Capability | Image | Video | Notes |
|---|:-:|:-:|---|
| Vehicle detection & counting (motorcycle, car, bus, truck, bicycle) | ✅ | ✅ unique via tracking | COCO-pretrained YOLOv8n |
| Rider ↔ two-wheeler association | ✅ | ✅ | geometric relationship analysis (`docs/methodology.md`) |
| **More than two persons on a two-wheeler** | ✅ | ✅ temporally validated | works with the shipped models |
| **Rider / pillion without helmet** | ✅ | ✅ temporally validated | helmet detector trained on the EdgeVision dataset + head-region search; if the helmet model file is removed the UI reports *"helmet not assessed"* instead of guessing |
| **Stop line crossed on red** | ❌ by design | ✅ experimental | model cannot read signals; operator supplies stop line + red window |
| Lane violations, number-plate OCR | — | — | not implemented (listed as such in the rule database) |
| Insufficient-evidence state, confidence bands, duplicate filtering | ✅ | ✅ | all thresholds configurable |
| Evidence images, annotated overlay, processed video (H.264) | ✅ | ✅ | |
| Traffic Intelligence Score + violations-over-time chart | ✅ | ✅ | analytical summary, not a legal judgment |
| Detection history (SQLite), model page with real metrics, rule database page | ✅ | ✅ | |

## Architecture

```
React + Vite dashboard ──▶ FastAPI ──▶ preprocessing → YOLO (COCO + rider [+ helmet]) → tracking (video)
                                      → relationship analysis → rule engine (traffic_rules.json)
                                      → violation classification + confidence → evidence → history (SQLite)
```

Details: [docs/architecture.md](docs/architecture.md) · logic: [docs/methodology.md](docs/methodology.md) · model: [docs/model.md](docs/model.md)

## Tech stack

Python 3.11 · Ultralytics YOLOv8 (PyTorch) · ByteTrack · OpenCV · FastAPI · SQLite · React 18 + Vite · pytest

## Dataset

The reference dataset (see the Kaggle notebook it comes from, *motorbike-and-helmet-yolo-detection*) was inspected rather than assumed:

* 795 images, YOLO txt labels, **one class: `person_bike`** (a person on a two-wheeler), **no helmet class**.
* About half the files are horizontally flipped copies; the split script keeps a flip in the same split as its original (no leakage).
* Split used: train 594 / val 120 / test 80 images (`dataset_stats.json`).

The rider dataset has no helmet labels, so a second dataset trains the helmet detector:

* **EdgeVision Dataset** (Gajjar et al., Mendeley Data, DOI 10.17632/j82bnw7gsr.1, **CC BY 4.0**): 2,392 images, classes `BikeWithRider`, `NoHelmet`, `Helmet`.
  Split used: train 1794 / val 359 / test 239 images, downscaled to 800 px on the long side.

Datasets are **not** in the repository. Setup: [datasets/README.md](datasets/README.md).

## Model

* `yolov8n.pt` (COCO) for vehicles and persons.
* `rider_detector.pt`: YOLOv8n fine-tuned on `person_bike` (committed, ~6 MB).
* `helmet_detector.pt`: YOLOv8n fine-tuned on EdgeVision (committed, ~6 MB) - classes `bike_with_rider`, `no_helmet`, `helmet`. Trained here from the official `yolov8n.pt`; no third-party weights are used.

## Installation

Requirements: Python 3.10+, Node 18+, Git. A GPU is optional (CUDA is used automatically if available).

```bash
git clone https://github.com/muhammedaabirturab/traffic-camera-detection.git
cd traffic-camera-detection

# Backend
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
# PyTorch: pick ONE (CPU build is smaller; CUDA build is faster for training)
pip install torch torchvision                                                     # CPU / default
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # NVIDIA GPU
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

The first run downloads the small official `yolov8n.pt` (≈6 MB) into `app/models/` if it is not there yet.

### Environment

`cp .env.example .env` (optional). Every setting has a default; the main ones are `MODEL_PATH`, `CONFIDENCE_THRESHOLD`,
`IOU_THRESHOLD`, `DEVICE` (`auto`/`cpu`/`0`), `VIDEO_FRAME_SKIP`, `DATABASE_URL`.

## Running

Backend (http://127.0.0.1:8000, API docs at `/docs`):

```bash
python -m app.main
```

Frontend (dev server, http://localhost:5173):

```bash
cd frontend
npm run dev
```

Alternatively `npm run build` once; the backend then serves the dashboard itself at http://127.0.0.1:8000.
If port 8000 is taken, set `PORT=8001` and, for the dev server, change the proxy target in `frontend/vite.config.js`.

## Training

```bash
python scripts/prepare_dataset.py --source <folder with images + .txt labels>
python scripts/train.py --epochs 50 --batch 8 --imgsz 640
python scripts/validate.py --split test          # re-run the held-out evaluation
python scripts/export_model.py --format onnx     # optional
python scripts/predict.py some_image.jpg         # CLI inference without the server
```

Training writes the weights to `app/models/`, real metrics to `app/models/rider_detector.metrics.json` and curves to
`docs/figures/rider_detector/`. The **Model** page reads them automatically.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, device, whether models/helmet model are loaded |
| POST | `/api/analyze/image` | multipart `file` (jpg/jpeg/png/webp) → detections, overlay, violations, summary, evidence |
| POST | `/api/analyze/video` | multipart `file` (mp4/avi/mov) + optional `stop_line`, `red_from`, `red_to`, `direction` → `{job_id}` |
| GET | `/api/jobs/{id}` | video progress / result id |
| GET | `/api/rules` | rule database with runtime status |
| GET | `/api/history`, `/api/history/{id}` | saved analyses (DELETE to remove) |
| GET | `/api/model/info` | loaded models, real metrics (or "unavailable") |

```bash
curl -F "file=@traffic.jpg" http://127.0.0.1:8000/api/analyze/image
```

```json
{
  "status": "success",
  "detections": [ ... ],
  "violations": [
    { "violation_id": "triple_riding", "name": "More Than Two Persons on a Two-Wheeler",
      "status": "possible", "confidence": 0.81, "confidence_level": "medium", "vehicle": "Motorcycle",
      "evidence": "3 persons positioned on the same two-wheeler (permitted: driver + 1 pillion).",
      "requires_human_verification": true, "evidence_image": "/media/<id>/evidence_1.jpg" }
  ],
  "confidence": 0.81,
  "processing_time": 1.34
}
```

(abridged; the real response also contains `summary`, `overlay`, `intelligence`, `media`, `notes`).

## Testing

```bash
python -m pytest
```

Covers API health, image/video upload, rule engine, confidence bands, detection parsing, duplicate filtering, tracking and invalid files.
Tests use a scripted detector so rule/API logic is deterministic; model quality is measured separately by `scripts/validate.py`.

## Performance metrics

Measured by `scripts/train.py` on the held-out **test** split (80 images, 86 boxes) and read from `app/models/rider_detector.metrics.json` (also shown on the Model page):

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 0.965 |
| F1 | 0.982 |
| mAP@50 | 0.980 |
| mAP@50-95 | 0.748 |

Model YOLOv8n (fine-tuned), 50 epochs, image size 640, batch 8, trained on NVIDIA GeForce MX550.
Dataset: single class `person_bike`; train/val/test = 594/120/80 images.

**Read these numbers with care.** The test split is tiny and about half of it consists of flipped copies of the other half, so it holds only ~40 distinct scenes;
the images are mostly clear daytime scooter photos. This measures how well the rider detector fits *this dataset*, not real CCTV performance.
Metrics for the triple-riding rule itself were **not** measured: the dataset has no per-rider or violation labels, so only qualitative checks were done
(on the dataset images the rule is deliberately conservative - most crowded scenes end up as *insufficient evidence*). Training curves: `docs/figures/rider_detector/`.

### Helmet detector (EdgeVision, held-out test split: 239 images, 815 boxes)

| Metric | Value |
|---|---|
| Precision | 0.907 |
| Recall | 0.870 |
| F1 | 0.888 |
| mAP@50 | 0.915 |
| mAP@50-95 | 0.655 |
| AP@50 `no_helmet` | 0.885 |
| AP@50 `helmet` | 0.902 |

YOLOv8n, 25 epochs, image size 640, batch 8, NVIDIA GeForce MX550. Curves: `docs/figures/helmet_detector/`.
These are detector metrics on the EdgeVision test split only. The end-to-end *rule* (rider association + head-region search) has no labelled benchmark; it was spot-checked on the rider dataset images
(helmeted police convoy: no violations; bare-headed scooter riders: flagged). Expect more errors on small/blurred heads and unfamiliar camera angles.

## Screenshots

Run the app and save your own screenshots to `docs/screenshots/` (none are committed because none were fabricated):
dashboard, analysis result with a violation card, model page, rules page.

## Demo flow (under 2 minutes)

1. Open the **Dashboard** — point out the status tiles and the pipeline strip.
2. **Analyze → Image** — drop a traffic photo.
3. Left: original, right: YOLO overlay. Hover boxes for class and confidence.
4. Explain the **Detection summary** (vehicles, motorcycles, riders linked).
5. For a motorcycle with three riders, show the **Possible Traffic Violation** card: confidence bar, evidence text, legal reference, *Requires Human Verification*, evidence crop.
6. Show the **Traffic Intelligence Score**.
7. **Detection History** — the analysis was saved; open it again.
8. **Traffic Rules** — the JSON-driven rule table (status shows what is active vs needs a helmet model).
9. **Model** — real metrics for both detectors read from their training runs; explain limitations honestly.
10. Optional: **Analyze → Video** to show tracking, the timeline chart and temporal validation.

## Limitations

* The helmet detector was trained on one dataset (EdgeVision); small, blurred, occluded or far-away heads are the main failure cases, and a helmet-like object (hood, cap) can confuse it.
* Small training set (a few hundred distinct images, mostly clear daytime scooter photos); expect errors on real CCTV footage (night, rain, occlusion, distance).
* Association logic is heuristic geometry, not learned; dense crowds of scooters make rider counting ambiguous (such cases become *insufficient evidence*).
* Red-light checking depends on operator-supplied stop line and signal timing. Lane violations and number-plate OCR are not implemented.
* A single image carries no temporal information; only video can support temporal validation.

## Future improvements

More diverse helmet data (night, rain, CCTV angles) · number-plate detection + OCR (clearly labelled as AI-generated reading) · automatic signal-state detection · lane segmentation · per-camera calibration · larger and more diverse training data.

## Legal disclaimer

For educational use only. Indian traffic laws and penalties change; the legal references stored in `app/rules/traffic_rules.json` are
flagged `legal_verified: false` and must be checked against the current Motor Vehicles Act, 1988, the Central Motor Vehicles Rules and
official notifications before being relied upon. No penalty amounts are stored. Outputs are AI-detected *possible* violations based on
available visual evidence and require human verification. Ultralytics YOLO is AGPL-3.0 licensed. The EdgeVision dataset is CC BY 4.0 (attribution above).

## Authors

Muhammed Aabir Turab — college deep-learning project.
