# Methodology

All reasoning is geometric and documented here so it can be explained in a viva. It is heuristic: **it does not claim
100 % accuracy and every output requires human verification.**

## 1. Detectors

* **COCO detector** (YOLOv8n): `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`.
* **Rider detector** (YOLOv8n fine-tuned here): one class `person_bike` - a box around a person *and* the two-wheeler they ride.
* **Helmet detector** (optional, not shipped): `helmet` / `no_helmet`. Without it, helmet status is `not_assessed`.

Two-wheelers come from the COCO `motorcycle` class and from the rider detector; boxes describing the same machine are
fused (intersection-over-smaller-box > 0.6) into one **unit**. Near-identical boxes of one class are removed
(IoU > 0.7, keep the most confident).

## 2. Person <-> two-wheeler association

For every person `p` and unit `u` an association score in `[0, 1]` is computed from:

| Cue | Meaning |
|---|---|
| overlap | share of the person's box inside the *rider region* (bike box widened 10 %, extended upward by 1.3 x bike height) |
| horizontal alignment | person's centre x inside the bike's x-range (decays linearly outside) |
| seating height | person's bottom edge lies inside the bike box (hips/legs reach the machine) |
| scale | person height is 0.7-2.8 x the bike height |
| containment | share of the person inside the rider-model box (only if that box exists) |

Weights: `0.30 overlap + 0.20 horizontal + 0.15 height + 0.10 scale + 0.25 containment` (without a rider box:
`0.40 / 0.25 / 0.20 / 0.15`). Each person is assigned to the best unit.

* score >= `MIN_ASSOCIATION_SCORE` (0.60) -> **strong**: counted as a rider/pillion.
* `WEAK_ASSOCIATION_SCORE` (0.40) <= score < 0.60 -> **weak**: ambiguous, *never counted*.
* below -> a bystander.

The largest person on a bike is labelled the driver, others pillions. **Same-depth check:** an additional person only stays "strong"
if their box area is >= 25 % and height >= 50 % of the driver's, and their centre lies inside the central 80 % of the bike's width.
Otherwise they are demoted to *weak*: they are background pedestrians or riders of *other* bikes in a convoy (found while testing on the dataset:
without this check, crowded scooter scenes produced many false "3 on board" results).

## 3. Rules

**More than two on a two-wheeler** (`triple_riding`)

* >= 3 strong riders -> *Possible violation*; confidence = `min(bike_conf, mean person_conf) x (0.5 + 0.5 x mean association)`.
* 2 strong + >= 1 weak -> **Insufficient visual evidence** (reported, not counted).
* People merely standing near a motorcycle do not reach the strong threshold.

**Helmet rules** (`no_helmet_rider`, `no_helmet_pillion`) - only when a helmet model is installed

1. Take each strong rider; head region = top 30 % of the person box (widened 10 %).
2. Helmet detection in that region -> `helmet` (no finding).
3. Explicit `no_helmet` detection -> finding with normal confidence.
4. Nothing found -> `no_helmet_inferred`: finding with confidence **damped** (`0.7 x person conf x association`), because *not seeing* a helmet is weaker evidence than *seeing* a bare head.
5. Head too small/cut off -> `unknown`, no finding.

**Stop line on red** (`red_light_jump`, video only, experimental) - the model cannot read signals. The operator supplies
the stop-line height and the red-phase window; a ByteTrack track whose bottom edge crosses the line (in the chosen
direction) inside that window is flagged. A single image can never support this rule.

Not implemented (listed in the rule database as such): lane violations, number-plate OCR.

## 4. Confidence & false-positive control

* Detector threshold `CONFIDENCE_THRESHOLD` (default 0.40) and NMS IoU `IOU_THRESHOLD`.
* Bands (configurable): **high** >= 0.85, **medium** 0.65-0.84, **low** < 0.65.
* A finding below `MIN_VIOLATION_CONFIDENCE` (0.50) becomes *insufficient evidence*.
* Spatial validation and relationship checks (section 2); duplicate filtering (section 1).
* **Temporal validation (video):** a candidate must appear in >= `VIDEO_MIN_VIOLATION_FRAMES` analysed frames and in >= 30 % of the frames in which that vehicle was seen; shorter blips are discarded and counted as "short-lived candidates".
* UI wording is always "Possible violation" / "Requires human verification".

## 5. Traffic Intelligence Score

An analytical summary, not a legal judgment: object counts, normal vs flagged objects, mean detection confidence and

`risk_index = 100 x sum(confidence of possible violations) / number of vehicles`

mapped to LOW (no possible violations), MODERATE (< 50) or HIGH (>= 50). For video it also draws violations per second.

## 6. Legal references

`app/rules/traffic_rules.json` carries the references. They are marked `legal_verified: false` and the UI says to
verify them against current official sources (India Code / MoRTH). Penalty amounts are deliberately not stored.
