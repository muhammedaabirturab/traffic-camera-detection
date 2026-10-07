"""Video analysis: YOLO detection + ByteTrack/BoT-SORT tracking + temporal validation.

Frames are sampled at ``Settings.video_target_fps`` (e.g. 6 analysed frames per second
of footage), each sampled frame goes through the same per-frame reasoning as an image,
and the ``TemporalAggregator`` only confirms violations that persist across frames for
the same tracked vehicle / rider. An annotated H.264 MP4 is written for playback in the
browser.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from app.analysis.evidence_generator import EvidenceWriter
from app.analysis.image_analyzer import DISCLAIMER, attach_plate, model_flags
from app.config import Settings
from app.detection.preprocessing import enhance_low_light, limit_size
from app.detection.signal import SignalStateEstimator
from app.detection.tracker import TemporalAggregator
from app.detection.types import VEHICLE_LABELS, Box, Observation
from app.detection.violation_detector import analyse_frame, rider_summary
from app.detection.yolo_detector import ModelManager, deduplicate
from app.rules.rule_engine import RuleEngine
from app.utils import visualization as viz

log = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]
MAX_VIDEO_SIDE = 1280


class InvalidVideoError(ValueError):
    pass


class _VideoWriter:
    """H.264 writer via imageio-ffmpeg (plays in browsers); falls back to OpenCV mp4v."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        self.path = path
        self.browser_playable = True
        self._gen = None
        self._cv = None
        try:
            import imageio_ffmpeg

            self._gen = imageio_ffmpeg.write_frames(str(path), (width, height), fps=max(1.0, fps), codec="libx264",
                                                    pix_fmt_in="bgr24", pix_fmt_out="yuv420p", quality=6,
                                                    macro_block_size=2, ffmpeg_log_level="error",
                                                    output_params=["-movflags", "+faststart"])
            self._gen.send(None)
        except Exception:
            log.warning("imageio-ffmpeg unavailable, falling back to OpenCV mp4v (may not play in browsers)")
            self._gen = None
            self.browser_playable = False
            self._cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height))

    def write(self, frame: np.ndarray) -> None:
        if self._gen is not None:
            self._gen.send(np.ascontiguousarray(frame))
        elif self._cv is not None:
            self._cv.write(frame)

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
        if self._cv is not None:
            self._cv.release()


def _even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]


class VideoAnalyzer:
    def __init__(self, settings: Settings, models: ModelManager, rules: RuleEngine):
        self.s = settings
        self.models = models
        self.rules = rules

    def analyze(self, path: Path, filename: str, analysis_id: str, created_at: str,
                stop_line: Optional[float] = None, line_direction: str = "any",
                progress: Optional[ProgressFn] = None) -> dict:
        s = self.s
        progress = progress or (lambda f, m: None)
        t0 = time.perf_counter()
        self.models.load()

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise InvalidVideoError("Video could not be opened (unsupported codec or corrupt file)")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 1 or fps > 240:
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_frames = int(s.video_max_seconds * fps)
        frames_to_read = min(total, max_frames) if total > 0 else max_frames
        stride = max(1, int(round(fps / s.video_target_fps)))
        truncated = total > max_frames

        tracker = self.models.new_tracking_detector()
        writer = EvidenceWriter(s.results_dir, analysis_id)
        signal = SignalStateEstimator(s.signal_smoothing_frames)
        aggregator: Optional[TemporalAggregator] = None
        video_out: Optional[_VideoWriter] = None
        timeline: list[dict] = []
        per_frame_max_vehicles = 0
        rider_peak = {"riders": 0}
        first_frame: Optional[np.ndarray] = None
        signal_seen = False
        low_light_frames = 0
        frame_size = None

        idx = analysed = 0
        progress(0.01, "Starting video analysis")
        try:
            while idx < frames_to_read:
                ok = cap.grab()
                if not ok:
                    break
                if idx % stride != 0:
                    idx += 1
                    continue
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    idx += 1
                    continue
                frame = _even(limit_size(frame, MAX_VIDEO_SIDE))
                frame, enhanced = enhance_low_light(frame)
                low_light_frames += int(enhanced)
                h, w = frame.shape[:2]
                if frame_size is None:
                    frame_size = (w, h)
                    first_frame = frame.copy()
                    line_px = stop_line * h if stop_line is not None else None
                    aggregator = TemporalAggregator(s, line_px, line_direction)
                    video_out = _VideoWriter(writer.dir / "processed.mp4", w, h, fps / stride)
                elif (w, h) != frame_size:
                    frame = cv2.resize(frame, frame_size)

                ts = idx / fps
                dets = self.models.filter_by_confidence(tracker.track(frame, s.tracker))
                dets.extend(self.models.detect_auxiliary(frame))
                dets = deduplicate(dets, s.duplicate_iou)
                for i, d in enumerate(dets):
                    d.det_id = i

                fa = analyse_frame(dets, s, self.models.helmet_available, frame.shape[0])
                state, agreement = signal.update(frame, dets)
                signal_seen |= any(d.label == "traffic_light" for d in dets)
                aggregator.update(idx, ts, frame, fa, state, agreement)

                n_veh = sum(1 for d in dets if d.label in VEHICLE_LABELS)
                per_frame_max_vehicles = max(per_frame_max_vehicles, n_veh)
                rs = rider_summary(fa.units)
                rider_peak["riders"] = max(rider_peak["riders"], rs["riders"])
                timeline.append({"t": round(ts, 2), "frame": idx, "vehicles": n_veh,
                                 "riders": rs["riders"], "candidates": len(fa.candidates), "signal": state})

                annotated = viz.draw_detections(frame, dets)
                for c in fa.candidates:
                    rule = self.rules.get(c.rule_id)
                    annotated = viz.draw_violation(annotated, c.region, rule.short_label if rule else c.rule_id, c.confidence)
                if stop_line is not None:
                    annotated = viz.draw_stop_line(annotated, stop_line * h, state)
                annotated = viz.draw_hud(annotated, f"TRAFFICGUARD AI  |  t={ts:6.2f}s  frame {idx}")
                video_out.write(annotated)

                analysed += 1
                idx += 1
                if analysed % 5 == 0:
                    progress(min(0.95, idx / max(frames_to_read, 1)), f"Analysed {analysed} frames ({ts:.1f}s)")
        finally:
            cap.release()
            if video_out is not None:
                video_out.close()

        if aggregator is None or first_frame is None:
            raise InvalidVideoError("No frames could be read from the video")

        progress(0.96, "Validating violations across frames")
        confirmed, temporal_obs = aggregator.finalize()

        violations: list[dict] = []
        observations: list[Observation] = list(temporal_obs)
        for cv in confirmed:
            cand = replace(cv.candidate, confidence=cv.confidence)
            v, obs = self.rules.evaluate(cand, "video")
            if v is None:
                if obs is not None:
                    observations.append(obs)
                continue
            v["id"] = len(violations) + 1
            v.update({
                "frame_index": cv.frame_index,
                "timestamp": round(cv.timestamp, 2),
                "first_seen": round(cv.first_frame / fps, 2),
                "last_seen": round(cv.last_frame / fps, 2),
                "frames_observed": cv.frames_observed,
                "frames_evaluable": cv.frames_evaluable,
                "temporal_consistency": round(cv.frames_observed / max(cv.frames_evaluable, 1), 3),
            })
            v["evidence_images"] = writer.violation_evidence(v["id"], cv.frame, cv.detections, Box(*v["region"]),
                                                             v["title"], v["confidence"])
            attach_plate(v, cv.frame, cv.detections, self.models, s)
            violations.append(v)

        if not self.models.helmet_available and rider_peak["riders"] > 0:
            observations.insert(0, Observation("NO_HELMET", "Helmet checks were not performed because no helmet "
                                                            "detection model is loaded (see models/README.md)."))
        if stop_line is not None and not signal_seen:
            observations.append(Observation("RED_LIGHT_JUMP", "A stop line was configured but no traffic light was "
                                                              "detected in the footage — red-light rule not evaluated."))

        unique = aggregator.unique_counts()
        unique_vehicles = sum(v for k, v in unique.items() if k in VEHICLE_LABELS)
        media = {
            "processed_video": writer.url("processed.mp4"),
            "video_browser_playable": video_out.browser_playable if video_out else False,
            "thumbnail": writer.thumbnail(first_frame),
            "first_frame": writer.save("first_frame.jpg", first_frame),
        }
        progress(1.0, "Completed")
        return {
            "status": "success",
            "analysis_id": analysis_id,
            "kind": "video",
            "filename": filename,
            "created_at": created_at,
            "video": {
                "fps": round(fps, 2), "total_frames": total, "duration": round(total / fps, 2) if total else None,
                "analysed_frames": analysed, "frame_stride": stride, "analysed_fps": round(fps / stride, 2),
                "width": frame_size[0], "height": frame_size[1], "truncated": truncated,
                "max_seconds": s.video_max_seconds, "low_light_frames": low_light_frames,
                "tracker": s.tracker.replace(".yaml", ""),
                "stop_line": stop_line, "line_direction": line_direction if stop_line is not None else None,
            },
            "media": media,
            "summary": {
                "vehicles": unique_vehicles,
                "unique_vehicles": unique_vehicles,
                "unique_by_class": unique,
                "max_vehicles_in_frame": per_frame_max_vehicles,
                "max_riders_in_frame": rider_peak["riders"],
                "possible_violations": len(violations),
                "insufficient_evidence": len(observations),
                "signal_frames": aggregator.signal_frames,
            },
            "violations": violations,
            "observations": [o.to_dict() for o in observations],
            "timeline": timeline[:: max(1, len(timeline) // 600)],
            "confidence": max((v["confidence"] for v in violations), default=None),
            "models": model_flags(self.models),
            "processing_time": round(time.perf_counter() - t0, 3),
            "disclaimer": DISCLAIMER,
        }
