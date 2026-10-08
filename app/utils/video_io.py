"""H.264 video writer (browsers cannot play OpenCV's default mp4v), with a safe fallback."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.path, self.size = path, size
        self.proc = None
        self.cv = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                 "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                 "-crf", "26", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as exc:
            log.warning("ffmpeg unavailable (%s); falling back to mp4v (may not play in browsers)", exc)
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        if self.proc is not None:
            try:
                self.proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                log.warning("ffmpeg pipe closed early")
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            self.proc.wait(timeout=60)
        elif self.cv is not None:
            self.cv.release()
