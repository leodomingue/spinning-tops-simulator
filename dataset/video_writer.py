"""Escritura de MP4 SOLO para inspeccion humana / debug (flag --save-mp4).

IMPORTANTE: el MP4 NO se usa para entrenar (la fuente de entrenamiento son los
frames JPG). La compresion temporal H.264 emborrona la rotacion rapida y
dania la regresion de cuaternion/omega. Esto es solo para mirar episodios.

Usa imageio + imageio-ffmpeg (backend H.264). Si no estan instalados, avisa y
no rompe el run.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


class VideoWriter:
    def __init__(self, video_root: str, episode_id: int, fps: int = 30):
        os.makedirs(video_root, exist_ok=True)
        self.path = os.path.join(video_root, f"ep_{episode_id:04d}.mp4")
        self.fps = int(fps)
        self._writer = None
        self._ok = True
        try:
            import imageio
            self._imageio = imageio
            self._writer = imageio.get_writer(
                self.path, fps=self.fps, codec="libx264",
                quality=8, macro_block_size=None,
            )
        except Exception as e:  # imageio / ffmpeg no disponible
            print(f"[WARN] --save-mp4 deshabilitado (imageio/ffmpeg): {e}")
            self._ok = False

    def append(self, img_rgb: np.ndarray) -> None:
        if self._ok and self._writer is not None:
            self._writer.append_data(np.ascontiguousarray(img_rgb))

    def close(self) -> None:
        if self._ok and self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
