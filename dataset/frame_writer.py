"""Escritura de frames JPG individuales (30 FPS) + index.json por episodio.

Por que JPG individuales y NO MP4 como fuente de entrenamiento:
  (1) acceso aleatorio directo sin decodificar video -> no satura CPU ni deja
      la GPU de Vast.ai esperando;
  (2) sin compresion temporal inter-frame que emborrona la rotacion rapida y
      dana la regresion de cuaternion/omega;
  (3) compatible con dataloaders estandar y paralelizable.

El index.json alinea cada frame_XXXXX.jpg con su timestamp y con el indice del
estado fisico (100 Hz) mas cercano (``state_idx``), para el dataloader de
VideoMamba.
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
import cv2


class FrameWriter:
    def __init__(self, frames_root: str, episode_id: int, jpg_quality: int = 95):
        self.ep_dir = os.path.join(frames_root, f"ep_{episode_id:04d}")
        os.makedirs(self.ep_dir, exist_ok=True)
        self.jpg_quality = int(jpg_quality)
        self.index_entries: List[dict] = []

    def write_frame(self, frame_idx: int, img_rgb: np.ndarray) -> str:
        """Guarda un frame RGB uint8 como JPG. Devuelve el nombre de archivo."""
        fname = f"frame_{frame_idx:05d}.jpg"
        path = os.path.join(self.ep_dir, fname)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality])
        return fname

    def add_index(self, fname: str, t: float, state_idx: int) -> None:
        self.index_entries.append({
            "file": fname,
            "t": round(float(t), 5),
            "state_idx": int(state_idx),
        })

    def write_index(self, fps: int, resolution_hw: Tuple[int, int]) -> None:
        """Escribe index.json (fps, n_frames, resolution, frames[])."""
        h, w = resolution_hw
        doc = {
            "fps": int(fps),
            "n_frames": len(self.index_entries),
            "resolution": [int(w), int(h)],  # [width, height]
            "frames": self.index_entries,
        }
        with open(os.path.join(self.ep_dir, "index.json"), "w",
                  encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
