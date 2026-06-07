"""Logger de estados fisicos a 100 Hz -> JSON (formato exacto de la Seccion 5).

Los estados se guardan SIEMPRE a 100 Hz (cada 10 timesteps de 1 ms) en AMBOS
modos. main.py decide CUANDO llamar a ``add_state`` (cada 10 pasos). Aqui solo
se acumulan y se serializa el JSON con su metadata.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np


def _r(x, nd):
    """Redondeo recursivo para compactar el JSON sin perder precision util."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, nd) for v in x]
    return round(float(x), nd)


class StateLogger:
    """Acumula frames de estado y escribe el JSON del episodio."""

    def __init__(self, params, mode: str):
        self.params = params
        self.mode = mode
        self.frames: List[dict] = []

    def add_state(self, t: float, q, omega, x, v,
                  has_contact: bool, contact_force: float,
                  angle_from_vertical: float, fall_detected: bool,
                  motion_state: str = "spinning") -> None:
        """Anade un frame de estado (a 100 Hz).

        ``fall_detected`` es el latch monotono (para el cutoff de la UDE).
        ``motion_state`` es el estado INSTANTANEO de ese frame:
          * "spinning" : de pie y girando (no ha caido todavia).
          * "fallen"   : se cayo (angulo > umbral) pero aun se mueve.
          * "stopped"  : practicamente detenido (||omega|| ~ 0).
        """
        self.frames.append({
            "t": _r(t, 5),
            "q": _r(q, 7),
            "omega": _r(omega, 6),
            "x": _r(x, 6),
            "v": _r(v, 6),
            "has_contact": bool(has_contact),
            "contact_force": _r(contact_force, 4),
            "angle_from_vertical": _r(angle_from_vertical, 3),
            "fall_detected": bool(fall_detected),
            "motion_state": str(motion_state),
        })

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def _fall_time(self) -> Optional[float]:
        """t del primer frame con fall_detected==True (latch). None si nunca cae."""
        for fr in self.frames:
            if fr["fall_detected"]:
                return fr["t"]
        return None

    def write(self, path: str, video_fps: Optional[int],
              post_fall_seconds: Optional[float] = None) -> dict:
        """Escribe el JSON y devuelve la metadata (para progress/manifest)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        duration = self.frames[-1]["t"] if self.frames else 0.0
        fall_time = self._fall_time()

        metadata = {
            "episode_id": self.params.episode_id,
            "seed": self.params.seed,
            "mode": self.mode,
            "top_type": self.params.top_type,
            "floor_type": self.params.floor_type,
            "physics": self.params.physics_metadata(),
            # Marcadores de color (constantes por episodio): posiciones locales
            # y colores de los >=3 puntos no colineales pegados al cuerpo.
            "markers": (self.params.markers or []),
            "simulation": {
                "physics_timestep": 0.001,
                "state_sample_rate_hz": 100,
                "video_fps": (int(video_fps) if video_fps is not None else None),
                "integrator": "implicitfast",
                "duration_seconds": _r(duration, 4),
                "total_state_frames": len(self.frames),
                "post_fall_seconds": (float(post_fall_seconds)
                                      if post_fall_seconds is not None else None),
            },
            "fall_time": (_r(fall_time, 5) if fall_time is not None else None),
            "valid": True,  # validate_physics() puede ponerlo a false despues
        }

        doc = {"metadata": metadata, "frames": self.frames}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        return metadata
