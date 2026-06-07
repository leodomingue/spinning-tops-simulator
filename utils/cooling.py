"""Gestion de cooling / batching para runs largos en Vast.ai.

Cada N episodios:
  * guarda el estado de progreso (lo hace main.py via progress.json),
  * fuerza gc.collect(),
  * libera cache de CUDA si torch esta disponible,
  * duerme unos segundos para que la GPU/CPU baje temperatura,
  * loggea ETA.
"""

from __future__ import annotations

import gc
import time
from typing import Optional


class CoolingManager:
    """Pausas periodicas de enfriamiento + estimacion de ETA."""

    def __init__(self, interval: int, sleep_seconds: float = 30.0,
                 enabled: bool = True):
        self.interval = max(1, int(interval))
        self.sleep_seconds = float(sleep_seconds)
        self.enabled = enabled
        self._t_start = time.time()
        self._episodes_done = 0

    def _empty_cuda_cache(self) -> bool:
        """Libera la cache de CUDA si torch esta instalado. Devuelve True si lo hizo."""
        try:
            import torch  # import opcional
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return True
        except Exception:
            pass
        return False

    def tick(self, episode_idx: int, total: int) -> None:
        """Llamar tras completar cada episodio. ``episode_idx`` es 1-based."""
        self._episodes_done += 1
        if not self.enabled:
            return
        if episode_idx % self.interval != 0:
            return

        # ETA basada en throughput medio observado hasta ahora.
        elapsed = time.time() - self._t_start
        rate = self._episodes_done / max(elapsed, 1e-6)  # eps/s
        remaining = max(0, total - episode_idx)
        eta_min = (remaining / rate) / 60.0 if rate > 0 else float("nan")

        gc.collect()
        freed = self._empty_cuda_cache()
        print(
            f"[COOLING] Episodio {episode_idx}/{total} | "
            f"{rate*60:.1f} eps/min | ETA: {eta_min:.1f} min | "
            f"cuda_cache_freed={freed} | sleep {self.sleep_seconds:.0f}s",
            flush=True,
        )
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
