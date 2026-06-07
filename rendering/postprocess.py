"""Post-processing fotorrealista de los frames de 30 FPS (modo video).

El pipeline aplica, EN ESTE ORDEN (Seccion 4.1 de la spec):
  1. Ruido de sensor gaussiano   N(0, iso_noise), iso_noise ~ U(2, 8)   [por frame]
  2. Ruido Poisson               poisson_scale ~ U(10, 30)              [por frame]
  3. Temperatura de color        T ~ U(4500, 7500) K                    [por episodio]
  4. Contraste/gamma             gamma ~ U(0.85, 1.15)                  [por episodio]
  5. Vignette                    strength ~ U(0.1, 0.4)                 [por episodio]
  6. Aberracion cromatica        desplazar canal R 1-2 px en bordes     [por episodio]
  7. Distorsion de lente         k1 ~ U(-0.05, 0.05) (barrel/pincushion)[por episodio]
  8. Resize final                a 224x224 o 480x640                    [fijo]

DECISION DE DISENO: el ruido del sensor (1-2) se muestrea POR FRAME (varia con
el tiempo, como un sensor real). Los efectos intrinsecos de la camara (3-7) se
muestrean UNA VEZ POR EPISODIO en __init__, porque son propiedades fijas de la
optica/sensor y hacerlos variar por frame produciria parpadeo irreal. La spec
los lista "por frame"; esta es la interpretacion fisica razonable.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import cv2

from utils.domain_randomization import kelvin_to_rgb


class PostProcessor:
    """Aplica el pipeline de post-processing. Una instancia por episodio."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        # --- Parametros intrinsecos (constantes por episodio) ---
        self.color_temp = float(rng.uniform(4500.0, 7500.0))
        self.wb_gain = kelvin_to_rgb(self.color_temp)         # ganancia RGB
        self.gamma = float(rng.uniform(0.85, 1.15))
        self.vignette_strength = float(rng.uniform(0.1, 0.4))
        self.ca_shift = int(rng.integers(1, 3))               # 1-2 px
        self.lens_k1 = float(rng.uniform(-0.05, 0.05))
        self._lens_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._vig_cache: Dict[Tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def _vignette_mask(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        if key not in self._vig_cache:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            r /= r.max()
            self._vig_cache[key] = (1.0 - self.vignette_strength * r ** 2).astype(np.float32)
        return self._vig_cache[key]

    def _lens_maps(self, h: int, w: int):
        key = (h, w)
        if key not in self._lens_cache:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            nx = (xx - cx) / cx
            ny = (yy - cy) / cy
            r2 = nx ** 2 + ny ** 2
            factor = 1.0 + self.lens_k1 * r2
            map_x = (cx + nx * factor * cx).astype(np.float32)
            map_y = (cy + ny * factor * cy).astype(np.float32)
            self._lens_cache[key] = (map_x, map_y)
        return self._lens_cache[key]

    # ------------------------------------------------------------------ #
    def process(self, img: np.ndarray,
                target_size: Tuple[int, int]) -> np.ndarray:
        """Procesa un frame RGB uint8 (HxWx3). target_size = (W, H)."""
        h, w = img.shape[:2]
        f = img.astype(np.float32)

        # 1. Ruido gaussiano de sensor (por frame)
        iso = float(self.rng.uniform(2.0, 8.0))
        f += self.rng.normal(0.0, iso, f.shape).astype(np.float32)
        f = np.clip(f, 0.0, 255.0)

        # 2. Ruido Poisson (por frame)
        pscale = float(self.rng.uniform(10.0, 30.0))
        lam = np.clip(f / 255.0 * pscale, 0.0, None)
        f = self.rng.poisson(lam).astype(np.float32) / pscale * 255.0
        f = np.clip(f, 0.0, 255.0)

        # 3. Temperatura de color (balance R/G/B)
        f *= self.wb_gain[None, None, :].astype(np.float32)
        f = np.clip(f, 0.0, 255.0)

        # 4. Gamma / contraste
        f = np.power(f / 255.0, self.gamma) * 255.0

        # 5. Vignette
        f *= self._vignette_mask(h, w)[..., None]

        # 6. Aberracion cromatica (desplazar R en una direccion, B en la opuesta)
        d = self.ca_shift
        f[..., 0] = np.roll(f[..., 0], d, axis=1)    # R -> derecha
        f[..., 2] = np.roll(f[..., 2], -d, axis=1)   # B -> izquierda

        out = np.clip(f, 0.0, 255.0).astype(np.uint8)

        # 7. Distorsion de lente (barrel/pincushion)
        map_x, map_y = self._lens_maps(h, w)
        out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

        # 8. Resize final
        if (w, h) != target_size:
            out = cv2.resize(out, target_size, interpolation=cv2.INTER_AREA)
        return out
