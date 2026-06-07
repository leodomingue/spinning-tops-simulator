"""Wrapper de ``mujoco.Renderer`` + camara libre por episodio.

Solo se encarga de RENDERIZAR el estado actual de ``data``. El motion blur
(promediado de subframes y stepping de fisica entre ellos) se orquesta en
main.py, que llama a ``render()`` en los instantes de subframe.

Resolucion (--resolution):
  * 224 -> imagenes 224x224  (H=224, W=224)
  * 480 -> imagenes 480x640  (H=480, W=640, VGA)

Backend GL headless: se controla con la variable de entorno MUJOCO_GL
(egl en GPU de servidor, osmesa en CPU). main.py la configura antes de
importar mujoco.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import mujoco


def resolution_to_hw(resolution: int) -> Tuple[int, int]:
    """--resolution -> (height, width)."""
    if int(resolution) == 224:
        return 224, 224
    if int(resolution) == 480:
        return 480, 640
    raise ValueError(f"resolution no soportada: {resolution} (usa 224 o 480)")


class TopRenderer:
    def __init__(self, model: mujoco.MjModel, params, resolution: int,
                 rng: np.random.Generator, body_id: None | None = None):
        self.height, self.width = resolution_to_hw(resolution)
        self.model = model
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.rng = rng if rng is not None else np.random.default_rng(params.seed + 13)
        self.body_id = body_id

        # Camara libre configurada con los parametros DR del episodio.
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.array(params.cam_lookat, dtype=np.float64)
        cam.distance = float(params.cam_distance)
        cam.azimuth = float(params.cam_azimuth)
        cam.elevation = float(params.cam_elevation)
        self.cam = cam
        self._base_lookat = np.array(params.cam_lookat, dtype=np.float64)

        # Opciones de escena (sombras on para realismo).
        self.scene_option = mujoco.MjvOption()

    def render(self, data: mujoco.MjData, jitter: bool = False) -> np.ndarray:
        """Renderiza el estado actual -> RGB uint8 (H, W, 3).

        Si jitter=True, aplica una vibracion de camara ~0.1 mm (obturador),
        usada entre subframes para un motion blur mas realista.
        """
        # Actualizar lookat a la posición actual del body
        if self.body_id is not None:
            body_pos = data.xpos[self.body_id]
            self.cam.lookat[:] = body_pos
        else:
            self.cam.lookat[:] = self._base_lookat

        if jitter:
            j = self.rng.normal(0.0, 1e-4, 3)
            self.cam.lookat[:] += j

        self.renderer.update_scene(data, camera=self.cam)
        return self.renderer.render()

    @property
    def out_size(self) -> Tuple[int, int]:
        """(W, H) para cv2.resize."""
        return (self.width, self.height)

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            pass
