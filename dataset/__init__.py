"""Escritores de salida: estados JSON (100 Hz), frames JPG y MP4 de debug.

NOTA sobre el nombre del paquete: por la especificacion, este paquete de
codigo se llama ``dataset`` y la carpeta de SALIDA por defecto tambien se
llama ``dataset`` (``--out dataset``). No colisionan: la salida crea
subcarpetas (``frames/``, ``states/``, ``trajectories/``, ``ude/``) que no
chocan con los modulos ``.py`` de este paquete. Aun asi, se recomienda usar
``--out output`` si se prefiere separar codigo y datos.
"""

from .state_logger import StateLogger
from .frame_writer import FrameWriter
from .video_writer import VideoWriter

__all__ = ["StateLogger", "FrameWriter", "VideoWriter"]
