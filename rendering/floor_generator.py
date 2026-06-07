"""Generacion AUTOMATICA, procedural y aleatoria del suelo por episodio.

No requiere imagenes manuales: cada episodio elige al azar uno de varios
generadores de textura para maximizar la diversidad visual del fondo
(robustez sim-to-real de VideoMamba).

Las texturas numpy se pasan a MuJoCo como bytes PNG en memoria mediante el
VFS de ``mujoco.MjModel.from_xml_string(xml, assets)`` -> NO se escriben
ficheros temporales y funciona headless en Vast.ai.

``generate_floor(...)`` devuelve ``(asset_xml, floor_type, assets_dict)``:
  * asset_xml : snippet ``<texture/><material name="floor_mat"/>`` para inyectar
                en <asset> del modelo.
  * floor_type: string trazable que se guarda en metadata.floor_type.
  * assets_dict: {nombre_virtual: bytes_png} para from_xml_string(xml, assets).

Pool de generadores: checker (builtin), gradient (builtin), madera (numpy),
marmol (numpy), fractal/perlin (numpy), liso con micro-ruido (numpy), y
opcionalmente imagenes reales de assets/floors/ con prob --real-floor-prob.
"""

from __future__ import annotations

import os
import glob
from typing import Dict, Tuple

import numpy as np
import cv2

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------- #
# Helpers de ruido (numpy puro; sin dependencias pesadas)                     #
# --------------------------------------------------------------------------- #
def _fbm(h: int, w: int, octaves: int, rng: np.random.Generator) -> np.ndarray:
    """Fractal Brownian Motion barato: suma de octavas de ruido upsampleado."""
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        res = 2 ** (o + 2)
        small = rng.random((res, res)).astype(np.float32)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += up * amp
        tot += amp
        amp *= 0.5
    acc /= max(tot, 1e-6)
    return np.clip(acc, 0.0, 1.0)


def _encode_png(rgb: np.ndarray) -> bytes:
    """RGB uint8 (HxWx3) -> bytes PNG."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("fallo al codificar PNG de la textura del suelo")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Generadores de textura numpy -> RGB uint8                                   #
# --------------------------------------------------------------------------- #
def _tex_wood(rng, n) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    turb = _fbm(n, n, 4, rng)
    freq = rng.uniform(6.0, 16.0)
    grain = np.sin((xx * freq + turb * rng.uniform(1.5, 4.0)) * np.pi)
    grain = (grain + 1.0) * 0.5
    c1 = np.array(rng.uniform([0.30, 0.16, 0.06], [0.55, 0.32, 0.16]))
    c2 = np.array(rng.uniform([0.55, 0.34, 0.18], [0.78, 0.55, 0.34]))
    img = c1[None, None, :] + grain[..., None] * (c2 - c1)[None, None, :]
    img += (turb[..., None] - 0.5) * 0.06
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _tex_marble(rng, n) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    turb = _fbm(n, n, 5, rng)
    freq = rng.uniform(3.0, 9.0)
    veins = np.sin((xx + yy + turb * rng.uniform(3.0, 7.0)) * freq * np.pi)
    veins = np.abs(veins) ** rng.uniform(0.4, 1.2)
    base = np.array(rng.uniform([0.55, 0.52, 0.48], [0.85, 0.83, 0.80]))
    vein_c = np.array(rng.uniform([0.20, 0.20, 0.22], [0.45, 0.43, 0.40]))
    img = base[None, None, :] + (vein_c - base)[None, None, :] * veins[..., None]
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _tex_fractal(rng, n) -> np.ndarray:
    """Granular tipo cemento/ceramica."""
    base_field = _fbm(n, n, 6, rng)
    grain = rng.normal(0, 0.05, (n, n)).astype(np.float32)
    field = np.clip(base_field + grain, 0, 1)
    c1 = np.array(rng.uniform([0.25, 0.25, 0.25], [0.55, 0.55, 0.55]))
    c2 = np.array(rng.uniform([0.55, 0.55, 0.55], [0.90, 0.90, 0.90]))
    # tinte de color aleatorio leve
    tint = np.array(rng.uniform([0.85, 0.85, 0.85], [1.0, 1.0, 1.0]))
    img = (c1[None, None, :] + field[..., None] * (c2 - c1)[None, None, :]) * tint
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _tex_flat_noise(rng, n) -> np.ndarray:
    """Color plano aleatorio + grano sutil."""
    color = np.array(rng.uniform(0.2, 0.9, 3), np.float32)
    img = np.ones((n, n, 3), np.float32) * color[None, None, :]
    img += rng.normal(0, rng.uniform(0.01, 0.04), (n, n, 3)).astype(np.float32)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


_NUMPY_GENERATORS = {
    "wood_procedural": _tex_wood,
    "marble_procedural": _tex_marble,
    "fractal_procedural": _tex_fractal,
    "flat_micronoise": _tex_flat_noise,
}


# --------------------------------------------------------------------------- #
# Material XML                                                                 #
# --------------------------------------------------------------------------- #
def _material_xml(rng: np.random.Generator) -> str:
    texrepeat = rng.uniform(1.0, 8.0)
    reflectance = rng.uniform(0.0, 0.4)
    specular = rng.uniform(0.0, 0.6)
    shininess = rng.uniform(0.1, 0.8)
    # tinte (mayormente claro para no tapar la textura)
    tint = rng.uniform(0.8, 1.0, 3)
    return (
        f'    <material name="floor_mat" texture="floortex" texuniform="true" '
        f'texrepeat="{texrepeat:.3f} {texrepeat:.3f}" '
        f'reflectance="{reflectance:.4f}" specular="{specular:.4f}" '
        f'shininess="{shininess:.4f}" '
        f'rgba="{tint[0]:.3f} {tint[1]:.3f} {tint[2]:.3f} 1"/>\n'
    )


def _list_real_floors(assets_floors_dir: str):
    if not assets_floors_dir or not os.path.isdir(assets_floors_dir):
        return []
    files = []
    for ext in _IMG_EXTS:
        files += glob.glob(os.path.join(assets_floors_dir, f"*{ext}"))
        files += glob.glob(os.path.join(assets_floors_dir, f"*{ext.upper()}"))
    return sorted(set(files))


def generate_floor(
    rng: np.random.Generator,
    real_floor_prob: float = 0.3,
    assets_floors_dir: str = "assets/floors",
    tex_size: int = 512,
) -> Tuple[str, str, Dict[str, bytes]]:
    """Genera el suelo del episodio. Ver docstring del modulo."""
    assets: Dict[str, bytes] = {}

    # --- Opcion 1: imagen real (si hay y la moneda cae) ---
    real_files = _list_real_floors(assets_floors_dir)
    if real_files and rng.random() < real_floor_prob:
        path = str(rng.choice(real_files))
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (tex_size, tex_size))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            assets["floor.png"] = _encode_png(rgb)
            tex_xml = '    <texture name="floortex" type="2d" file="floor.png"/>\n'
            floor_type = f"real_{os.path.basename(path)}"
            return tex_xml + _material_xml(rng), floor_type, assets

    # --- Opcion 2: builtin (checker / gradient) ---
    builtin_choice = rng.random()
    if builtin_choice < 0.22:
        c1 = rng.uniform(0.1, 0.9, 3)
        c2 = rng.uniform(0.1, 0.9, 3)
        mark = rng.uniform(0.0, 1.0, 3)
        tex_xml = (
            f'    <texture name="floortex" type="2d" builtin="checker" '
            f'width="300" height="300" '
            f'rgb1="{c1[0]:.3f} {c1[1]:.3f} {c1[2]:.3f}" '
            f'rgb2="{c2[0]:.3f} {c2[1]:.3f} {c2[2]:.3f}" '
            f'mark="edge" markrgb="{mark[0]:.3f} {mark[1]:.3f} {mark[2]:.3f}"/>\n'
        )
        return tex_xml + _material_xml(rng), "checker_procedural", assets

    if builtin_choice < 0.40:
        c1 = rng.uniform(0.1, 0.9, 3)
        c2 = rng.uniform(0.1, 0.9, 3)
        tex_xml = (
            f'    <texture name="floortex" type="2d" builtin="gradient" '
            f'width="300" height="300" '
            f'rgb1="{c1[0]:.3f} {c1[1]:.3f} {c1[2]:.3f}" '
            f'rgb2="{c2[0]:.3f} {c2[1]:.3f} {c2[2]:.3f}"/>\n'
        )
        return tex_xml + _material_xml(rng), "gradient_procedural", assets

    # --- Opcion 3: textura numpy (madera / marmol / fractal / liso) ---
    name = str(rng.choice(list(_NUMPY_GENERATORS.keys())))
    rgb = _NUMPY_GENERATORS[name](rng, tex_size)
    assets["floor.png"] = _encode_png(rgb)
    tex_xml = '    <texture name="floortex" type="2d" file="floor.png"/>\n'
    return tex_xml + _material_xml(rng), name, assets
