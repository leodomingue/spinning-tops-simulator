"""Validacion fisica de un JSON de estados (offline, sin MuJoCo).

``validate_physics(json_path)`` comprueba una lista de invariantes fisicas.
Si alguna invariante "dura" falla, marca ``metadata.valid = false`` en el JSON
(reescribiendolo) pero NO descarta el episodio. Devuelve ``(valid, issues)``.

Invariantes (Seccion 12 de la spec):
  1. Energia total (rotacional + potencial) decae de forma aprox. monotona
     (ruido numerico tolerado ~0.1%).               [SOFT]
  2. Sin penetracion: z del CM >= -0.001 para todo t. [HARD]
  3. Cuaternion normalizado: ||q|| = 1 +/- 1e-6.      [HARD]
  4. El trompo no nace cayendo: punta <= 5 mm en t=0. [HARD]
  5. Duracion razonable: 3-30 s.                      [SOFT]
  6. fall_detected es monotono (nunca true -> false). [HARD]
  7. Si fall_time != null, el frame en fall_time cumple angle>70 o ||w||<0.1. [HARD]
  8. Sin NaN/Inf en los estados.                      [HARD]

SOFT vs HARD: las invariantes SOFT se reportan y loggean como warning pero no
ponen valid=false por si solas, porque pueden activarse en episodios fisicos
legitimos (p.ej. un "oval" inestable que cae en <3 s, o el transitorio de
energia al apoyarse en el suelo en los primeros ms). Las HARD si fuerzan
valid=false. Esto se documenta en el README.
"""

from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Cuaternion Hamilton [w,x,y,z] -> matriz de rotacion 3x3 (body->world)."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def validate_physics(json_path: str, rewrite: bool = True) -> Tuple[bool, List[str]]:
    """Valida un JSON de estados. Devuelve (valid, issues)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    frames = data.get("frames", [])
    issues: List[str] = []
    hard_fail = False

    if len(frames) < 2:
        issues.append("HARD: episodio con <2 frames")
        _finish(data, json_path, False, rewrite)
        return False, issues

    phys = meta.get("physics", {})
    mass = float(phys.get("mass", 0.1))
    Ixx, Iyy, Izz = [float(v) for v in phys.get("inertia_diag", [1e-4, 1e-4, 1e-4])]
    g = 9.81

    ts = np.array([fr["t"] for fr in frames], dtype=np.float64)
    qs = np.array([fr["q"] for fr in frames], dtype=np.float64)
    ws = np.array([fr["omega"] for fr in frames], dtype=np.float64)
    xs = np.array([fr["x"] for fr in frames], dtype=np.float64)
    angles = np.array([fr["angle_from_vertical"] for fr in frames], dtype=np.float64)
    falls = np.array([bool(fr["fall_detected"]) for fr in frames])

    # 8. NaN / Inf
    for name, arr in (("q", qs), ("omega", ws), ("x", xs)):
        if not np.all(np.isfinite(arr)):
            issues.append(f"HARD: valores no finitos en '{name}'")
            hard_fail = True

    # 3. Cuaternion normalizado
    qnorm = np.linalg.norm(qs, axis=1)
    if np.any(np.abs(qnorm - 1.0) > 1e-6):
        worst = float(np.max(np.abs(qnorm - 1.0)))
        issues.append(f"HARD: cuaternion no normalizado (max |‖q‖-1|={worst:.2e})")
        hard_fail = True

    # 2. Penetracion (proxy: z del CM nunca por debajo de -0.001 m)
    min_cm_z = float(np.min(xs[:, 2]))
    if min_cm_z < -0.001:
        issues.append(f"HARD: penetracion (min z_CM={min_cm_z:.4f} < -0.001)")
        hard_fail = True

    # 4. No nace cayendo: punta <= 5 mm en t=0
    init_pos = phys.get("initial_position", [0, 0, 0])
    tip0 = float(init_pos[2])
    if tip0 > 0.005 + 1e-9:
        issues.append(f"HARD: nace alto (punta t=0 = {tip0*1000:.1f} mm > 5 mm)")
        hard_fail = True

    # 6. fall_detected monotono (latch)
    if np.any((~falls[1:]) & (falls[:-1])):
        issues.append("HARD: fall_detected no es monotono (paso de true a false)")
        hard_fail = True

    # 7. Consistencia de fall_time
    fall_time = meta.get("fall_time", None)
    if fall_time is not None:
        idx = int(np.argmin(np.abs(ts - float(fall_time))))
        wn = float(np.linalg.norm(ws[idx]))
        if not (angles[idx] > 70.0 or wn < 0.1):
            issues.append(
                f"HARD: fall_time inconsistente en t={fall_time} "
                f"(angle={angles[idx]:.1f}, ‖w‖={wn:.3f})"
            )
            hard_fail = True

    # 1. Energia ~ monotona decreciente (SOFT)
    energies = np.empty(len(frames))
    for i in range(len(frames)):
        R = _quat_to_rotmat(qs[i])
        wb = R.T @ ws[i]  # omega en frame del cuerpo
        ke = 0.5 * (Ixx * wb[0] ** 2 + Iyy * wb[1] ** 2 + Izz * wb[2] ** 2)
        pe = mass * g * xs[i, 2]
        energies[i] = ke + pe
    e0 = max(abs(energies[0]), 1e-12)
    incr = np.diff(energies)
    # cuantos saltos suben mas del 0.1% de la energia inicial
    bad = int(np.sum(incr > 1e-3 * e0))
    if bad > max(3, int(0.05 * len(frames))):
        issues.append(
            f"SOFT: energia no monotona ({bad}/{len(frames)-1} saltos > 0.1% E0)"
        )

    # 5. Duracion razonable (SOFT)
    duration = float(ts[-1] - ts[0])
    if duration < 3.0 or duration > 30.0:
        issues.append(f"SOFT: duracion {duration:.2f}s fuera de [3, 30]s")

    valid = not hard_fail
    _finish(data, json_path, valid, rewrite)

    if issues:
        tag = "INVALID" if not valid else "ok-con-avisos"
        print(f"[VALIDATION:{tag}] {json_path}")
        for it in issues:
            print(f"    - {it}")
    return valid, issues


def _finish(data: dict, json_path: str, valid: bool, rewrite: bool) -> None:
    """Reescribe metadata.valid si procede."""
    if rewrite:
        data.setdefault("metadata", {})["valid"] = bool(valid)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        v, iss = validate_physics(p)
        print(f"{p}: valid={v}, issues={len(iss)}")
