"""Condiciones iniciales del trompo (qpos / qvel) y helpers de orientacion.

El trompo NO cae del cielo: nace practicamente apoyado sobre su punta
(``position_z`` = 1-5 mm sobre el suelo), casi vertical, con un pequeno tilt y
girando alrededor de su eje de simetria (Z local).

Recordatorio MuJoCo (freejoint):
  qpos = [x, y, z, qw, qx, qy, qz]   (cuaternion Hamilton)
  qvel = [vx, vy, vz, wx, wy, wz]
         * lineal (vx,vy,vz) en el frame del MUNDO
         * angular (wx,wy,wz) en el frame LOCAL del cuerpo
           (por eso un spin puro sobre el eje de simetria Z local es [0,0,spin])
"""

from __future__ import annotations

import math

import numpy as np
import mujoco

from .top_model import TIP_RADIUS

# La punta nace JUSTO tocando el suelo (penetracion 0 => fuerza inicial 0, sin
# "patada" que la lance). La fuerza de contacto sube suave hasta mg en ~1 ms
# (contacto criticamente amortiguado). has_contact ya es True en t=0 gracias al
# margen de deteccion del contacto (TIP_CONTACT_MARGIN en top_model.py).
REST_PENETRATION = 0.0


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _axis_angle_to_quat(axis, angle: float) -> np.ndarray:
    """Eje-angulo -> cuaternion Hamilton [w,x,y,z]."""
    ax = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(ax)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = ax / n
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), ax[0] * s, ax[1] * s, ax[2] * s])


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rota el vector v (en body) al mundo usando q (Hamilton, body->world)."""
    w, x, y, z = q
    # R @ v sin construir R explicitamente
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)


def get_symmetry_axis_world(q: np.ndarray) -> np.ndarray:
    """Eje de simetria (Z local) expresado en el mundo, dado el cuaternion."""
    return _quat_rotate(np.asarray(q, dtype=np.float64),
                        np.array([0.0, 0.0, 1.0]))


def angle_from_vertical_deg(q: np.ndarray) -> float:
    """Angulo (grados) entre el eje de simetria del trompo y la vertical +Z.

    Aqui se calcula ``angle_from_vertical``: arccos del componente Z del eje de
    simetria (Z local) rotado al mundo.
    """
    sym = get_symmetry_axis_world(q)
    cos_ang = float(np.clip(sym[2], -1.0, 1.0))
    return math.degrees(math.acos(cos_ang))


def set_initial_conditions(model: mujoco.MjModel, data: mujoco.MjData, params,
                           rng: np.random.Generator | None = None) -> None:
    """Coloca el trompo casi apoyado, casi vertical, con spin y tilt."""
    if rng is None:
        rng = np.random.default_rng(params.seed + 777)

    mujoco.mj_resetData(model, data)

    qadr = int(model.jnt_qposadr[0])  # primer (y unico) freejoint
    vadr = int(model.jnt_dofadr[0])

    # --- Posicion: la punta APOYADA en el suelo (no cae del cielo). El centro
    #     de la esfera-punta va a TIP_RADIUS sobre el suelo, menos una micro-
    #     penetracion para que el contacto este activo desde t=0 sin rebote.
    #     position_z=0 (apoyado); se deja por si se quisiera elevar. ---
    data.qpos[qadr + 0] = 0.0
    data.qpos[qadr + 1] = 0.0
    data.qpos[qadr + 2] = params.position_z + TIP_RADIUS - REST_PENETRATION

    # --- Orientacion: identidad (eje de simetria Z local = vertical) + tilt
    #     pequeno alrededor de un eje horizontal aleatorio. ---
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    tilt_axis = (math.cos(phi), math.sin(phi), 0.0)
    q_tilt = _axis_angle_to_quat(tilt_axis, params.tilt)
    data.qpos[qadr + 3: qadr + 7] = q_tilt

    # --- Velocidad: arrancar en PRECESION ESTACIONARIA (no "soltar desde el
    #     reposo"). Soltar con spin puro induce una nutacion grande que hace
    #     SALTAR la punta (contacto intermitente), sobre todo en oval/acorn con
    #     CM alto. En precesion estacionaria el eje describe un cono sin nutar,
    #     asi que la punta queda CLAVADA y el contacto es continuo.
    #
    #     Aproximacion de trompo rapido (raiz lenta):
    #         phidot = m*g*l / (I_spin * spin)      (precesion sobre +Z mundo)
    #     omega_world = phidot * z_mundo + spin * eje_simetria
    #     Se convierte a frame LOCAL (qvel angular del freejoint es local). ---
    spin = float(params.spin)
    Ispin = float(params.Izz)                  # inercia del eje de spin (Z local)
    ell = float(params.com_height + TIP_RADIUS)
    e_sym = get_symmetry_axis_world(q_tilt)    # eje de simetria en el mundo
    if abs(spin) > 1e-9 and Ispin > 1e-12:
        phidot = (params.mass * 9.81 * ell) / (Ispin * spin)
    else:
        phidot = 0.0
    omega_world = phidot * np.array([0.0, 0.0, 1.0]) + spin * e_sym
    omega_local = _quat_rotate(_quat_conjugate(q_tilt), omega_world)
    data.qvel[vadr + 0: vadr + 3] = 0.0                 # lineal mundo = 0
    data.qvel[vadr + 3: vadr + 6] = omega_local         # angular local

    mujoco.mj_forward(model, data)
