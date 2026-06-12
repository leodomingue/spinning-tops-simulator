"""Domain Randomization (DR): muestreo de TODOS los parametros por episodio.

Todo se muestrea con un ``numpy.random.default_rng(seed)`` para que cada
episodio sea 100% reproducible a partir de su semilla.

------------------------------------------------------------------------------
NOTA IMPORTANTE SOBRE EL EJE DE SIMETRIA (Y vs Z)
------------------------------------------------------------------------------
La especificacion menciona en algunos sitios el "eje Y local" como eje de
simetria/spin, pero la geometria que describe (``inertial pos="0 0 {com_height}"``,
punta abajo, ``position_z`` = altura de la punta) esta definida a lo largo del
eje **Z local**. Ambas convenciones no pueden ser ciertas a la vez.

Para que la fisica sea CORRECTA y coherente con la geometria (cono/elipsoide
construidos a lo largo de Z, CM desplazado en +Z sobre la punta), este
simulador usa de forma consistente el **eje Z local** como eje de
simetria/spin:

  * El momento de inercia mayor (eje de spin) es ``Izz``.
  * El spin inicial se aplica alrededor del eje Z local.
  * ``angle_from_vertical`` es el angulo entre el eje Z local (rotado por q)
    y la vertical del mundo (+Z).

El vector de inercia que se guarda en el JSON sigue siendo
``inertia_diag = [Ixx, Iyy, Izz]`` y el contrato 14D de la UDE
``[..., Ixx, Iyy, Izz, ...]`` se respeta tal cual; simplemente el valor grande
vive en ``Izz`` en vez de ``Iyy``. Esto se documenta tambien en el README.

------------------------------------------------------------------------------
RESTRICCION FISICA DE LA INERCIA (triangle inequality)
------------------------------------------------------------------------------
MuJoCo exige que la inercia diagonal cumpla A+B>=C en sus tres permutaciones,
o el modelo NO compila. Para un trompo simetrico (Ixx~Iyy=It transversal,
Izz=Is spin) esto implica Is <= 2*It. Como ademas la spec pide que el eje de
spin tenga la inercia mayor (Is>It), muestreamos Is = It * U(1.05, 1.85),
garantizando It < Is < 2*It (cuerpo "oblato" tipo trompo/peonza).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Tipos de trompo soportados (solo trompos CON PUNTA).
TOP_TYPES = ("cone", "acorn", "oval")

# Radio de la punta de contacto (m). FUENTE UNICA: physics/top_model.py la
# importa de aqui (este modulo es mujoco-free) para que la geometria de la
# punta y el brazo de palanca ell = com_height + TIP_RADIUS nunca se
# desincronicen.
TIP_RADIUS = 0.005


def kelvin_to_rgb(temp_k: float) -> np.ndarray:
    """Aproximacion de temperatura de color (K) -> ganancia RGB (~1.0 media).

    Basado en la aproximacion de Tanner Helland, normalizada a ~1.0 para no
    cambiar el brillo global, solo el balance R/B.
    """
    temp = np.clip(temp_k, 1000.0, 40000.0) / 100.0
    # Rojo
    if temp <= 66:
        r = 255.0
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592)
    # Verde
    if temp <= 66:
        g = 99.4708025861 * np.log(temp) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
    # Azul
    if temp >= 66:
        b = 255.0
    elif temp <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(temp - 10) - 305.0447927307
    rgb = np.clip(np.array([r, g, b], dtype=np.float64), 0, 255) / 255.0
    # Normalizar para que la media sea ~1.0 (ganancia de balance de blancos).
    rgb = rgb / max(rgb.mean(), 1e-6)
    return rgb


@dataclass
class EpisodeParams:
    """Todos los parametros aleatorios de un episodio."""

    episode_id: int
    seed: int
    mode: str
    top_type: str

    # --- Fisica del trompo ---
    mass: float
    Ixx: float
    Iyy: float
    Izz: float
    spin: float                 # rad/s, con signo
    tilt: float                 # rad
    friction_slide: float
    friction_spin: float
    friction_roll: float
    coulomb_torque: float       # N*m  (fc, prior UDE)
    viscous_friction: float     # N*m/(rad/s) (fv, prior UDE)
    position_z: float           # altura de la PUNTA sobre el suelo en t=0 (m)

    # --- Geometria (derivada del tipo) ---
    com_height: float           # altura del CM sobre la punta (m)
    geom_seed: int              # semilla para variaciones de geometria

    # --- Camara (DR visual) ---
    cam_distance: float
    cam_azimuth: float
    cam_elevation: float
    cam_fov: float
    cam_lookat: Tuple[float, float, float]

    # --- Iluminacion (DR visual) ---
    light_dir_intensity: float
    light_rgb: Tuple[float, float, float]
    light_azimuth: float
    light_elevation: float

    # --- Apariencia del trompo ---
    top_rgba: Tuple[float, float, float, float]
    top_specular: float
    top_shininess: float

    # --- Suelo (lo rellena floor_generator) ---
    floor_type: str = "pending"

    # --- Marcadores de color (los rellena physics/top_model.build_top_model_xml).
    # Lista de {pos:[x,y,z] (local), rgba:[r,g,b,1], radius}. >=3 no colineales.
    markers: list | None = None

    def physics_metadata(self) -> dict:
        """Bloque ``physics`` del JSON de estados (constantes por episodio)."""
        return {
            "mass": float(self.mass),
            "inertia_diag": [float(self.Ixx), float(self.Iyy), float(self.Izz)],
            # --- Brazo de palanca del torque gravitatorio (prior UDE) ---
            # com_height : distancia origen del cuerpo (centro de la esfera
            #              punta) -> CM.
            # pivot_to_com (ell) : distancia PUNTO DE CONTACTO -> CM
            #              = com_height + tip_radius. ES la que entra en
            #              tau = m*g*ell*sin(theta) y la que usa la UDE (14D).
            "com_height": float(self.com_height),
            "tip_radius": float(TIP_RADIUS),
            "pivot_to_com": float(self.com_height + TIP_RADIUS),
            "friction_slide": float(self.friction_slide),
            "friction_spin": float(self.friction_spin),
            "friction_roll": float(self.friction_roll),
            "coulomb_torque": float(self.coulomb_torque),
            "viscous_friction": float(self.viscous_friction),
            "initial_spin": float(self.spin),
            "initial_tilt_rad": float(self.tilt),
            "initial_position": [0.0, 0.0, float(self.position_z)],
        }


# Alturas tipicas del CM sobre la punta por tipo (m). El "oval" tiene el CM mas
# alto => menos estable => se cae con mas facilidad (objetivo de la spec).
_COM_HEIGHT_RANGE = {
    "cone": (0.020, 0.030),
    "acorn": (0.024, 0.034),
    "oval": (0.034, 0.050),
}

# Rango de tilt por tipo (rad). El oval admite tilt mayor (mas inestable).
_TILT_RANGE = {
    "cone": (0.01, 0.06),
    "acorn": (0.01, 0.06),
    "oval": (0.03, 0.08),
}


def sample_episode_params(
    seed: int,
    episode_id: int,
    mode: str,
    top_type_arg: str = "random",
) -> EpisodeParams:
    """Muestrea los parametros de un episodio de forma reproducible.

    Parameters
    ----------
    seed : int
        Semilla del episodio (reproducibilidad total).
    episode_id : int
        Indice del episodio.
    mode : str
        "video" o "trajectories".
    top_type_arg : str
        "cone" | "acorn" | "oval" | "random".
    """
    rng = np.random.default_rng(seed)

    # --- Tipo de trompo ---
    if top_type_arg == "random":
        top_type = str(rng.choice(TOP_TYPES))
    else:
        assert top_type_arg in TOP_TYPES, f"top-type invalido: {top_type_arg}"
        top_type = top_type_arg

    # --- Masa (50-300 g) ---
    mass = float(rng.uniform(0.05, 0.50))

    # Radio del cuerpo del trompo en rango realista [1 cm, 4 cm]
    body_radius = float(rng.uniform(0.01, 0.04))

    # Factor de forma para sólidos de revolución (cono~0.3, disco~0.5)
    shape_factor = float(rng.uniform(0.25, 0.50))

    # --- Inercia (ver nota de cabecera). Izz = eje de spin (mayor). ---
    It = shape_factor * mass * body_radius**2           # transversal base
    Ixx = It
    Iyy = It * float(rng.uniform(0.86, 1.20))      # leve asimetria => nutacion
    ratio = float(rng.uniform(1.05, 1.85))         # oblatez (spin > transversal)
    Izz = max(Ixx, Iyy) * ratio 
    Izz = float(np.clip(Izz, 2e-5, 1e-3))
    # Garantizar triangle inequality con margen (Ixx + Iyy >= Izz).
    Izz = float(min(Izz, 0.98 * (Ixx + Iyy)))

    # --- Geometria derivada del tipo ---
    clo, chi = _COM_HEIGHT_RANGE[top_type]
    com_height = float(rng.uniform(clo, chi))
    geom_seed = int(rng.integers(0, 2**31 - 1))

    # --- Spin inicial (rad/s) con signo aleatorio ---
    # Fisica correcta: omega_crit = 2*sqrt(m*g*l*It) / Izz   (trompo prolato)
    pivot_to_com = com_height + TIP_RADIUS
    omega_crit = 2.0 * np.sqrt(mass * 9.81 * pivot_to_com * It) / Izz
    omega_min = max(80.0, 2.2 * omega_crit)          # nunca bajo de 80, y siempre > crit
    omega_max = omega_min + 300.0                     # rango razonable por encima
    spin = float(rng.uniform(omega_min, omega_max)) * float(rng.choice([-1.0, 1.0]))

    # --- Tilt inicial (rad) ---
    tlo, thi = _TILT_RANGE[top_type]
    tilt = float(rng.uniform(tlo, thi))

    # --- Fricciones (NO hardcodeadas) ---
    friction_slide = float(rng.uniform(0.2, 1.2))
    friction_spin = float(rng.uniform(0.0001, 0.001))
    friction_roll = float(rng.uniform(1e-5, 1e-4))
    coulomb_torque = float(rng.uniform(1e-6, 5e-5))
    viscous_friction = float(rng.uniform(1e-7, 1e-5))

    position_z = 0.0

    

    # --- Camara (Seccion 7.1) ---
    cam_distance = float(rng.uniform(0.10, 0.20))
    cam_azimuth = float(rng.uniform(0.0, 360.0))
    cam_elevation = float(-rng.uniform(20.0, 60.0))  # MuJoCo: elevacion negativa mira hacia abajo
    cam_fov = float(rng.uniform(50.0, 70.0))
    lookat_z = position_z + com_height + TIP_RADIUS  # ~2 cm sobre la base
    cam_lookat = (
        float(rng.uniform(-0.005, 0.005)),
        float(rng.uniform(-0.005, 0.005)),
        lookat_z,
    )

    # --- Iluminacion (Seccion 7.2) ---
    light_dir_intensity = float(rng.uniform(0.6, 1.0))
    light_temp = float(rng.uniform(4500.0, 7500.0))
    light_rgb = tuple(float(c * light_dir_intensity) for c in kelvin_to_rgb(light_temp))
    light_azimuth = float(rng.uniform(0.0, 360.0))
    light_elevation = float(rng.uniform(30.0, 80.0))

    # --- Apariencia del trompo (Seccion 7.3) ---
    top_rgba = (
        float(rng.uniform(0.2, 1.0)),
        float(rng.uniform(0.2, 1.0)),
        float(rng.uniform(0.2, 1.0)),
        1.0,
    )
    top_specular = float(rng.uniform(0.1, 0.8))
    top_shininess = float(rng.uniform(0.1, 0.9))

    return EpisodeParams(
        episode_id=episode_id,
        seed=seed,
        mode=mode,
        top_type=top_type,
        mass=mass,
        Ixx=Ixx,
        Iyy=Iyy,
        Izz=Izz,
        spin=spin,
        tilt=tilt,
        friction_slide=friction_slide,
        friction_spin=friction_spin,
        friction_roll=friction_roll,
        coulomb_torque=coulomb_torque,
        viscous_friction=viscous_friction,
        position_z=position_z,
        com_height=com_height,
        geom_seed=geom_seed,
        cam_distance=cam_distance,
        cam_azimuth=cam_azimuth,
        cam_elevation=cam_elevation,
        cam_fov=cam_fov,
        cam_lookat=cam_lookat,
        light_dir_intensity=light_dir_intensity,
        light_rgb=light_rgb,
        light_azimuth=light_azimuth,
        light_elevation=light_elevation,
        top_rgba=top_rgba,
        top_specular=top_specular,
        top_shininess=top_shininess,
    )
