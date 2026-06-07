"""Construccion del XML de MuJoCo para el trompo + suelo.

Solo se modelan TROMPOS CON PUNTA (cone, acorn, oval): cuerpos que precesan,
nutan y acaban cayendo. NO discos, ruedas ni cuerpos que nunca caen.

Convencion de ejes (ver utils/domain_randomization.py):
  * El cuerpo se construye a lo largo del eje **Z local**.
  * La **punta de contacto** es una esfera de radio 5 mm en el origen del
    cuerpo (z_local = 0). Es el unico apoyo nominal.
  * El cuerpo crece hacia +Z; el CM (inertial) se coloca en (0, 0, com_height).
  * El eje de simetria/spin es Z local => la inercia mayor es Izz.

Como MuJoCo 3.x NO tiene un primitivo "cono", los conos se aproximan apilando
cilindros de radio creciente (frustum). Esto da una silueta conica suficiente
a 224 px y un contacto bien definido. La inercia NO se calcula de los geoms:
se fija explicitamente con <inertial>, asi que la forma solo afecta a la
COLISION y a la APARIENCIA visual.

------------------------------------------------------------------------------
COMO ANADIR MESHES STL REALES (futuro)
------------------------------------------------------------------------------
1) Coloca los STL en assets/meshes/{cone,acorn,oval}.stl con la PUNTA en el
   origen y el eje de simetria a lo largo de +Z (en metros).
2) En _asset_xml() anade, por tipo:
       <mesh name="cone_mesh" file="assets/meshes/cone.stl" scale="1 1 1"/>
   (usa <compiler meshdir="assets/meshes"/> para rutas relativas).
3) En la funcion de geometria del tipo, sustituye los cilindros apilados por:
       <geom type="mesh" mesh="cone_mesh" material="top_mat"
             friction="..." condim="3" contype="1" conaffinity="1"/>
   MANTEN la esfera de la punta (5 mm) como geom de contacto primario: da un
   apoyo estable, mientras que un vertice de malla afilado genera contactos
   inestables.
4) El resto del pipeline (inercia explicita, spin, etc.) no cambia.
"""

from __future__ import annotations

import math

import numpy as np

# Radio de la punta de contacto (m), 5 mm. Fuente unica en
# utils.domain_randomization (modulo mujoco-free) para que la geometria de la
# punta y el brazo de palanca ell = com_height + TIP_RADIUS no se desincronicen.
from utils.domain_randomization import TIP_RADIUS

TOP_TYPES = ("cone", "acorn", "oval")


def _geom(gtype: str, size: str, pos: float, friction: str,
          condim: int = 3, extra: str = "") -> str:
    """Helper para un geom de colision+visual centrado en (0,0,pos)."""
    return (
        f'      <geom type="{gtype}" size="{size}" pos="0 0 {pos:.6f}" '
        f'material="top_mat" friction="{friction}" '
        f'condim="{condim}" contype="1" conaffinity="1" {extra}/>\n'
    )


def _stacked_cone(z0: float, z1: float, r0: float, r1: float,
                  friction: str, n: int = 7) -> str:
    """Frustum (cono truncado) apilando ``n`` cilindros de radio interpolado."""
    out = ""
    for i in range(n):
        f0 = i / n
        f1 = (i + 1) / n
        za = z0 + f0 * (z1 - z0)
        zb = z0 + f1 * (z1 - z0)
        zc = 0.5 * (za + zb)
        hz = 0.5 * (zb - za)
        # radio en el centro del segmento
        r = r0 + (0.5 * (f0 + f1)) * (r1 - r0)
        r = max(r, 0.0012)
        out += _geom("cylinder", f"{r:.6f} {hz:.6f}", zc, friction, condim=3)
    return out


def _cone_geoms(rng: np.random.Generator, friction: str):
    """Trompo conico clasico (peonza): punta abajo, ancho arriba.

    Devuelve (xml, shape_info). shape_info describe la envolvente para colocar
    los marcadores de color sobre la superficie.
    """
    H = float(rng.uniform(0.045, 0.075))
    Rmax = float(rng.uniform(0.018, 0.030))
    body = _stacked_cone(0.0, H, 0.0012, Rmax, friction, n=8)
    # Pequeno vastago/perilla superior (caracter visual de peonza).
    stem_r = float(rng.uniform(0.003, 0.006))
    stem_h = float(rng.uniform(0.006, 0.012))
    body += _geom("cylinder", f"{stem_r:.6f} {stem_h/2:.6f}", H + stem_h / 2,
                  friction, condim=3)
    return body, {"kind": "cone", "H": H, "Rmax": Rmax}


def _acorn_geoms(rng: np.random.Generator, friction: str):
    """Trompo tipo bellota: cono inferior + esfera/ovoide superior."""
    H = float(rng.uniform(0.030, 0.050))      # altura del cono inferior
    Rmid = float(rng.uniform(0.014, 0.022))   # radio donde acopla la esfera
    Rs = float(rng.uniform(Rmid * 0.95, Rmid * 1.25))  # radio de la esfera
    body = _stacked_cone(0.0, H, 0.0012, Rmid, friction, n=7)
    # Esfera superior apoyada sobre el cono.
    sphere_z = H + Rs * 0.55
    body += _geom("sphere", f"{Rs:.6f}", sphere_z, friction, condim=3)
    return body, {"kind": "acorn", "H": H, "Rmid": Rmid, "Rs": Rs,
                  "sphere_z": sphere_z}


def _oval_geoms(rng: np.random.Generator, friction: str):
    """Trompo ovalado: elipsoide alargado (CM alto, inestable) + punta."""
    a = float(rng.uniform(0.013, 0.019))            # semieje transversal
    c = float(a * rng.uniform(1.6, 2.4))            # semieje vertical (alargado)
    stem_top = float(rng.uniform(0.006, 0.012))     # vastago corto sobre la punta
    body = _stacked_cone(0.0, stem_top, 0.0012, a * 0.55, friction, n=4)
    # Elipsoide centrado de modo que su polo inferior quede sobre el vastago.
    zc = stem_top + c * 0.9
    body += _geom("ellipsoid", f"{a:.6f} {a:.6f} {c:.6f}", zc, friction, condim=3)
    return body, {"kind": "oval", "a": a, "c": c, "zc": zc}


_GEOM_BUILDERS = {
    "cone": _cone_geoms,
    "acorn": _acorn_geoms,
    "oval": _oval_geoms,
}


def _light_dir(azimuth_deg: float, elevation_deg: float):
    """Direccion en la que VIAJA la luz (desde la fuente hacia la escena)."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    fx = math.cos(el) * math.cos(az)
    fy = math.cos(el) * math.sin(az)
    fz = math.sin(el)
    return (-fx, -fy, -fz)


# --------------------------------------------------------------------------- #
# Marcadores de color para que VideoMamba observe q y omega                    #
# --------------------------------------------------------------------------- #
# >=3 puntos de color NO colineales, pegados al cuerpo, en posiciones y colores
# aleatorios por episodio. Hacen visualmente OBSERVABLES la orientacion
# (cuaternion) y la velocidad angular: al girar el trompo, los puntos trazan
# circulos cuyo patron determina q y omega. Son geoms SOLO visuales
# (contype=0/conaffinity=0): NO colisionan ni cambian la inercia (que es
# explicita), asi que la dinamica es identica con o sin marcadores.
N_MARKERS_DEFAULT = 4


def _surface_point(shape: dict, u: float, theta: float):
    """Punto (x,y,z) en la superficie del cuerpo y su normal exterior aprox.

    u in [0,1] parametriza la altura/latitud; theta es el azimut.
    """
    kind = shape["kind"]
    ct, st = math.cos(theta), math.sin(theta)
    if kind == "cone":
        H, Rmax = shape["H"], shape["Rmax"]
        z = (0.40 + 0.52 * u) * H            # zona ancha del cono
        r = Rmax * (z / H)
        p = np.array([r * ct, r * st, z])
        n = np.array([ct, st, 0.0])          # normal aprox radial
    elif kind == "acorn":
        Rs, cz = shape["Rs"], shape["sphere_z"]
        phi = (0.28 + 0.55 * u) * math.pi    # evita los polos exactos
        sp, cp = math.sin(phi), math.cos(phi)
        n = np.array([sp * ct, sp * st, cp])
        p = np.array([0.0, 0.0, cz]) + Rs * n
    else:  # oval (elipsoide)
        a, c, zc = shape["a"], shape["c"], shape["zc"]
        phi = (0.25 + 0.55 * u) * math.pi
        sp, cp = math.sin(phi), math.cos(phi)
        p = np.array([a * sp * ct, a * sp * st, zc + c * cp])
        n = np.array([sp * ct / a, sp * st / a, cp / c])
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-9 else np.array([ct, st, 0.0])
    return p, n


def _hsv_to_rgb(h: float, s: float, v: float):
    import colorsys
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


def sample_markers(shape: dict, rng: np.random.Generator,
                   n: int = N_MARKERS_DEFAULT):
    """Muestrea n marcadores no colineales sobre la superficie del cuerpo.

    Devuelve lista de dicts {pos:[x,y,z] (local), rgba:[r,g,b,1], radius}.
    """
    char_r = {"cone": shape.get("Rmax"),
              "acorn": shape.get("Rs"),
              "oval": shape.get("a")}[shape["kind"]]
    rm = float(np.clip(char_r * 0.22, 0.003, 0.006))  # radio del marcador (m)

    # Azimuts repartidos (evita colinealidad) + jitter; alturas aleatorias.
    base_h = float(rng.uniform(0.0, 1.0))
    for _ in range(8):  # reintenta si saliesen casi colineales
        markers = []
        positions = []
        for i in range(n):
            theta = (2.0 * math.pi * i / n) + float(rng.uniform(-0.45, 0.45))
            u = float(np.clip(base_h + rng.uniform(-0.35, 0.35) + 0.0, 0.0, 1.0))
            u = float(rng.uniform(0.0, 1.0)) if i == 0 else u
            p, normal = _surface_point(shape, u, theta)
            p = p + normal * (rm * 0.55)      # protruye hacia afuera
            positions.append(p)
        P = np.array(positions)
        # No colineal: el area del triangulo de los 3 primeros no es ~0.
        v1, v2 = P[1] - P[0], P[2] - P[0]
        if np.linalg.norm(np.cross(v1, v2)) > (char_r ** 2) * 0.05:
            break

    # Colores distintos y saturados (tonos repartidos en el circulo HSV).
    hue0 = float(rng.uniform(0.0, 1.0))
    markers = []
    for i, p in enumerate(positions):
        hue = hue0 + i / n + float(rng.uniform(-0.04, 0.04))
        r, g, b = _hsv_to_rgb(hue, float(rng.uniform(0.75, 1.0)),
                              float(rng.uniform(0.85, 1.0)))
        markers.append({
            "pos": [float(p[0]), float(p[1]), float(p[2])],
            "rgba": [round(r, 4), round(g, 4), round(b, 4), 1.0],
            "radius": round(rm, 5),
        })
    return markers


def _markers_xml(markers) -> str:
    """Geoms de marcador: esferas pequenas SOLO visuales (sin colision)."""
    out = ""
    for i, m in enumerate(markers):
        x, y, z = m["pos"]
        r, g, b, a = m["rgba"]
        out += (
            f'      <geom name="marker_{i}" type="sphere" '
            f'size="{m["radius"]:.5f}" pos="{x:.6f} {y:.6f} {z:.6f}" '
            f'rgba="{r:.4f} {g:.4f} {b:.4f} {a:.2f}" '
            f'contype="0" conaffinity="0" group="2"/>\n'
        )
    return out


def build_top_model_xml(params, floor_asset_xml: str) -> str:
    """Devuelve el XML completo de MuJoCo para un episodio.

    Parameters
    ----------
    params : EpisodeParams
        Parametros muestreados del episodio.
    floor_asset_xml : str
        Snippet ``<texture.../><material name="floor_mat".../>`` generado por
        rendering/floor_generator.py.
    """
    rng = np.random.default_rng(params.geom_seed)

    friction = (
        f"{params.friction_slide:.6f} "
        f"{params.friction_spin:.6f} "
        f"{params.friction_roll:.8f}"
    )

    # Geometria del cuerpo segun el tipo.
    body_geoms, shape = _GEOM_BUILDERS[params.top_type](rng, friction)

    # Marcadores de color (>=3, no colineales, posiciones/colores aleatorios)
    # pegados al cuerpo, para que VideoMamba observe q y omega. Se guardan en
    # params.markers para trazabilidad en la metadata del JSON.
    n_markers = getattr(params, "n_markers", None) or N_MARKERS_DEFAULT
    markers = sample_markers(shape, rng, n=int(n_markers))
    params.markers = markers
    marker_geoms = _markers_xml(markers)

    # Punta de contacto (esfera 5 mm) en el origen del cuerpo. condim=6 para
    # habilitar friccion torsional (spin) y de rodadura en el contacto.
    tip_geom = _geom("sphere", f"{TIP_RADIUS:.6f}", 0.0, friction,
                     condim=6, extra='name="tip"')

    # Pose inicial del cuerpo: la punta (parte baja de la esfera) a position_z.
    # El origen del cuerpo (centro de la esfera punta) va a position_z + R.
    z0 = params.position_z + TIP_RADIUS

    ldx, ldy, ldz = _light_dir(params.light_azimuth, params.light_elevation)
    lr, lg, lb = params.light_rgb
    tr, tg, tb, ta = params.top_rgba

    xml = f"""<mujoco model="spinning_top">
  <compiler angle="radian" autolimits="true"/>

  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81"
          cone="elliptic" impratio="3"/>

  <!-- 'extent' y 'center' pequenos: el trompo mide ~5 cm pero el suelo es
       enorme; esto evita clipping del near-plane al renderizar de cerca. -->
  <statistic center="0 0 0.03" extent="0.45" meansize="0.02"/>

  <visual>
    <global fovy="{params.cam_fov:.3f}" offwidth="1280" offheight="1280"/>
    <map znear="0.005" zfar="50"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.15 0.15 0.15" specular="0 0 0"/>
  </visual>

  <asset>
{floor_asset_xml}
    <material name="top_mat" rgba="{tr:.4f} {tg:.4f} {tb:.4f} {ta:.4f}"
              specular="{params.top_specular:.4f}" shininess="{params.top_shininess:.4f}"
              reflectance="0.05"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1" material="floor_mat"
          friction="{friction}" condim="6" contype="1" conaffinity="1"/>

    <light name="key" directional="true" castshadow="true"
           dir="{ldx:.4f} {ldy:.4f} {ldz:.4f}" pos="0 0 1"
           diffuse="{lr:.4f} {lg:.4f} {lb:.4f}" specular="0.3 0.3 0.3"/>
    <light name="fill" directional="false" castshadow="false"
           pos="0.2 0.2 0.4" diffuse="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>

    <body name="top" pos="0 0 {z0:.6f}">
      <freejoint name="root"/>
      <inertial pos="0 0 {params.com_height:.6f}" mass="{params.mass:.6f}"
                diaginertia="{params.Ixx:.8e} {params.Iyy:.8e} {params.Izz:.8e}"/>
{tip_geom}{body_geoms}{marker_geoms}    </body>
  </worldbody>
</mujoco>
"""
    return xml
