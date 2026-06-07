#!/usr/bin/env python3
"""Aplana los JSON de estados/trayectorias -> tensores .npz para entrenar la UDE.

Preprocesado OFFLINE (no usa MuJoCo). Leer .npz precomputado es mucho mas
rapido que parsear JSON en cada batch (importante en Vast.ai).

Por episodio construye (ver Seccion 10 de la spec):
  X        (T, 14)  = [q(4), omega(3), mass, Ixx, Iyy, Izz, fc, fv, ell]
                       (priors broadcasteados a todos los frames; ell = brazo
                        de palanca punta->CM = com_height + tip_radius)
  Y_next   (T-1, 7) = [q, omega] del frame SIGUIENTE (target autoregresivo)
  dY       (T-1, 7) = (q_{t+1}-q_t)/dt, (omega_{t+1}-omega_t)/dt  (target ODE)
  t        (T,)
  fall_mask(T,)     booleano (fall_detected)
  dt       escalar  (= 0.01 s a 100 Hz)
  top_type str

Normalizacion: mean/std por dimension sobre TODO el dataset, usando SOLO
frames con fall_detected==False. Se guarda en norm_stats.json. Los .npz se
dejan CRUDOS (sin normalizar); el dataloader aplica la normalizacion. El
cuaternion NO se normaliza (mean=0, std=1 en esas dimensiones).

Contrato 14D de priors (Seccion 5.1):
  input(14) = [ q(4), omega(3), mass, Ixx, Iyy, Izz, fc, fv, ell ]
donde ell = pivot_to_com = com_height + tip_radius (distancia punta->CM, el
brazo de palanca del torque gravitatorio). Sin ell, dos trompos con la misma
masa/inercia/friccion pero distinta altura de CM tendrian dinamicas distintas
con input identico -> no identificable para la UDE.
"""

from __future__ import annotations

import os
import json
import glob
import argparse
from typing import Optional

import numpy as np


PRIOR_NAMES = ["mass", "Ixx", "Iyy", "Izz", "fc", "fv", "ell"]
N_PRIORS = len(PRIOR_NAMES)  # 7
X_DIM = 14
QUAT_DIMS = slice(0, 4)  # las primeras 4 dims de X (no se normalizan)


def _episode_arrays(doc: dict):
    """Construye los arrays crudos de un episodio desde su JSON."""
    meta = doc["metadata"]
    frames = doc["frames"]
    phys = meta["physics"]

    q = np.array([f["q"] for f in frames], dtype=np.float32)         # (T,4)
    omega = np.array([f["omega"] for f in frames], dtype=np.float32)  # (T,3)
    t = np.array([f["t"] for f in frames], dtype=np.float32)          # (T,)
    fall = np.array([bool(f["fall_detected"]) for f in frames])       # (T,)

    mass = float(phys["mass"])
    Ixx, Iyy, Izz = [float(v) for v in phys["inertia_diag"]]
    fc = float(phys["coulomb_torque"])
    fv = float(phys["viscous_friction"])
    # ell = distancia punto de contacto (punta) -> CM = brazo del torque
    # gravitatorio = com_height + tip_radius. Fallback para JSON antiguos.
    ell = phys.get("pivot_to_com")
    if ell is None:
        ch = phys.get("com_height")
        ell = (float(ch) + 0.005) if ch is not None else 0.0
    ell = float(ell)
    priors = np.array([mass, Ixx, Iyy, Izz, fc, fv, ell], dtype=np.float32)  # (7,)

    T = len(frames)
    priors_b = np.broadcast_to(priors, (T, N_PRIORS))                 # (T,7)
    X = np.concatenate([q, omega, priors_b], axis=1).astype(np.float32)  # (T,14)

    # dt desde el sample rate de la metadata (fallback: mediana de diffs).
    rate = meta.get("simulation", {}).get("state_sample_rate_hz", 100)
    dt = float(1.0 / rate) if rate else float(np.median(np.diff(t)) if T > 1 else 0.01)

    return X, q, omega, t, fall, dt, meta.get("top_type", "unknown"), priors


def _cut_index(fall: np.ndarray, cut_on_fall: bool) -> int:
    """Indice de corte: primer frame con fall_detected==True (o T si no cae)."""
    if not cut_on_fall:
        return len(fall)
    idx = np.argmax(fall) if fall.any() else len(fall)
    return int(idx) if fall.any() else len(fall)


def process_episode(doc, out_dir, cut_on_fall, target):
    X, q, omega, t, fall, dt, top_type, _ = _episode_arrays(doc)
    cut = _cut_index(fall, cut_on_fall)
    cut = max(cut, 2)  # necesitamos >=2 frames para derivadas
    cut = min(cut, len(X))

    X = X[:cut]
    q = q[:cut]
    omega = omega[:cut]
    t = t[:cut]
    fall = fall[:cut]

    YQO = np.concatenate([q, omega], axis=1).astype(np.float32)  # (T,7)
    Y_next = YQO[1:]                                              # (T-1,7)
    dY = ((YQO[1:] - YQO[:-1]) / dt).astype(np.float32)          # (T-1,7)

    ep_id = doc["metadata"].get("episode_id", 0)
    npz_path = os.path.join(out_dir, f"ep_{int(ep_id):04d}.npz")

    save_kw = dict(X=X, t=t, fall_mask=fall, dt=np.float32(dt), top_type=top_type)
    if target in ("next", "both"):
        save_kw["Y_next"] = Y_next
    if target in ("deriv", "both"):
        save_kw["dY"] = dY
    np.savez_compressed(npz_path, **save_kw)

    n_valid = int(np.sum(~fall))
    return npz_path, X, Y_next, dY, fall, top_type, n_valid


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="JSON de estados -> tensores UDE (.npz)")
    p.add_argument("--in", dest="in_dir", required=True,
                   help="carpeta con JSONs (dataset/states o dataset/trajectories)")
    p.add_argument("--out", dest="out_dir", required=True,
                   help="carpeta de salida de los .npz (dataset/ude)")
    p.add_argument("--cut-on-fall", action=argparse.BooleanOptionalAction,
                   default=True, help="recortar frames posteriores a la caida")
    p.add_argument("--target", choices=["next", "deriv", "both"], default="both")
    p.add_argument("--skip-invalid", action="store_true", default=False,
                   help="omitir episodios con valid==false")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "ep_*.json")))
    if not files:
        print(f"[ERROR] no se encontraron ep_*.json en {args.in_dir}")
        return 1

    # Acumuladores para normalizacion (solo frames NO caidos) y rangos.
    x_sum = np.zeros(X_DIM, np.float64)
    x_sqsum = np.zeros(X_DIM, np.float64)
    x_count = 0
    dy_sum = np.zeros(7, np.float64)
    dy_sqsum = np.zeros(7, np.float64)
    dy_count = 0
    prior_min = np.full(N_PRIORS, np.inf)
    prior_max = np.full(N_PRIORS, -np.inf)

    manifest_eps = []
    n_done = 0

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if args.skip_invalid and not doc.get("metadata", {}).get("valid", True):
            continue

        npz_path, X, Y_next, dY, fall, top_type, n_valid = process_episode(
            doc, args.out_dir, args.cut_on_fall, args.target)

        # Stats SOLO con frames no caidos.
        keep = ~fall
        if keep.any():
            Xk = X[keep].astype(np.float64)
            x_sum += Xk.sum(axis=0)
            x_sqsum += (Xk ** 2).sum(axis=0)
            x_count += Xk.shape[0]
            # dY: pares cuyo frame de origen no esta caido.
            keep_pairs = keep[:-1]
            if keep_pairs.any():
                dYk = dY[keep_pairs].astype(np.float64)
                dy_sum += dYk.sum(axis=0)
                dy_sqsum += (dYk ** 2).sum(axis=0)
                dy_count += dYk.shape[0]

        priors = X[0, 7:7 + N_PRIORS]
        prior_min = np.minimum(prior_min, priors)
        prior_max = np.maximum(prior_max, priors)

        manifest_eps.append({
            "episode_id": int(doc["metadata"].get("episode_id", 0)),
            "file": os.path.basename(npz_path),
            "n_frames_total": int(X.shape[0]),
            "n_frames_valid": int(n_valid),
            "top_type": top_type,
            "valid": bool(doc.get("metadata", {}).get("valid", True)),
        })
        n_done += 1

    # --- norm_stats.json ---
    x_mean = (x_sum / max(x_count, 1)).astype(np.float64)
    x_var = (x_sqsum / max(x_count, 1)) - x_mean ** 2
    x_std = np.sqrt(np.clip(x_var, 1e-12, None))
    # El cuaternion NO se normaliza.
    x_mean[QUAT_DIMS] = 0.0
    x_std[QUAT_DIMS] = 1.0

    dy_mean = (dy_sum / max(dy_count, 1)).astype(np.float64)
    dy_var = (dy_sqsum / max(dy_count, 1)) - dy_mean ** 2
    dy_std = np.sqrt(np.clip(dy_var, 1e-12, None))

    norm_stats = {
        "note": ("Aplicar (x - mean) / std al construir el batch. Dims 0-3 de X "
                 "son el cuaternion y NO se normalizan (mean=0, std=1). Stats "
                 "calculadas solo con frames fall_detected==False."),
        "X_dim_layout": ["qw", "qx", "qy", "qz", "wx", "wy", "wz",
                         "mass", "Ixx", "Iyy", "Izz", "fc", "fv", "ell"],
        "n_frames_used": int(x_count),
        "X": {"mean": x_mean.tolist(), "std": x_std.tolist()},
        "dY": {"mean": dy_mean.tolist(), "std": dy_std.tolist()},
    }
    with open(os.path.join(args.out_dir, "norm_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=2)

    # --- manifest.json ---
    manifest = {
        "n_episodes": n_done,
        "source": args.in_dir,
        "target": args.target,
        "cut_on_fall": bool(args.cut_on_fall),
        "physical_ranges": {
            name: [float(prior_min[i]), float(prior_max[i])]
            for i, name in enumerate(PRIOR_NAMES)
        },
        "episodes": manifest_eps,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[PREPARE] {n_done} episodios -> {args.out_dir} | "
          f"frames_norm={x_count} | target={args.target} | "
          f"cut_on_fall={args.cut_on_fall}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
