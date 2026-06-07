#!/usr/bin/env python3
"""Test de humo end-to-end: 1 secuencia de video + 1 trayectoria + test UDE.

Hace:
  1. Genera 1 episodio VIDEO (frames JPG 30 FPS + estados 100 Hz).
  2. Genera 1 episodio TRAYECTORIAS (estados 100 Hz, sin render).
  3. Valida la fisica de ambos e imprime un resumen (fall_time, frames validos).
  4. Ejecuta prepare_ude_dataset.py sobre los episodios, carga los .npz y
     comprueba los shapes del contrato UDE: X (T,14), Y_next (T-1,7), dY (T-1,7).

Para que sea rapido se limita la duracion maxima (MAX_T) a 6 s.
"""

from __future__ import annotations

import os
import sys
import json
import glob
import shutil

import numpy as np

import main as sim_main
import prepare_ude_dataset
from utils.validation import validate_physics

OUT = "test_output"


def _summary(json_path: str, label: str):
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    m, fr = doc["metadata"], doc["frames"]
    n_valid = sum(1 for x in fr if not x["fall_detected"])
    print(f"\n=== Resumen {label}: {json_path} ===")
    print(f"  top_type        : {m['top_type']}")
    print(f"  floor_type      : {m['floor_type']}")
    print(f"  mass / inertia  : {m['physics']['mass']:.4f} kg / "
          f"{m['physics']['inertia_diag']}")
    print(f"  ell (punta->CM) : {m['physics'].get('pivot_to_com')} m  "
          f"(com_height={m['physics'].get('com_height')}, "
          f"tip_radius={m['physics'].get('tip_radius')})")
    print(f"  initial_spin    : {m['physics']['initial_spin']:.2f} rad/s")
    print(f"  duration        : {m['simulation']['duration_seconds']} s")
    print(f"  total_state_frm : {m['simulation']['total_state_frames']} (100 Hz)")
    print(f"  video_fps       : {m['simulation']['video_fps']}")
    print(f"  fall_time       : {m['fall_time']}")
    print(f"  post_fall_secs  : {m['simulation'].get('post_fall_seconds')}")
    print(f"  frames validos  : {n_valid} (fall_detected==False)")
    print(f"  n markers       : {len(m.get('markers', []))} (puntos de color)")
    from collections import Counter
    print(f"  motion_state    : {dict(Counter(f['motion_state'] for f in fr))}")
    print(f"  metadata.valid  : {m['valid']}")
    # Si cae, la duracion debe ser ~ fall_time + post_fall.
    if m["fall_time"] is not None and m["simulation"].get("post_fall_seconds"):
        expected = m["fall_time"] + m["simulation"]["post_fall_seconds"]
        got = m["simulation"]["duration_seconds"]
        print(f"  check post-fall : dur={got}s ~ fall+post={expected:.2f}s "
              f"-> {'OK' if abs(got - expected) < 0.2 else 'REVISAR'}")
    return doc


def _check_npz(npz_path: str, label: str):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]
    print(f"\n=== UDE tensors {label}: {npz_path} ===")
    print(f"  X       : {X.shape}  (esperado (T,14))")
    assert X.shape[1] == 14, "X debe ser (T,14) segun el contrato UDE (incl. ell)"
    if "Y_next" in z:
        print(f"  Y_next  : {z['Y_next'].shape}  (esperado (T-1,7))")
        assert z["Y_next"].shape[1] == 7
    if "dY" in z:
        print(f"  dY      : {z['dY'].shape}  (esperado (T-1,7))")
        assert z["dY"].shape[1] == 7
    print(f"  t       : {z['t'].shape}")
    print(f"  fall_mask: {z['fall_mask'].shape}  (validos={int((~z['fall_mask']).sum())})")
    print(f"  dt      : {float(z['dt'])}")
    print(f"  top_type: {str(z['top_type'])}")

    # "test UDE": comprobar que el vector 14D es consumible (forward trivial).
    layout = ["qw", "qx", "qy", "qz", "wx", "wy", "wz",
              "mass", "Ixx", "Iyy", "Izz", "fc", "fv", "ell"]
    x0 = X[0]
    print("  primer input(14) =")
    for name, val in zip(layout, x0):
        print(f"      {name:>5} = {val: .6g}")
    # forward dummy: una capa lineal random 14->7 para verificar dimensiones.
    rng = np.random.default_rng(0)
    W = rng.normal(size=(14, 7)).astype(np.float32)
    y = X.astype(np.float32) @ W
    print(f"  forward dummy 13->7 OK: salida {y.shape}")
    return z


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)

    # Limitar duracion para que el test sea rapido.
    sim_main.MAX_T = 6.0
    print(f"[TEST] MAX_T limitado a {sim_main.MAX_T}s para el test.")

    # 1) Episodio VIDEO (seed elegido para que el trompo CAIGA dentro del cap,
    #    y asi ejercitar fall_detected/fall_time y el post-fall recording).
    print("\n########## Generando episodio VIDEO ##########")
    sim_main.main(["--mode", "video", "--n", "1", "--out", OUT,
                   "--top-type", "cone", "--resolution", "224",
                   "--subframes", "2", "--seed", "13",
                   "--no-cooling-sleep", "--no-validate"])

    # 2) Episodio TRAYECTORIAS (seed elegido para que caiga -> cut-on-fall real).
    print("\n########## Generando episodio TRAYECTORIAS ##########")
    sim_main.main(["--mode", "trajectories", "--n", "1", "--out", OUT,
                   "--top-type", "oval", "--seed", "5",
                   "--no-cooling-sleep", "--no-validate"])

    # 3) Validacion + resumen
    print("\n########## Validacion fisica ##########")
    video_states = os.path.join(OUT, "states", "ep_0000.json")
    traj_states = os.path.join(OUT, "trajectories", "ep_0000.json")
    v1, i1 = validate_physics(video_states)
    v2, i2 = validate_physics(traj_states)
    print(f"\nVIDEO  valid={v1}  issues={len(i1)}")
    print(f"TRAYEC valid={v2}  issues={len(i2)}")
    _summary(video_states, "VIDEO")
    _summary(traj_states, "TRAYECTORIAS")

    # Comprobar frames JPG + index.json del episodio de video.
    ep_dir = os.path.join(OUT, "frames", "ep_0000")
    jpgs = sorted(glob.glob(os.path.join(ep_dir, "frame_*.jpg")))
    with open(os.path.join(ep_dir, "index.json"), "r", encoding="utf-8") as f:
        idx = json.load(f)
    print(f"\n[VIDEO] {len(jpgs)} frames JPG en {ep_dir}")
    print(f"[VIDEO] index.json: fps={idx['fps']} n_frames={idx['n_frames']} "
          f"resolution={idx['resolution']}")
    print(f"[VIDEO] primera entrada index: {idx['frames'][0]}")
    assert len(jpgs) == idx["n_frames"], "mismatch frames JPG vs index.json"

    # 4) prepare_ude_dataset.py + carga de .npz
    print("\n########## prepare_ude_dataset (VIDEO states) ##########")
    prepare_ude_dataset.main(["--in", os.path.join(OUT, "states"),
                              "--out", os.path.join(OUT, "ude_video"),
                              "--target", "both", "--cut-on-fall"])
    _check_npz(os.path.join(OUT, "ude_video", "ep_0000.npz"), "VIDEO")

    print("\n########## prepare_ude_dataset (TRAYECTORIAS) ##########")
    prepare_ude_dataset.main(["--in", os.path.join(OUT, "trajectories"),
                              "--out", os.path.join(OUT, "ude_traj"),
                              "--target", "both", "--cut-on-fall"])
    _check_npz(os.path.join(OUT, "ude_traj", "ep_0000.npz"), "TRAYECTORIAS")

    # 5) export_to_parquet (opcional: requiere la libreria 'datasets').
    # IMPORTANTE: se ejecuta en un SUBPROCESO. Mezclar el contexto GL de MuJoCo
    # (usado al renderizar el video) con datasets/pyarrow/PIL en el MISMO
    # proceso provoca un segfault en libs nativas. El flujo real ya son dos
    # comandos separados (generar -> exportar), asi que esto refleja el uso.
    print("\n########## export_to_parquet (HuggingFace) ##########")
    import subprocess
    # NO importar 'datasets' en ESTE proceso: tras usar el contexto GL de MuJoCo
    # (render del video), importar datasets/pyarrow/PIL aqui segfaultea por
    # conflicto de libs nativas. Se comprueba e invoca todo en subprocesos.
    has_datasets = subprocess.run(
        [sys.executable, "-c", "import datasets, pyarrow"],
        capture_output=True).returncode == 0

    if has_datasets:
        hf_out = os.path.join(OUT, "hf_export")
        r = subprocess.run([sys.executable, "export_to_parquet.py", "--in", OUT,
                            "--out", hf_out, "--what", "all"])
        fp = os.path.join(hf_out, "frames.parquet")
        ok = (r.returncode == 0 and os.path.isfile(fp))
        if ok:
            mb = os.path.getsize(fp) / 1e6
            print(f"[PARQUET] OK -> {fp} ({mb:.2f} MB) "
                  f"+ states.parquet + trajectories.parquet")
        assert ok, "export_to_parquet fallo"
    else:
        print("[PARQUET] 'datasets' no instalado -> paso omitido. "
              "Instala con: pip install datasets pyarrow")

    print("\n[TEST OK] pipeline completo (video + trayectorias + UDE + parquet).")
    print(f"[TEST] salida en: {os.path.abspath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
