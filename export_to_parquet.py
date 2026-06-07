#!/usr/bin/env python3
"""Exporta el dataset generado (frames JPG + JSON de estados/trayectorias) a
ficheros .parquet listos para subir a HuggingFace.

Usa la libreria ``datasets`` (idiomatica de HF): los frames se guardan con la
feature ``Image`` (bytes JPG embebidos en el parquet), de modo que el dataset
se renderiza solo en el Hub y se carga con ``load_dataset``.

Splits que produce (segun lo que exista en la carpeta de entrada):
  * frames        -> 1 fila por frame de video: imagen + label (q, omega del
                     estado alineado) + motion_state, fall_detected, priors
                     fisicos (incl. ell) y markers_json.   [VideoMamba]
  * states        -> 1 fila por estado (100 Hz) de dataset/states.   [UDE]
  * trajectories  -> 1 fila por estado de dataset/trajectories.      [UDE]

Uso:
  pip install datasets pyarrow

  # A parquet local
  python export_to_parquet.py --in dataset_test --out hf_export --what all

  # Directo al Hub (requiere `huggingface-cli login`)
  python export_to_parquet.py --in dataset --out hf_export \
         --push-to-hub usuario/trompos-sim --private

Cada split se puede cargar luego con:
  from datasets import load_dataset
  ds = load_dataset("parquet", data_files="hf_export/frames.parquet")
  # o desde el Hub:
  ds = load_dataset("usuario/trompos-sim", "frames")
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
from typing import Iterator


def _require_datasets():
    try:
        import datasets  # noqa
        import pyarrow  # noqa
        return __import__("datasets")
    except Exception:
        sys.exit("[ERROR] Falta 'datasets'/'pyarrow'. Instala: "
                 "pip install datasets pyarrow")


# --------------------------------------------------------------------------- #
# Generadores de filas                                                         #
# --------------------------------------------------------------------------- #
def _iter_frames(dataset_dir: str) -> Iterator[dict]:
    """1 fila por frame JPG, con la imagen y el estado fisico alineado."""
    frames_root = os.path.join(dataset_dir, "frames")
    states_root = os.path.join(dataset_dir, "states")
    for ep_dir in sorted(glob.glob(os.path.join(frames_root, "ep_*"))):
        ep_id = int(os.path.basename(ep_dir).split("_")[1])
        idx_path = os.path.join(ep_dir, "index.json")
        st_path = os.path.join(states_root, f"ep_{ep_id:04d}.json")
        if not (os.path.isfile(idx_path) and os.path.isfile(st_path)):
            print(f"[WARN] salto ep {ep_id}: falta index.json o states json")
            continue
        with open(idx_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        with open(st_path, "r", encoding="utf-8") as f:
            sdoc = json.load(f)
        meta, phys, states = sdoc["metadata"], sdoc["metadata"]["physics"], sdoc["frames"]
        markers_json = json.dumps(meta.get("markers", []))
        ell = phys.get("pivot_to_com")
        ell = float(ell) if ell is not None else float("nan")
        for fr in index["frames"]:
            si = min(int(fr["state_idx"]), len(states) - 1)
            s = states[si]
            with open(os.path.join(ep_dir, fr["file"]), "rb") as imf:
                img_bytes = imf.read()
            yield {
                "image": {"bytes": img_bytes,
                          "path": f"ep_{ep_id:04d}/{fr['file']}"},
                "episode_id": ep_id,
                "frame_file": fr["file"],
                "t": float(fr["t"]),
                "state_idx": si,
                "q": [float(x) for x in s["q"]],
                "omega": [float(x) for x in s["omega"]],
                "angle_from_vertical": float(s["angle_from_vertical"]),
                "fall_detected": bool(s["fall_detected"]),
                "motion_state": s.get("motion_state", "unknown"),
                "top_type": meta["top_type"],
                "floor_type": meta["floor_type"],
                "mass": float(phys["mass"]),
                "inertia_diag": [float(x) for x in phys["inertia_diag"]],
                "fc": float(phys["coulomb_torque"]),
                "fv": float(phys["viscous_friction"]),
                "ell": ell,
                "seed": int(meta["seed"]),
                "markers_json": markers_json,
            }


def _iter_states(dataset_dir: str, subdir: str) -> Iterator[dict]:
    """1 fila por estado (100 Hz) de dataset/<subdir>/ep_*.json."""
    root = os.path.join(dataset_dir, subdir)
    for jp in sorted(glob.glob(os.path.join(root, "ep_*.json"))):
        with open(jp, "r", encoding="utf-8") as f:
            sdoc = json.load(f)
        meta, phys = sdoc["metadata"], sdoc["metadata"]["physics"]
        ell = phys.get("pivot_to_com")
        ell = float(ell) if ell is not None else float("nan")
        markers_json = json.dumps(meta.get("markers", []))
        for s in sdoc["frames"]:
            yield {
                "episode_id": int(meta["episode_id"]),
                "t": float(s["t"]),
                "q": [float(x) for x in s["q"]],
                "omega": [float(x) for x in s["omega"]],
                "x": [float(x) for x in s["x"]],
                "v": [float(x) for x in s["v"]],
                "has_contact": bool(s["has_contact"]),
                "contact_force": float(s["contact_force"]),
                "angle_from_vertical": float(s["angle_from_vertical"]),
                "fall_detected": bool(s["fall_detected"]),
                "motion_state": s.get("motion_state", "unknown"),
                "top_type": meta["top_type"],
                "floor_type": meta["floor_type"],
                "mass": float(phys["mass"]),
                "inertia_diag": [float(x) for x in phys["inertia_diag"]],
                "fc": float(phys["coulomb_torque"]),
                "fv": float(phys["viscous_friction"]),
                "ell": ell,
                "seed": int(meta["seed"]),
                "markers_json": markers_json,
            }


# --------------------------------------------------------------------------- #
# Features                                                                     #
# --------------------------------------------------------------------------- #
def _frames_features(ds):
    F, V, S, Img = ds.Features, ds.Value, ds.Sequence, ds.Image
    return F({
        "image": Img(),
        "episode_id": V("int32"),
        "frame_file": V("string"),
        "t": V("float32"),
        "state_idx": V("int32"),
        "q": S(V("float32"), length=4),
        "omega": S(V("float32"), length=3),
        "angle_from_vertical": V("float32"),
        "fall_detected": V("bool"),
        "motion_state": V("string"),
        "top_type": V("string"),
        "floor_type": V("string"),
        "mass": V("float32"),
        "inertia_diag": S(V("float32"), length=3),
        "fc": V("float32"), "fv": V("float32"), "ell": V("float32"),
        "seed": V("int64"),
        "markers_json": V("string"),
    })


def _states_features(ds):
    F, V, S = ds.Features, ds.Value, ds.Sequence
    return F({
        "episode_id": V("int32"),
        "t": V("float32"),
        "q": S(V("float32"), length=4),
        "omega": S(V("float32"), length=3),
        "x": S(V("float32"), length=3),
        "v": S(V("float32"), length=3),
        "has_contact": V("bool"),
        "contact_force": V("float32"),
        "angle_from_vertical": V("float32"),
        "fall_detected": V("bool"),
        "motion_state": V("string"),
        "top_type": V("string"),
        "floor_type": V("string"),
        "mass": V("float32"),
        "inertia_diag": S(V("float32"), length=3),
        "fc": V("float32"), "fv": V("float32"), "ell": V("float32"),
        "seed": V("int64"),
        "markers_json": V("string"),
    })


# --------------------------------------------------------------------------- #
def _build_and_save(ds_lib, gen, gen_kwargs, features, split_name, args):
    dataset = ds_lib.Dataset.from_generator(
        gen, gen_kwargs=gen_kwargs, features=features)
    n = len(dataset)
    if n == 0:
        print(f"[SKIP] '{split_name}': 0 filas (no hay datos en la entrada).")
        return 0

    if args.push_to_hub:
        print(f"[HUB] subiendo '{split_name}' ({n} filas) a "
              f"{args.push_to_hub} ...")
        dataset.push_to_hub(args.push_to_hub, config_name=split_name,
                            private=args.private)
    else:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, f"{split_name}.parquet")
        dataset.to_parquet(out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[OK] '{split_name}': {n} filas -> {out_path} ({size_mb:.2f} MB)")
    return n


def _write_dataset_card(out_dir: str, splits) -> None:
    card = f"""# Spinning Top Simulator dataset

Dataset sintetico de trompos (MuJoCo) para Sim-to-Real.

Splits incluidos: {', '.join(splits)}

- **frames**: 1 fila por frame de video (224x224 JPG) con la imagen y, como
  label, el cuaternion `q=[w,x,y,z]` y la velocidad angular `omega` (mundo) del
  estado fisico alineado. Incluye `motion_state` (spinning/fallen/stopped),
  `fall_detected`, priors fisicos (`mass`, `inertia_diag`, `fc`, `fv`, `ell`) y
  `markers_json` (>=3 marcadores de color no colineales pegados al cuerpo).
  Pensado para VideoMamba.
- **states / trajectories**: 1 fila por estado a 100 Hz para entrenar la UDE.

```python
from datasets import load_dataset
frames = load_dataset("parquet", data_files="frames.parquet", split="train")
print(frames[0]["image"], frames[0]["q"], frames[0]["omega"])
```
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Exporta el dataset a .parquet para HuggingFace.")
    p.add_argument("--in", dest="in_dir", required=True,
                   help="carpeta del dataset (p.ej. dataset_test, dataset)")
    p.add_argument("--out", dest="out", default="hf_export",
                   help="carpeta de salida de los .parquet")
    p.add_argument("--what", choices=["frames", "states", "trajectories", "all"],
                   default="all")
    p.add_argument("--push-to-hub", default=None,
                   help="repo_id del Hub (p.ej. usuario/dataset). Requiere login")
    p.add_argument("--private", action="store_true", default=False)
    args = p.parse_args(argv)

    ds_lib = _require_datasets()

    targets = (["frames", "states", "trajectories"]
               if args.what == "all" else [args.what])
    written = []
    total = 0

    for split in targets:
        if split == "frames":
            if not os.path.isdir(os.path.join(args.in_dir, "frames")):
                continue
            n = _build_and_save(ds_lib, _iter_frames,
                                {"dataset_dir": args.in_dir},
                                _frames_features(ds_lib), "frames", args)
        else:  # states | trajectories
            if not os.path.isdir(os.path.join(args.in_dir, split)):
                continue
            n = _build_and_save(ds_lib, _iter_states,
                                {"dataset_dir": args.in_dir, "subdir": split},
                                _states_features(ds_lib), split, args)
        if n:
            written.append(split)
            total += n

    if not written:
        print(f"[ERROR] no se encontro nada que exportar en '{args.in_dir}'. "
              f"Esperaba subcarpetas frames/ , states/ o trajectories/.")
        return 1

    if not args.push_to_hub:
        _write_dataset_card(args.out, written)
    print(f"[DONE] splits={written} | filas totales={total} | "
          f"{'Hub: ' + args.push_to_hub if args.push_to_hub else 'out=' + args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
