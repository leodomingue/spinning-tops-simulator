# Spinning Top Simulator (MuJoCo 3.x) — Sim-to-Real para VideoMamba + UDE

Simulador de **trompos con punta** (peonzas) en MuJoCo 3.x que genera dos tipos
de dataset para un pipeline Sim-to-Real:

- **Modo VIDEO** → frames **JPG** sintéticos post-procesados a **30 FPS** (para
  entrenar **VideoMamba**: regresión de cuaternión + velocidad angular) **+**
  JSON de estados físicos a **100 Hz** (para la **UDE**).
- **Modo TRAYECTORIAS** → solo JSON de estados a **100 Hz** (sin render), para
  entrenar la UDE de forma rápida y barata.

> La física se integra a **1 ms (1000 Hz)** internamente, pero los estados se
> guardan a **100 Hz** (cada 10 timesteps) en **ambos** modos. El vídeo se graba
> siempre a **30 FPS** como **frames JPG individuales** (el MP4 es solo debug).

---

## 1. Los 3 tipos de trompo (solo trompos CON PUNTA)

Todos tienen una **punta de contacto real** (esfera de radio 5 mm), CM
desplazado verticalmente sobre la punta, e inercia diagonal con el eje de spin
como momento mayor. Seleccionables con `--top-type` (o `random`):

| tipo    | forma                                                            | estabilidad |
|---------|------------------------------------------------------------------|-------------|
| `cone`  | peonza cónica clásica (punta abajo, ancho arriba)                | alta        |
| `acorn` | bellota: cono inferior + esfera/ovoide superior                  | media       |
| `oval`  | elipsoide alargado + punta (CM alto, se cae con facilidad)       | baja        |

Como MuJoCo 3.x no tiene primitivo "cono", los conos se aproximan apilando
cilindros de radio creciente. La inercia **no** se calcula de los geoms: se fija
explícitamente con `<inertial>`, así que la forma solo afecta a la **colisión**
y a la **apariencia**. Ver [physics/top_model.py](physics/top_model.py) para
cómo añadir **meshes STL reales** en el futuro.

### Nota sobre el eje de simetría (Y vs Z)

La especificación menciona en algunos sitios el "eje Y local" como eje de
spin, pero la geometría que describe (CM en `pos="0 0 com_height"`, punta abajo
a lo largo de Z, `position_z` como altura de la punta) está definida sobre el
**eje Z local**. Como ambas no pueden ser ciertas a la vez, este simulador usa
de forma **consistente el eje Z local** como eje de simetría/spin:

- el momento de inercia mayor (spin) es `Izz`;
- el spin inicial se aplica sobre Z local;
- `angle_from_vertical` = ángulo entre el eje Z local (rotado por `q`) y `+Z` mundo.

El vector `inertia_diag = [Ixx, Iyy, Izz]` del JSON y el contrato de la UDE
`[..., Ixx, Iyy, Izz, ...]` se respetan tal cual; simplemente el valor grande
vive en `Izz`. Además, MuJoCo exige que la inercia cumpla la desigualdad
triangular (`A+B>=C`), por lo que el spin se muestrea como `Izz = It·U(1.05,1.85)`
con `It` transversal (cuerpo oblato tipo peonza). Ver
[utils/domain_randomization.py](utils/domain_randomization.py).

---

## 2. Requisitos del sistema

- **Python 3.10+** (probado en 3.13).
- **MuJoCo 3.x** (`pip install mujoco`).
- **FFmpeg**: lo trae `imageio-ffmpeg` automáticamente. Solo es necesario para
  `--save-mp4` (debug). Alternativa de sistema: instalar `ffmpeg` y ponerlo en
  el `PATH`.
- **Render headless** (servidores / Vast.ai): `export MUJOCO_GL=egl`
  (fallback CPU: `export MUJOCO_GL=osmesa`).
- **GPU NVIDIA opcional** con drivers CUDA (acelera el render con backend EGL).
- Compatible **Linux y Windows**.

---

## 3. Instalación paso a paso

```bash
git clone <repo> && cd spinning_top_simulator
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`torch` es **opcional** (solo se usa, vía `try/except`, para
`torch.cuda.empty_cache()` durante el cooling). El simulador funciona sin él.

---

## 4. Cómo ejecutar cada modo

```bash
# Modo VIDEO: frames JPG a 30 FPS + estados 100 Hz (sin MP4)
python main.py --mode video --n 500 --out dataset --resolution 224 --subframes 2 \
               --top-type random --cooling-interval 20

# Igual pero guardando también MP4 de debug
python main.py --mode video --n 50 --out dataset --save-mp4

# Mezclar suelos procedurales (70%) con texturas reales si existen (30%)
python main.py --mode video --n 500 --out dataset --real-floor-prob 0.3

# 100% procedural (assets/floors vacía o prob 0)
python main.py --mode video --n 500 --out dataset --real-floor-prob 0.0

# Un solo tipo de trompo
python main.py --mode video --n 100 --top-type acorn --out dataset

# Trayectorias (solo estados 100 Hz, sin render)
python main.py --mode trajectories --n 2000 --out dataset --top-type random \
               --cooling-interval 50

# Preprocesar para la UDE
python prepare_ude_dataset.py --in dataset/states       --out dataset/ude --target both --cut-on-fall
python prepare_ude_dataset.py --in dataset/trajectories --out dataset/ude --target both --cut-on-fall

# Resumir un run interrumpido
python main.py --mode video --n 500 --out dataset --resume-from 150

# Vast.ai headless
export MUJOCO_GL=egl
export VAST_AI=true
python main.py --mode video --n 1000 --out dataset
```

Flags de `main.py`: `--mode {video,trajectories}`, `--n`, `--out`,
`--resume-from`, `--top-type {cone,acorn,oval,random}`,
`--resolution {224,480}`, `--subframes {2,4}`, `--cooling-interval`,
`--save-mp4`, `--real-floor-prob`, `--jpg-quality`, `--seed`, `--floors-dir`.
(`--seed` fija la semilla base reproducible; si se omite, se genera una y se
imprime para poder reproducir el run.)

---

## 5. Verificación rápida

```bash
python test_single_episode.py
```

Genera 1 secuencia de frames (30 FPS) + 1 trayectoria (100 Hz), valida la
física, corre `prepare_ude_dataset.py` sobre los episodios y carga los `.npz`
imprimiendo los shapes `X (T,14)`, `Y_next (T-1,7)`, `dY (T-1,7)`. Todo en
`test_output/`.

---

## 6. Salida para VideoMamba (frames JPG, NO MP4)

El dataset de vídeo son **frames JPG individuales** (`cv2.imwrite`, calidad 95),
no MP4. Razones:

1. **Acceso aleatorio** directo sin decodificar vídeo → no satura CPU ni deja la
   GPU esperando.
2. **Sin compresión temporal** inter-frame que emborrona la rotación rápida y
   daña la regresión de cuaternión/ω.
3. Compatible con dataloaders estándar y **paralelizable**.

`--save-mp4` es **solo para inspección visual**, no para entrenar.

Cada episodio trae un `index.json` que alinea cada `frame_XXXXX.jpg` con su
timestamp y con el índice del estado físico (100 Hz):

```json
{
  "fps": 30, "n_frames": 375, "resolution": [224, 224],
  "frames": [
    {"file": "frame_00000.jpg", "t": 0.0,    "state_idx": 0},
    {"file": "frame_00001.jpg", "t": 0.0333, "state_idx": 3}
  ]
}
```

`state_idx` apunta al índice del array `frames` del JSON de estados (100 Hz) más
cercano en el tiempo (`round(t·100)`).

### Dataset de PyTorch para VideoMamba

```python
import os, json, glob, cv2, numpy as np, torch
from torch.utils.data import Dataset

class TopVideoClips(Dataset):
    """Devuelve (clip[T,3,H,W], label[7]) = cuaternión(4)+omega(3) del último frame."""
    def __init__(self, dataset_root, clip_len=16):
        self.clip_len = clip_len
        self.eps = []
        for ep_dir in sorted(glob.glob(os.path.join(dataset_root, "frames", "ep_*"))):
            ep_id = os.path.basename(ep_dir).split("_")[1]
            idx = json.load(open(os.path.join(ep_dir, "index.json")))
            states = json.load(open(os.path.join(
                dataset_root, "states", f"ep_{ep_id}.json")))["frames"]
            self.eps.append((ep_dir, idx["frames"], states))

    def __len__(self):
        return sum(max(0, len(f) - self.clip_len + 1) for _, f, _ in self.eps)

    def _locate(self, i):
        for ep_dir, frames, states in self.eps:
            n = max(0, len(frames) - self.clip_len + 1)
            if i < n:
                return ep_dir, frames, states, i
            i -= n
        raise IndexError

    def __getitem__(self, i):
        ep_dir, frames, states, start = self._locate(i)
        clip = []
        for f in frames[start:start + self.clip_len]:
            img = cv2.cvtColor(cv2.imread(os.path.join(ep_dir, f["file"])),
                               cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            clip.append(torch.from_numpy(img).permute(2, 0, 1))
        last = frames[start + self.clip_len - 1]
        s = states[last["state_idx"]]
        label = torch.tensor(s["q"] + s["omega"], dtype=torch.float32)  # (7,)
        return torch.stack(clip), label
```

---

## 7. Tensor de entrenamiento de la UDE (contrato 14D)

La UDE consume por timestep un vector **14D** (Sección 5.1):

```
input(14) = [ q(4), omega(3), mass, Ixx, Iyy, Izz, fc, fv, ell ]
```

Los **priors** (`mass, Ixx, Iyy, Izz, fc=coulomb_torque, fv=viscous_friction,
ell`) viven en `metadata.physics` (constantes por episodio) y se
**broadcastean** a todos los frames.

### `ell` (ℓ): distancia punta → CM — el brazo del torque gravitatorio

`ell = pivot_to_com = com_height + tip_radius` es la distancia del **punto de
contacto (punta) al centro de masa**. Es imprescindible para la UDE porque el
término físico conocido incluye el torque gravitatorio:

```
τ_gravedad = m · g · ℓ · sin(θ)      →      Ω_precesión ≈ (m·g·ℓ) / (I_spin · ω_spin)
```

Sin `ell`, dos trompos con la **misma** masa/inercia/fricción pero distinta
altura de CM tendrían dinámicas distintas con input **idéntico** → no
identificable. (Analogía: a una persona no le das un tensor de inercia; le das
la masa, la forma y la **altura medida con una regla**, de donde sale ℓ.)

> Nota física: el brazo correcto es la distancia **punta→CM**, no origen→CM, por
> eso `ell = com_height + tip_radius` (la punta es una esfera de 5 mm; el
> contacto está en su parte inferior). `g = 9.81` es constante conocida y no se
> incluye en el vector. En el JSON se guardan `com_height`, `tip_radius` y
> `pivot_to_com` para trazabilidad; la UDE usa `pivot_to_com` (= ℓ).

Flujo de preprocesado (offline, sin MuJoCo, mucho más rápido que parsear JSON en
cada batch):

```bash
python prepare_ude_dataset.py --in dataset/states --out dataset/ude --target both
```

Genera por episodio `dataset/ude/ep_XXXX.npz` con keys `X (T,14)`,
`Y_next (T-1,7)`, `dY (T-1,7)`, `t (T,)`, `fall_mask (T,)`, `dt`, `top_type`;
más `norm_stats.json` (mean/std por dimensión, **solo** frames con
`fall_detected==False`; el **cuaternión no se normaliza**) y `manifest.json`
(episodios, frames válidos y rangos físicos vistos).

`--cut-on-fall` (default `True`) recorta los frames a partir de la caída para no
entrenar con dinámica degenerada. `--target {next,deriv,both}` elige el objetivo
(autoregresivo / derivada tipo Neural ODE / ambos).

### Dataset de PyTorch para la UDE

```python
import os, json, glob, numpy as np, torch
from torch.utils.data import Dataset

class UDEDataset(Dataset):
    def __init__(self, ude_dir, target="dY"):
        self.files = sorted(glob.glob(os.path.join(ude_dir, "ep_*.npz")))
        ns = json.load(open(os.path.join(ude_dir, "norm_stats.json")))
        self.x_mean = np.array(ns["X"]["mean"], np.float32)
        self.x_std  = np.array(ns["X"]["std"],  np.float32)
        self.target = target

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=True)
        X = (z["X"] - self.x_mean) / self.x_std          # cuaternión: mean0/std1
        Y = z[self.target]                                # (T-1,7)
        fall = z["fall_mask"]                             # (T,)
        return (torch.from_numpy(X.astype(np.float32)),
                torch.from_numpy(Y.astype(np.float32)),
                torch.from_numpy(fall))
```

### `fall_detected` / `fall_time` como horizonte de integración

- `angle_from_vertical` (grados) = ángulo entre el eje de simetría y la vertical.
- `fall_detected = (angle_from_vertical > 70°) OR (‖omega‖ < 0.1)`, **latch
  monótono** (una vez `true`, sigue `true`).
- `fall_time` = `t` del primer frame con `fall_detected==true` (`null` si nunca
  cae).

Úsalo como **horizonte máximo de integración** del filtro de Kalman / UDE: no
integres más allá de `fall_time` (la dinámica post-caída es degenerada).

---

## 8. Suelos procedurales automáticos

**No hace falta descargar ni colocar imágenes**: cada episodio genera un suelo
aleatorio (checker, gradiente, madera, mármol, fractal/cemento, liso con
micro-ruido). Las texturas numpy se pasan a MuJoCo como **PNG en memoria** (VFS
de `from_xml_string`), sin ficheros temporales (headless-safe).

- `assets/floors/` es **opcional**: si pones JPG/PNG reales ahí, se mezclan con
  probabilidad `--real-floor-prob` (default 0.3). Si está vacía → 100% procedural
  sin error.
- El tipo de suelo usado se guarda en `metadata.floor_type` (trazabilidad).

---

## 9. Notas para Vast.ai

- Variables de entorno: `export MUJOCO_GL=egl` y `export VAST_AI=true`.
- Con `VAST_AI=true`: `--resolution 224`, `--subframes 2` y cooling cada 50 por
  defecto.
- **Cooling** (`--cooling-interval N`): cada N episodios hace `gc.collect()`,
  `torch.cuda.empty_cache()` (si hay torch+CUDA), duerme 30 s y loggea ETA. En
  local usa `--no-cooling-sleep` para no dormir.
- **Resume**: el run guarda `dataset/progress.json` tras cada episodio. Reanuda
  con `--resume-from N` (¡usa la **misma `--seed`** para reproducir las semillas
  por episodio!).
- Usa un **volumen persistente** para `--out` para no perder el dataset si el
  contenedor se reinicia.

---

## 10. Estimación de almacenamiento

A 224² con post-processing y suelos texturizados, **~5–15 MB por episodio** de
vídeo (≈360 frames JPG a calidad 95; el tamaño sube con texturas/ruido de alta
frecuencia). Ej.: 500 episodios ≈ **2.5–7.5 GB**. Baja la calidad con
`--jpg-quality` (p.ej. 85) para reducir el tamaño. Las trayectorias son
~0.2–0.5 MB por episodio (solo JSON).

---

## 11. Estructura de la salida

```
dataset/
├── frames/ep_0000/frame_00000.jpg ...   # 30 FPS, calidad 95  -> VideoMamba
│                  index.json             # frame <-> t <-> state_idx
├── states/ep_0000.json                   # estados 100 Hz (modo video)
├── trajectories/ep_0000.json             # estados 100 Hz (modo trajectories)
├── video_debug/ep_0000.mp4               # SOLO si --save-mp4 (no entrenar)
├── ude/ep_0000.npz, norm_stats.json, manifest.json   # tras prepare_ude_dataset
└── progress.json                         # para --resume-from
```

- **Alimenta a VideoMamba**: `frames/ep_XXXX/*.jpg` + `index.json` (label =
  cuaternión + ω del estado alineado vía `state_idx`).
- **Alimenta a la UDE**: `ude/*.npz` (+ `norm_stats.json`), generados desde
  `states/` o `trajectories/`.

> El paquete de **código** se llama `dataset/` y la carpeta de **salida** por
> defecto también (`--out dataset`). No colisionan (las subcarpetas de salida no
> chocan con los `.py`), pero puedes usar `--out output` si prefieres separarlos.

---

## 12. Formato del JSON de estados

El bloque `metadata.physics` incluye `mass`, `inertia_diag=[Ixx,Iyy,Izz]`,
`com_height`, `tip_radius`, `pivot_to_com` (= ℓ), las fricciones,
`coulomb_torque` (fc), `viscous_friction` (fv), `initial_spin`,
`initial_tilt_rad` e `initial_position`.

Ver `metadata` (episodio, seed, modo, tipo de trompo, `floor_type`, `physics`,
`simulation`, `fall_time`, `valid`) y `frames` (`t`, `q=[w,x,y,z]`,
`omega` (mundo), `x`/`v` (CM mundo), `has_contact`, `contact_force`,
`angle_from_vertical`, `fall_detected`). Los estados están **siempre a 100 Hz**
en ambos modos; en `trajectories`, `video_fps` es `null`.

`omega` y `v` se toman de `data.cvel` (velocidad espacial com-based con ejes
alineados al mundo), evitando la ambigüedad local/mundo de `qvel` del freejoint.

---

## 13. Validación física

`utils/validation.py::validate_physics(json_path)` comprueba: energía
≈monótona, sin penetración, cuaternión unitario, no nace cayendo (punta ≤5 mm en
t=0), duración razonable, `fall_detected` monótono y consistencia de
`fall_time`. Los fallos "duros" ponen `metadata.valid=false` (pero **no**
descartan el episodio); los "suaves" (energía/duración) solo avisan.
```
