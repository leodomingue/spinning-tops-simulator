#!/usr/bin/env python3
"""Simulador de trompos (MuJoCo 3.x) -> datasets Sim-to-Real.

Dos modos:
  * video        : frames JPG a 30 FPS (VideoMamba) + estados 100 Hz (UDE)
  * trajectories : solo estados 100 Hz (UDE), sin renderizado (rapido/barato)

PUNTOS CLAVE DE TEMPORIZACION (controlados aqui en main.py):
  * La FISICA se integra a 1 ms (1000 Hz) -> ``model.opt.timestep = 0.001``.
  * Los ESTADOS se guardan a 100 Hz = cada 10 timesteps (NO a 1000 Hz).
        -> ver ``STATE_EVERY = 10`` y ``if step_count % STATE_EVERY == 0``.
  * El VIDEO se graba a 30 FPS -> ver el bucle por frame con ``target_step``.
  * ``fall_detected`` / ``angle_from_vertical`` se calculan por frame de estado
        -> ver ``_compute_state`` y el latch monotono ``self.fall_latched``.

Ejecutar ``python main.py --help`` para ver todos los flags.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import platform

import numpy as np

# Constantes de temporizacion
PHYS_DT = 0.001          # 1 ms (1000 Hz integracion interna)
STATE_EVERY = 10         # guardar estado cada 10 pasos => 100 Hz
VIDEO_FPS = 30
MAX_T = 20.0             # duracion maxima (s)
FALL_ANGLE_DEG = 70.0    # umbral de caida por angulo
OMEGA_REST = 0.1         # rad/s, umbral de "spin ~ cero"
VEL_REST = 0.01          # m/s, umbral de reposo lineal (modo video)
POST_FALL_SECONDS = 1.0  # tras la caida se graba solo este extra (s)


# --------------------------------------------------------------------------- #
# Deteccion de entorno cloud (Vast.ai) y backend GL headless                   #
# --------------------------------------------------------------------------- #
def detect_vast() -> bool:
    return os.environ.get("VAST_AI", "").lower() in ("1", "true", "yes", "on")


def setup_gl_backend(mode: str, is_vast: bool) -> None:
    """Configura MUJOCO_GL ANTES de importar mujoco (solo relevante en video)."""
    if mode != "video":
        return
    if "MUJOCO_GL" in os.environ:
        return
    # En cloud/Linux headless usamos EGL (GPU). En Windows/Mac dejamos el
    # backend nativo. Fallback documentado: osmesa (CPU).
    if is_vast or (platform.system() == "Linux" and not os.environ.get("DISPLAY")):
        os.environ["MUJOCO_GL"] = "egl"


# --------------------------------------------------------------------------- #
# Helpers de fisica (requieren mujoco; se importa dentro de run())             #
# --------------------------------------------------------------------------- #
def _read_contact(mujoco, model, data, floor_gid, top_bid):
    """Devuelve (has_contact, fuerza_normal_total) entre el trompo y el suelo."""
    has = False
    fmag = 0.0
    f6 = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        involves_top = (b1 == top_bid) or (b2 == top_bid)
        involves_floor = (c.geom1 == floor_gid) or (c.geom2 == floor_gid)
        if involves_top and involves_floor:
            has = True
            mujoco.mj_contactForce(model, data, i, f6)
            fmag += abs(float(f6[0]))  # componente normal
    return has, fmag


def _apply_pivot_friction(data, vadr, fc, fv, has_contact):
    """Aplica friccion de pivote (Coulomb + viscosa) sobre el eje de spin.

    Modelo: tau = -(fc * sign(w_spin) + fv * w_spin), aplicado alrededor del
    eje de simetria (Z local). qfrc_applied en los DOF rotacionales de un
    freejoint es torque en el frame LOCAL, por lo que el indice +5 actua sobre
    el eje Z local = eje de spin. Solo cuando hay contacto con el suelo.
    """
    data.qfrc_applied[vadr + 3: vadr + 6] = 0.0
    if not has_contact:
        return
    w_spin = float(data.qvel[vadr + 5])  # spin sobre Z local
    tau = -(fc * np.sign(w_spin) + fv * w_spin)
    data.qfrc_applied[vadr + 5] = tau


def _compute_state(mujoco, model, data, qadr, top_bid, floor_gid, angle_fn):
    """Extrae el estado fisico para el JSON (todo en frame MUNDO).

    omega y v se toman de ``data.cvel`` (velocidad espacial com-based, ejes
    alineados con el mundo): cvel[bid][0:3] = velocidad angular en MUNDO,
    cvel[bid][3:6] = velocidad lineal del CM en MUNDO. Esto evita la ambiguedad
    local/mundo de qvel para el freejoint.
    """
    q = np.array(data.qpos[qadr + 3: qadr + 7], dtype=np.float64)
    omega = np.array(data.cvel[top_bid][0:3], dtype=np.float64)  # mundo
    v = np.array(data.cvel[top_bid][3:6], dtype=np.float64)      # mundo (CM)
    x = np.array(data.xipos[top_bid], dtype=np.float64)          # CM mundo
    has_contact, fmag = _read_contact(mujoco, model, data, floor_gid, top_bid)
    angle = angle_fn(q)
    return q, omega, v, x, has_contact, fmag, angle


# --------------------------------------------------------------------------- #
# Captura de subframes para motion blur                                        #
# --------------------------------------------------------------------------- #
def _capture_steps(prev_step: int, target_step: int, subframes: int):
    """Pasos en los que renderizar subframes (obturador ~60% del intervalo)."""
    if subframes <= 1:
        return {target_step}
    win = max(1, int((target_step - prev_step) * 0.6))
    start = target_step - win
    steps = set()
    for i in range(subframes):
        s = start + round(i * win / (subframes - 1))
        steps.add(min(s, target_step))
    return steps


# --------------------------------------------------------------------------- #
# Simulacion de un episodio                                                     #
# --------------------------------------------------------------------------- #
def simulate_episode(mods, params, args):
    """Simula un episodio completo y escribe sus salidas. Devuelve metadata."""
    mujoco = mods["mujoco"]
    build_top_model_xml = mods["build_top_model_xml"]
    generate_floor = mods["generate_floor"]
    set_initial_conditions = mods["set_initial_conditions"]
    angle_from_vertical_deg = mods["angle_from_vertical_deg"]
    StateLogger = mods["StateLogger"]
    FrameWriter = mods["FrameWriter"]
    VideoWriter = mods["VideoWriter"]

    mode = params.mode

    # --- Suelo procedural (texturas via VFS de from_xml_string) ---
    rng_floor = np.random.default_rng(params.seed + 101)
    floor_xml, floor_type, assets = generate_floor(
        rng_floor, real_floor_prob=args.real_floor_prob,
        assets_floors_dir=args.floors_dir,
    )
    params.floor_type = floor_type

    # --- Modelo ---
    xml = build_top_model_xml(params, floor_xml)
    model = mujoco.MjModel.from_xml_string(xml, assets)
    # Reafirmar opciones de integracion (Seccion 3.4).
    model.opt.timestep = PHYS_DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.gravity[:] = [0.0, 0.0, -9.81]

    data = mujoco.MjData(model)
    rng_init = np.random.default_rng(params.seed + 777)
    set_initial_conditions(model, data, params, rng_init)

    top_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "top")
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    qadr = int(model.jnt_qposadr[0])
    vadr = int(model.jnt_dofadr[0])
    fc = params.coulomb_torque
    fv = params.viscous_friction

    logger = StateLogger(params, mode)
    fall_latched = False
    fall_time = [None]                  # holder mutable: t de la primera caida
    post_fall = float(args.post_fall_seconds)

    def log_current(t):
        nonlocal fall_latched
        q, omega, v, x, has_c, fmag, angle = _compute_state(
            mujoco, model, data, qadr, top_bid, floor_gid, angle_from_vertical_deg)
        wn = float(np.linalg.norm(omega))
        # Estado INSTANTANEO de ese frame (flag pedida en el JSON).
        if wn < OMEGA_REST:
            motion_state = "stopped"     # detenido (dejo de girar)
        elif angle > FALL_ANGLE_DEG:
            motion_state = "fallen"      # se cayo (sigue moviendose)
        else:
            motion_state = "spinning"    # de pie y girando
        fall_now = (angle > FALL_ANGLE_DEG) or (wn < OMEGA_REST)
        if fall_now and not fall_latched:
            fall_time[0] = t             # primer instante de caida (latch)
        fall_latched = fall_latched or fall_now
        logger.add_state(t, q, omega, x, v, has_c, fmag, angle,
                         fall_latched, motion_state)
        return wn, float(np.linalg.norm(v)), angle

    def reached_end(t):
        """Fin de la simulacion: 20 s como tope, o (caida + post_fall)."""
        if t >= MAX_T:
            return True
        if fall_time[0] is not None and t >= fall_time[0] + post_fall:
            return True
        return False

    # Estado inicial (t=0, indice de estado 0)
    log_current(0.0)
    step_count = 0

    # ===================== MODO TRAYECTORIAS ===================== #
    if mode == "trajectories":
        while True:
            has_c, _ = _read_contact(mujoco, model, data, floor_gid, top_bid)
            _apply_pivot_friction(data, vadr, fc, fv, has_c)
            mujoco.mj_step(model, data)
            step_count += 1
            if step_count % STATE_EVERY == 0:
                log_current(data.time)
                # Se graba el movimiento inicial + post_fall s tras la caida
                # (o reposo), con tope de 20 s. La caida queda registrada.
                if reached_end(data.time):
                    break
            elif data.time >= MAX_T:
                break

        states_path = os.path.join(args.out, "trajectories",
                                   f"ep_{params.episode_id:04d}.json")
        meta = logger.write(states_path, video_fps=None,
                            post_fall_seconds=post_fall)
        meta["_states_path"] = states_path
        meta["_fall_time"] = fall_time[0]
        return meta

    # ========================= MODO VIDEO ========================= #
    TopRenderer = mods["TopRenderer"]
    PostProcessor = mods["PostProcessor"]

    rng_post = np.random.default_rng(params.seed + 202)
    renderer = TopRenderer(model, params, args.resolution,
                           rng=np.random.default_rng(params.seed + 13), body_id=top_bid)
    post = PostProcessor(rng_post)
    fw = FrameWriter(os.path.join(args.out, "frames"), params.episode_id,
                     jpg_quality=args.jpg_quality)
    vw = None
    if args.save_mp4:
        vw = VideoWriter(os.path.join(args.out, "video_debug"),
                         params.episode_id, fps=VIDEO_FPS)

    out_size = renderer.out_size

    def render_blur(cap_renders):
        stack = np.stack(cap_renders).astype(np.float32)
        mean = np.mean(stack, axis=0).astype(np.uint8)
        return post.process(mean, out_size)

    # --- Frame 0 (t=0) ---
    subs0 = [renderer.render(data, jitter=(i > 0)) for i in range(args.subframes)]
    frame0 = render_blur(subs0)
    fname0 = fw.write_frame(0, frame0)
    fw.add_index(fname0, 0.0, 0)
    if vw:
        vw.append(frame0)

    prev_step = 0
    k = 0
    done = False
    while not done:
        k += 1
        target_step = round(k * (1.0 / VIDEO_FPS) / PHYS_DT)
        if target_step * PHYS_DT > MAX_T + 1e-9:
            break
        cap_steps = _capture_steps(prev_step, target_step, args.subframes)
        subs = []
        while step_count < target_step:
            has_c, _ = _read_contact(mujoco, model, data, floor_gid, top_bid)
            _apply_pivot_friction(data, vadr, fc, fv, has_c)
            mujoco.mj_step(model, data)
            step_count += 1
            if step_count % STATE_EVERY == 0:
                wn, vn, angle = log_current(data.time)
            if step_count in cap_steps:
                subs.append(renderer.render(data, jitter=True))
        while len(subs) < args.subframes:
            subs.append(renderer.render(data, jitter=True))

        frame = render_blur(subs)
        fname = fw.write_frame(k, frame)
        t_frame = target_step * PHYS_DT
        state_idx = int(round(t_frame * 100.0))
        fw.add_index(fname, t_frame, state_idx)
        if vw:
            vw.append(frame)
        prev_step = target_step

        # En video se sigue grabando AUNQUE caiga, pero solo post_fall s mas
        # tras la caida (todo el movimiento inicial + ~1 s). Tope de 20 s.
        if reached_end(data.time):
            done = True

    # Escribir estados (100 Hz) + index de frames (clamp de state_idx).
    n_states = logger.n_frames
    for e in fw.index_entries:
        e["state_idx"] = min(e["state_idx"], n_states - 1)

    states_path = os.path.join(args.out, "states",
                               f"ep_{params.episode_id:04d}.json")
    meta = logger.write(states_path, video_fps=VIDEO_FPS,
                        post_fall_seconds=post_fall)
    fw.write_index(VIDEO_FPS, (renderer.height, renderer.width))
    if vw:
        vw.close()
    renderer.close()

    meta["_states_path"] = states_path
    meta["_n_frames"] = len(fw.index_entries)
    meta["_fall_time"] = fall_time[0]
    return meta


# --------------------------------------------------------------------------- #
# Resume / progreso                                                            #
# --------------------------------------------------------------------------- #
def write_progress(out: str, info: dict) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "progress.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def read_progress(out: str):
    p = os.path.join(out, "progress.json")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def episode_seed(base_seed: int, episode_id: int) -> int:
    """Semilla reproducible y distinta por episodio."""
    rng = np.random.default_rng([int(base_seed), int(episode_id)])
    return int(rng.integers(0, 2**31 - 1))


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulador de trompos (MuJoCo) -> datasets Sim-to-Real.")
    p.add_argument("--mode", choices=["video", "trajectories"], required=True)
    p.add_argument("--n", type=int, required=True, help="numero de episodios")
    p.add_argument("--out", type=str, default="dataset", help="carpeta de salida")
    p.add_argument("--resume-from", type=int, default=0,
                   help="continuar desde el episodio N")
    p.add_argument("--top-type", choices=["cone", "acorn", "oval", "random"],
                   default="random")
    p.add_argument("--resolution", type=int, choices=[224, 480], default=None)
    p.add_argument("--subframes", type=int, choices=[2, 4], default=None)
    p.add_argument("--cooling-interval", type=int, default=None)
    p.add_argument("--save-mp4", action="store_true", default=False)
    p.add_argument("--real-floor-prob", type=float, default=0.3)
    p.add_argument("--post-fall-seconds", type=float, default=POST_FALL_SECONDS,
                   help="segundos a grabar despues de detectar la caida "
                        "(p.ej. 1.0 o 0.5); el resto se descarta")
    p.add_argument("--jpg-quality", type=int, default=95)
    p.add_argument("--seed", type=int, default=None,
                   help="semilla base reproducible (si se omite, aleatoria)")
    p.add_argument("--floors-dir", type=str, default="assets/floors",
                   help="carpeta opcional de texturas reales de suelo")
    p.add_argument("--no-cooling-sleep", action="store_true", default=False,
                   help="no dormir en el cooling (util en local)")
    p.add_argument("--no-validate", action="store_true", default=False,
                   help="no ejecutar validate_physics tras cada episodio")
    return p


def apply_env_defaults(args, is_vast: bool) -> None:
    """Defaults distintos en cloud (Vast.ai) vs local (Seccion 8.3)."""
    if args.resolution is None:
        args.resolution = 224  # 224 por defecto en ambos
    if args.subframes is None:
        args.subframes = 2     # 2 por defecto
    if args.cooling_interval is None:
        args.cooling_interval = 50 if is_vast else 20


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    is_vast = detect_vast()
    apply_env_defaults(args, is_vast)

    # Configurar GL ANTES de importar mujoco (solo afecta a render).
    setup_gl_backend(args.mode, is_vast)

    # Semilla base reproducible.
    if args.seed is None:
        args.seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
    print(f"[INFO] base_seed={args.seed} | mode={args.mode} | n={args.n} | "
          f"vast={is_vast} | MUJOCO_GL={os.environ.get('MUJOCO_GL','(default)')}",
          flush=True)

    # --- Imports diferidos (tras configurar el backend GL) ---
    import mujoco
    from physics.top_model import build_top_model_xml
    from physics.initial_conditions import (set_initial_conditions,
                                            angle_from_vertical_deg)
    from rendering.floor_generator import generate_floor
    from dataset.state_logger import StateLogger
    from dataset.frame_writer import FrameWriter
    from dataset.video_writer import VideoWriter
    from utils.domain_randomization import sample_episode_params
    from utils.cooling import CoolingManager

    mods = dict(
        mujoco=mujoco,
        build_top_model_xml=build_top_model_xml,
        set_initial_conditions=set_initial_conditions,
        angle_from_vertical_deg=angle_from_vertical_deg,
        generate_floor=generate_floor,
        StateLogger=StateLogger,
        FrameWriter=FrameWriter,
        VideoWriter=VideoWriter,
    )
    if args.mode == "video":
        from rendering.renderer import TopRenderer
        from rendering.postprocess import PostProcessor
        mods["TopRenderer"] = TopRenderer
        mods["PostProcessor"] = PostProcessor

    validate_physics = None
    if not args.no_validate:
        from utils.validation import validate_physics

    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **k):
            return x

    os.makedirs(args.out, exist_ok=True)
    cooling = CoolingManager(
        args.cooling_interval,
        sleep_seconds=(0.0 if args.no_cooling_sleep else 30.0),
    )

    start_ep = max(0, args.resume_from)
    if start_ep > 0:
        prog = read_progress(args.out)
        if prog:
            print(f"[RESUME] progress.json: last_episode="
                  f"{prog.get('last_episode')} base_seed={prog.get('base_seed')}")

    t0 = time.time()
    n_valid = 0
    for ep in tqdm(range(start_ep, args.n), desc=f"sim[{args.mode}]"):
        ep_seed = episode_seed(args.seed, ep)
        params = sample_episode_params(ep_seed, ep, args.mode, args.top_type)
        try:
            meta = simulate_episode(mods, params, args)
        except Exception as e:
            print(f"[ERROR] episodio {ep} fallo: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

        valid = True
        if validate_physics is not None and "_states_path" in meta:
            valid, _ = validate_physics(meta["_states_path"])
        n_valid += int(valid)

        # Informe por episodio: cuando y como cae el trompo.
        ft = meta.get("_fall_time")
        sim = meta.get("simulation", {})
        ft_str = f"{ft:.2f}s" if ft is not None else "no cae"
        print(f"[EP {ep:04d}] type={params.top_type:5s} fall={ft_str} "
              f"dur={sim.get('duration_seconds')}s "
              f"states={sim.get('total_state_frames')} "
              f"frames={meta.get('_n_frames', '-')} valid={valid}", flush=True)

        write_progress(args.out, {
            "base_seed": args.seed,
            "mode": args.mode,
            "n": args.n,
            "top_type": args.top_type,
            "last_episode": ep,
            "completed": ep + 1,
            "n_valid": n_valid,
            "timestamp": time.time(),
        })
        cooling.tick(ep + 1, args.n)

    dt = time.time() - t0
    print(f"[DONE] {args.n - start_ep} episodios en {dt/60:.1f} min | "
          f"validos={n_valid} | salida='{args.out}'", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
