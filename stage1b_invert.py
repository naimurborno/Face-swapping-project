# stage1b_invert.py
"""
Stage 1b — DDIM inversion of the aligned donor image.

Runs DDIM inversion on donor_aligned_pil.png, captures K,V tensors from
shallow attention layers at each timestep, and saves them to
artifacts/kv_store/ for per-step injection in stage2_diffusion.py.

Reads from artifacts/:
    donor_aligned_pil.png   — aligned donor image (written by stage1_segment.py)
    meta.json               — updated in-place with stage1b completion fields

Writes to artifacts/kv_store/:
    step{t:03d}_{layer_name}.pt   — {"k": tensor, "v": tensor} per step per layer

Updates meta.json with:
    stage1b_complete      : true
    kv_store_dir          : absolute path to kv_store/
    num_inversion_steps   : number of inversion steps run
    layers_stored         : list of shallow layer names captured

Usage:
    python stage1b_invert.py
    python stage1b_invert.py --steps 30
    python stage1b_invert.py --artifacts artifacts/ --config configs/default.yaml
    python stage1b_invert.py --dry-run
"""

import sys
import os
import json
import argparse
import traceback

import torch
import yaml
from PIL import Image


# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")
if os.path.isdir(_CORE):
    sys.path.insert(0, _HERE)

try:
    from core.kv_cache   import KVCache
    from core.patch_unet import patch_unet_shallow_only
except ImportError as e:
    print(f"[stage1b] FATAL — could not import core modules: {e}")
    traceback.print_exc()
    sys.exit(1)


DEFAULT_CONFIG_PATH = os.path.join(_HERE, "configs", "default.yaml")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[stage1b] Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.artifacts is not None:
        cfg["paths"]["artifacts_dir"] = args.artifacts
    if args.steps is not None:
        cfg["stage1b"]["num_inversion_steps"] = args.steps
    return cfg


def print_config(cfg: dict):
    s1b = cfg.get("stage1b", {})
    print("\n┌─ Stage 1b config ─────────────────────────────────────────")
    print(f"│  artifacts_dir      : {cfg['paths']['artifacts_dir']}")
    print(f"│  num_inversion_steps: {s1b.get('num_inversion_steps', 30)}")
    print(f"│  torch_dtype        : {s1b.get('torch_dtype', 'float16')}")
    print(f"│  enable_cpu_offload : {s1b.get('enable_cpu_offload', False)}")
    print(f"│  model_id           : {cfg['stage2']['model_id']}")
    print(f"└───────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline(cfg: dict):
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler

    model_id  = cfg["stage2"]["model_id"]
    dtype_str = cfg["stage1b"].get("torch_dtype", "float16")
    dtype     = torch.float16 if dtype_str == "float16" else torch.float32
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("[stage1b] WARNING: No GPU detected — running on CPU. Using float32.")
        dtype = torch.float32

    print(f"[stage1b] Loading {model_id} ({dtype_str}) ...")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype    = dtype,
        safety_checker = None,
    )

    # DDIM is required for deterministic inversion
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        set_alpha_to_one = False,
        steps_offset     = 1,
    )

    if cfg["stage1b"].get("enable_cpu_offload", False):
        print("[stage1b] CPU offload enabled.")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    pipe.unet.eval()
    print(f"[stage1b] Pipeline ready on {device}")
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# VAE ENCODE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_donor(pipe, donor_pil: Image.Image) -> torch.Tensor:
    device    = next(pipe.vae.parameters()).device
    vae_dtype = next(pipe.vae.parameters()).dtype
    pixel     = pipe.image_processor.preprocess(donor_pil).to(device, dtype=vae_dtype)
    latent    = pipe.vae.encode(pixel).latent_dist.sample()
    return latent * pipe.vae.config.scaling_factor


# ══════════════════════════════════════════════════════════════════════════════
# DDIM INVERSION LOOP
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inversion(
    pipe,
    z0:           torch.Tensor,
    kv_cache:     KVCache,
    kv_store_dir: str,
    num_steps:    int,
    verbose:      bool = True,
) -> list:
    device     = next(pipe.unet.parameters()).device
    unet_dtype = next(pipe.unet.parameters()).dtype

    pipe.scheduler.set_timesteps(num_steps)
    timesteps = pipe.scheduler.timesteps  # descending: [T, ..., dt]

    text_inputs = pipe.tokenizer(
        [""],
        padding        = "max_length",
        max_length     = pipe.tokenizer.model_max_length,
        truncation     = True,
        return_tensors = "pt",
    )
    # conditional embedding only — same as PnP
    encoder_hidden_states = pipe.text_encoder(
        text_inputs.input_ids.to(device)
    )[0].to(dtype=unet_dtype)

    latents     = z0.clone().to(dtype=unet_dtype)
    layers_seen = []

    B, C, H, W = latents.shape
    mask_dummy = torch.zeros(B, 1, H, W, device=device, dtype=unet_dtype)

    # reversed: low t → high t  (same as PnP: reversed(scheduler.timesteps))
    reversed_timesteps = list(reversed(timesteps))

    for step_idx, t in enumerate(reversed_timesteps):
        if verbose:
            print(f"[stage1b] Inversion step {step_idx + 1}/{num_steps}  t={int(t)}", end="  ")

        # 9-channel input: masked_latent mirrors current latent (zero mask = full donor visible)
        unet_input = torch.cat([latents, mask_dummy, latents.clone()], dim=1)

        kv_cache.clear()
        kv_cache.set_freq_mode("hf")
        kv_cache.set_mode("store")

        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float32):
            eps = pipe.unet(
                unet_input,
                t,
                encoder_hidden_states  = encoder_hidden_states,
                cross_attention_kwargs = {"kv_cache": kv_cache},
                return_dict            = False,
            )[0]

            # ── PnP manual DDIM inversion update ──────────────────────────
            # Matches Tumanyan et al. exactly:
            #   pred_x0 = (z_t - σ_prev · ε) / μ_prev
            #   z_{t+1} = μ · pred_x0 + σ · ε
            t_prev = reversed_timesteps[step_idx - 1] if step_idx > 0 else t

            alpha_prod_t      = pipe.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                pipe.scheduler.alphas_cumprod[t_prev]
                if step_idx > 0
                else pipe.scheduler.final_alpha_cumprod
            )

            mu      = alpha_prod_t      ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma   = (1 - alpha_prod_t)      ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            pred_x0 = (latents - sigma_prev * eps) / mu_prev
            latents = mu * pred_x0 + sigma * eps

        # save K,V + noisy latent for this timestep
        kv_cache.save_step(step_idx, kv_store_dir, noisy_latent=latents)
        kv_cache.set_mode("bypass")

        if step_idx == 0:
            layers_seen = kv_cache.hf_keys()
            print(f"captured {len(layers_seen)} shallow layers")
        elif verbose:
            print(f"saved {len(kv_cache.hf_keys())} layers")

    kv_cache.set_mode("bypass")
    print(f"[stage1b] Inversion complete. kv_store: {kv_store_dir}")
    return layers_seen


# ══════════════════════════════════════════════════════════════════════════════
# META UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def update_meta(
    artifacts_dir:  str,
    kv_store_dir:   str,
    num_steps:      int,
    layers_stored:  list,
):
    """Patch meta.json in-place with stage1b completion fields."""
    meta_path = os.path.join(artifacts_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    meta["stage1b_complete"]    = True
    meta["kv_store_dir"]        = os.path.abspath(kv_store_dir)
    meta["num_inversion_steps"] = num_steps
    meta["layers_stored"]       = layers_stored

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[stage1b] meta.json updated: stage1b_complete=true  layers={len(layers_stored)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_stage1b(cfg: dict):
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    s1b_cfg       = cfg.get("stage1b", {})
    num_steps     = s1b_cfg.get("num_inversion_steps", 30)
    verbose       = cfg.get("logging", {}).get("verbose", True)

    # ── Load donor image ──────────────────────────────────────────────────
    donor_path = os.path.join(artifacts_dir, "donor_aligned_pil.png")
    if not os.path.exists(donor_path):
        raise FileNotFoundError(
            f"[stage1b] donor_aligned_pil.png not found in '{artifacts_dir}'.\n"
            f"  Run stage1_segment.py first."
        )
    donor_pil = Image.open(donor_path).convert("RGB")
    print(f"[stage1b] Loaded donor: {donor_pil.size}")

    # ── Load pipeline + patch shallow layers ──────────────────────────────
    pipe = load_pipeline(cfg)
    pipe.unet, depth_map = patch_unet_shallow_only(pipe.unet)
    n_shallow = sum(1 for v in depth_map.values() if v == "shallow")
    print(f"[stage1b] UNet patched — {n_shallow} shallow KVInjectionAttention layers")

    # ── VAE encode ────────────────────────────────────────────────────────
    print("[stage1b] VAE-encoding donor ...")
    z0 = encode_donor(pipe, donor_pil)
    print(f"[stage1b] z0 shape: {tuple(z0.shape)}")

    # ── KV cache + kv_store dir ───────────────────────────────────────────
    kv_cache     = KVCache()
    kv_store_dir = os.path.join(artifacts_dir, "kv_store")
    os.makedirs(kv_store_dir, exist_ok=True)
    print(f"[stage1b] kv_store → {os.path.abspath(kv_store_dir)}")

    # ── Run inversion ─────────────────────────────────────────────────────
    layers_stored = run_inversion(
        pipe         = pipe,
        z0           = z0,
        kv_cache     = kv_cache,
        kv_store_dir = kv_store_dir,
        num_steps    = num_steps,
        verbose      = verbose,
    )

    # ── Update meta.json ──────────────────────────────────────────────────
    update_meta(artifacts_dir, kv_store_dir, num_steps, layers_stored)

    print(
        f"\n[stage1b] Done. {num_steps} steps × {len(layers_stored)} layers "
        f"saved to {kv_store_dir}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1b — DDIM inversion of aligned donor, captures K,V for PnP injection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",    default=DEFAULT_CONFIG_PATH,
                   help="Path to YAML config")
    p.add_argument("--artifacts", default=None,
                   help="Override paths.artifacts_dir")
    p.add_argument("--steps",     type=int, default=None,
                   help="Override stage1b.num_inversion_steps")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print config and exit without running")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"[stage1b] FATAL — {e}")
        sys.exit(1)

    cfg = apply_cli_overrides(cfg, args)
    print_config(cfg)

    if args.dry_run:
        print("[stage1b] --dry-run: exiting.\n")
        sys.exit(0)

    try:
        run_stage1b(cfg)
    except FileNotFoundError as e:
        print(f"\n[stage1b] FATAL — {e}\n")
        sys.exit(1)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(
                f"\n[stage1b] FATAL — CUDA OOM: {e}\n"
                f"  Options:\n"
                f"  1. Set stage1b.enable_cpu_offload: true in default.yaml\n"
                f"  2. Lower stage1b.num_inversion_steps\n"
            )
        else:
            print(f"\n[stage1b] FATAL — {e}\n")
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[stage1b] FATAL — unexpected error: {e}\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()