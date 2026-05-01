# stage2_diffusion.py
"""
Stage 2 — PnP KV Inpainting with Blended Latent Anchoring.

Reads everything from artifacts/ (written by stage1_segment.py and
stage1b_invert.py). Loads Stable Diffusion Inpainting, patches shallow
U-Net layers, runs blended-latent-anchored denoising with per-step
PnP KV injection from the pre-computed kv_store.

Prior construction has been removed. Texture transfer is handled entirely
by loading timestep-matched donor K,V tensors into the KVCache at each
denoising step, injecting them into shallow attention layers via
KVInjectionAttention.

All it needs from disk:
    artifacts/content_pil.png       — content image S (structure to keep)
    artifacts/donor_aligned_pil.png — aligned donor R̃ (for VAE encode only)
    artifacts/object_mask.pt        — (1,1,S,S) float32 mask tensor M
    artifacts/meta.json             — ablation flags + stage1b completion check
    artifacts/kv_store/             — per-timestep K,V tensors from stage1b

Denoising loop (per timestep t):
    ┌─ Load KV for timestep t (PnP) ───────────────────────────────────────────┐
    │  kv_cache ← kv_store[t][layer]  for each shallow layer                  │
    │  replaces the single static store pass from the old pipeline             │
    └──────────────────────────────────────────────────────────────────────────┘
    ┌─ Blended latent anchoring (pre-step) ────────────────────────────────────┐
    │  z_S_noised = add_noise(z_S, noise, t)                                   │
    │  latents    = Mz ⊙ latents + (1−Mz) ⊙ z_S_noised                       │
    └──────────────────────────────────────────────────────────────────────────┘
    ┌─ Inject pass ──────────────────────────────────────────────────────────── ┐
    │  noise_pred = unet(latents, t) with timestep-matched donor KV injected   │
    │  shallow layers ← kv_store[t][l] × λ_HF                                 │
    │  deep layers   ← standard attention, untouched                           │
    └──────────────────────────────────────────────────────────────────────────┘
    ┌─ Scheduler step ──────────────────────────────────────────────────────────┐
    │  latents = scheduler.step(noise_pred, t, latents).prev_sample            │
    └──────────────────────────────────────────────────────────────────────────┘
    ┌─ Blended latent anchoring (post-step) ───────────────────────────────────┐
    │  latents = Mz ⊙ latents + (1−Mz) ⊙ z_S_noised_next                     │
    └──────────────────────────────────────────────────────────────────────────┘

Key difference from old pipeline:
    OLD: _shallow_store_pass() — ONE UNet forward of the donor at a single
         representative mid-noise timestep. Same KV injected at every step.
         Temporal annealing (λ ramp) was a heuristic to compensate.
    NEW: kv_store[t] loaded per step from artifacts/kv_store/.
         KV is already timestep-calibrated — no heuristic needed.
         temporal_anneal becomes an optional ablation, not a requirement.

Ablation flags (configs/default.yaml → ablation.*):
    blended_anchoring   : true  → per-step latent replacement (proposed)
                          false → one-time img2img init only (baseline)
    shallow_injection   : true  → PnP KV injection (proposed)
                          false → pure inpainting, no KV (baseline)
    temporal_anneal     : true  → λ_HF ramps 0→injection_scale (now optional)
                          false → flat λ_HF = injection_scale (principled default)

Usage:
    python stage2_diffusion.py
    python stage2_diffusion.py --injection-scale 0.5
    python stage2_diffusion.py --no-blended-anchoring
    python stage2_diffusion.py --no-shallow-injection
    python stage2_diffusion.py --anneal
    python stage2_diffusion.py --steps 50 --guidance 9.0 --seed 123
    python stage2_diffusion.py --artifacts artifacts/ --output-dir outputs/
    python stage2_diffusion.py --dry-run

Output:
    outputs/result.png           — final output image
    outputs/comparison.png       — 4-panel: Content | Donor | Masked | Result
    outputs/meta_stage2.json     — complete Stage 2 config for reproducibility

Dependencies:
    pip install diffusers transformers accelerate torch pillow pyyaml
"""

import sys
import os
import json
import argparse
import datetime
import traceback

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")
if os.path.isdir(_CORE):
    sys.path.insert(0, _HERE)

try:
    from core.kv_cache    import KVCache
    from core.patch_unet  import patch_unet_shallow_only, patch_unet_attention
    from core.compositing import run_compositing
except ImportError as e:
    print(f"\n[stage2] FATAL — could not import core modules: {e}")
    print("  Make sure core/ exists and contains kv_cache.py, patch_unet.py,")
    print("  kv_attention.py, mask_utils.py. Run from the project root directory.")
    traceback.print_exc()
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADER
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG_PATH = os.path.join(_HERE, "configs", "default.yaml")


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[stage2] Config not found: {config_path}\n"
            f"  Expected at: {os.path.abspath(config_path)}"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.artifacts is not None:
        cfg["paths"]["artifacts_dir"] = args.artifacts
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = args.output_dir
    if args.injection_scale is not None:
        cfg["injection"]["injection_scale"] = args.injection_scale
    if args.anneal:
        cfg["ablation"]["temporal_anneal"] = True
    if args.no_blended_anchoring:
        cfg["ablation"]["blended_anchoring"] = False
    if args.no_shallow_injection:
        cfg["ablation"]["shallow_injection"] = False
    if args.steps is not None:
        cfg["stage2"]["num_inference_steps"] = args.steps
    if args.guidance is not None:
        cfg["stage2"]["guidance_scale"] = args.guidance
    if args.seed is not None:
        cfg["stage2"]["seed"] = args.seed
    if args.prompt is not None:
        cfg["stage2"]["prompt"] = args.prompt
    if args.model_id is not None:
        cfg["stage2"]["model_id"] = args.model_id
    return cfg


def print_config(cfg: dict, meta: dict):
    """Print the resolved Stage 2 config alongside key Stage 1 stats."""
    s1b_done = meta.get("stage1b_complete", False)
    print("\n┌─ Stage 2 config ──────────────────────────────────────────")
    print(f"│  model_id          : {cfg['stage2']['model_id']}")
    print(f"│  target_size       : {cfg['image']['target_size']}px")
    print(f"│  steps             : {cfg['stage2']['num_inference_steps']}")
    print(f"│  guidance          : {cfg['stage2']['guidance_scale']}")
    print(f"│  seed              : {cfg['stage2']['seed']}")
    print(f"│  prompt            : {cfg['stage2']['prompt'][:60]}...")
    print(f"│")
    print(f"│  [ablation]")
    print(f"│  blended_anchoring : {cfg['ablation'].get('blended_anchoring', True)}")
    print(f"│  shallow_injection : {cfg['ablation'].get('shallow_injection', True)}")
    print(f"│  temporal_anneal   : {cfg['ablation'].get('temporal_anneal', False)}  (flat λ is principled default)")
    print(f"│  injection_scale   : {cfg['injection']['injection_scale']}")
    print(f"│")
    print(f"│  [stage1b status]")
    print(f"│  stage1b_complete  : {s1b_done}  {'✓' if s1b_done else '⚠ run stage1b_invert.py first'}")
    print(f"│  kv_store_dir      : {meta.get('kv_store_dir', 'not set')}")
    print(f"│  inversion_steps   : {meta.get('num_inversion_steps', 'not set')}")
    print(f"│  layers_stored     : {meta.get('layers_stored', 'not set')}")
    print(f"│")
    print(f"│  [stage1 stats]")
    print(f"│  alignment_method  : {meta.get('alignment_method', '?')}")
    print(f"│  mask_coverage     : {meta.get('mask_coverage_pct', '?')}%")
    print(f"└───────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_artifacts(artifacts_dir: str) -> dict:
    """
    Load all Stage 1 / Stage 1b artifacts from disk.

    Required files:
        content_pil.png       — content image S
        donor_aligned_pil.png — aligned donor R̃
        object_mask.pt        — (1,1,S,S) float32 mask tensor M
        meta.json             — stage1 + stage1b config + stats

    Required directory (written by stage1b_invert.py):
        kv_store/             — per-timestep K,V tensors

    Note: masked_input_pil.png and prior_pil.png are no longer produced
    by Stage 1. Latent initialisation uses content_pil.png directly.

    Raises:
        FileNotFoundError — if any required file is missing
        RuntimeError      — if stage1b_complete is False in meta.json
    """
    required_files = {
        "content_pil.png"       : "PIL",
        "donor_aligned_pil.png" : "PIL",
        "object_mask.pt"        : "tensor",
        "meta.json"             : "json",
    }

    missing = [
        fname for fname in required_files
        if not os.path.exists(os.path.join(artifacts_dir, fname))
    ]
    if missing:
        raise FileNotFoundError(
            f"[stage2] Missing artifacts in '{artifacts_dir}': {missing}\n"
            f"  Run stage1_segment.py first.\n"
            f"  If you used a custom artifacts_dir, pass it with --artifacts."
        )

    def _path(fname):
        return os.path.join(artifacts_dir, fname)

    content_pil       = Image.open(_path("content_pil.png")).convert("RGB")
    donor_aligned_pil = Image.open(_path("donor_aligned_pil.png")).convert("RGB")
    face_mask         = torch.load(_path("object_mask.pt"), map_location="cpu")

    with open(_path("meta.json"), "r") as f:
        meta = json.load(f)

    # ── Stage 1b completion check ─────────────────────────────────────────
    if not meta.get("stage1b_complete", False):
        raise RuntimeError(
            "[stage2] Stage 1b has not completed.\n"
            "  meta.json shows stage1b_complete=False.\n"
            "  Run stage1b_invert.py before stage2_diffusion.py.\n"
            "  The kv_store/ directory must exist and be populated."
        )

    # ── kv_store directory check ──────────────────────────────────────────
    kv_store_dir = meta.get("kv_store_dir") or os.path.join(artifacts_dir, "kv_store")
    if not os.path.isdir(kv_store_dir):
        raise FileNotFoundError(
            f"[stage2] kv_store directory not found: {kv_store_dir}\n"
            f"  Run stage1b_invert.py to populate it."
        )

    print(f"[stage2] Artifacts loaded from '{os.path.abspath(artifacts_dir)}'")
    print(
        f"[stage2]   content={content_pil.size} | donor={donor_aligned_pil.size} | "
        f"mask={tuple(face_mask.shape)}"
    )
    print(
        f"[stage2]   kv_store={kv_store_dir} | "
        f"inversion_steps={meta.get('num_inversion_steps', '?')} | "
        f"layers={meta.get('layers_stored', '?')}"
    )

    return {
        "content_pil"       : content_pil,
        "donor_aligned_pil" : donor_aligned_pil,
        "face_mask"         : face_mask,
        "meta"              : meta,
        "kv_store_dir"      : kv_store_dir,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline(cfg: dict):
    """
    Load Stable Diffusion Inpainting pipeline and set up the scheduler.

    Uses StableDiffusionInpaintPipeline (9-channel UNet) with the
    stabilityai/stable-diffusion-2-inpainting checkpoint.
    """
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, PNDMScheduler
    try:
        from diffusers import DPMSolverMultistepScheduler
        _has_dpm = True
    except ImportError:
        _has_dpm = False

    model_id  = cfg["stage2"]["model_id"]
    dtype_str = cfg["stage2"]["torch_dtype"]
    dtype     = torch.float16 if dtype_str == "float16" else torch.float32
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print(
            "[stage2] WARNING: No GPU detected — running on CPU. "
            "This will be very slow. Using float32."
        )
        dtype = torch.float32

    print(f"[stage2] Loading {model_id} ({dtype_str}) ...")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype    = dtype,
        safety_checker = None,
    )

    sched_name = cfg["stage2"].get("scheduler", "DDIM")
    if sched_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif sched_name == "PNDM":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif sched_name == "DPMSolver" and _has_dpm:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
    else:
        print(f"[stage2] Unknown scheduler '{sched_name}', falling back to DDIM.")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if cfg["stage2"].get("enable_cpu_offload", False):
        print("[stage2] CPU offload enabled (low VRAM mode).")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    pipe.unet.eval()
    print(f"[stage2] Pipeline ready on {device} | scheduler={sched_name}")

    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# VAE ENCODING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _encode_to_latent(vae, image_processor, pil_image: Image.Image) -> torch.Tensor:
    """
    Encode a PIL RGB image to a VAE latent tensor.

    Returns:
        (1, 4, H/8, W/8) latent tensor scaled by vae.config.scaling_factor
    """
    device    = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    pixel     = image_processor.preprocess(pil_image).to(device, dtype=vae_dtype)
    latent    = vae.encode(pixel).latent_dist.sample()
    return latent * vae.config.scaling_factor


def _build_inpaint_conditioning_pil(
    content_pil: Image.Image,
    face_mask:   torch.Tensor,
) -> Image.Image:
    """
    Build the inpainting conditioning image for the 9-channel UNet.

    Standard inpainting: source pixels outside the mask, zeroed inside.
    Without a prior image to fill the hole with, we zero the masked region
    so the UNet sees a clean separation between known and unknown pixels.
    The donor texture enters entirely through KV injection — not through
    the conditioning image.

    Args:
        content_pil : Content image S (PIL RGB, target resolution)
        face_mask   : (1, 1, H, W) float32 mask tensor [0, 1]

    Returns:
        PIL RGB conditioning image — source outside mask, black inside mask
    """
    W, H = content_pil.size
    mask_resized = F.interpolate(
        face_mask.float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    mask_np = np.clip(mask_resized.squeeze().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

    # Invert mask for paste: paste black over the masked (editable) region
    mask_inv = Image.fromarray(255 - mask_np, mode="L")
    conditioning = content_pil.copy()
    black = Image.new("RGB", conditioning.size, (0, 0, 0))
    conditioning.paste(black, mask=mask_inv)
    return conditioning


# ══════════════════════════════════════════════════════════════════════════════
# MASK DOWNSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def _downsample_mask(
    face_mask: torch.Tensor,
    latent_h:  int,
    latent_w:  int,
    *,
    binary:     bool = True,
    feather_px: int  = 0,
) -> torch.Tensor:
    """
    Downsample pixel-space mask to latent spatial resolution (H/8 × W/8).

    Args:
        face_mask  : (1, 1, H, W) float32 [0, 1]
        latent_h   : latent height (H/8)
        latent_w   : latent width  (W/8)
        binary     : True → hard {0,1} mask; False → soft feathered mask
        feather_px : pixels to feather in pixel space (converted to latent units)

    Returns:
        (1, 1, latent_h, latent_w) float32 mask
    """
    m = F.interpolate(
        face_mask,
        size=(latent_h, latent_w),
        mode="bilinear",
        align_corners=False,
    )
    if binary:
        return (m > 0.5).float()

    if feather_px > 0:
        latent_radius = max(1, int(round(feather_px / 8.0)))
        kernel = latent_radius * 2 + 1
        m = F.avg_pool2d(m, kernel_size=kernel, stride=1, padding=latent_radius)

    return m.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# BLENDED LATENT ANCHORING
# ══════════════════════════════════════════════════════════════════════════════

def _blend_latents(
    latents:    torch.Tensor,
    z_S_noised: torch.Tensor,
    Mz:         torch.Tensor,
) -> torch.Tensor:
    """
    Blended latent anchoring — enforce source structure outside the mask.

    Inside mask  (Mz = 1): latent unchanged — diffusion fills freely
    Outside mask (Mz = 0): latent ← z_S_noised — source is frozen

    Applied twice per step (pre and post scheduler) to prevent drift.
    """
    return Mz * latents + (1.0 - Mz) * z_S_noised


# ══════════════════════════════════════════════════════════════════════════════
# PnP KV LOADER  ★ replaces _shallow_store_pass from old pipeline
# ══════════════════════════════════════════════════════════════════════════════

def load_kv_for_timestep(
    kv_cache:    KVCache,
    kv_store_dir: str,
    step_idx:    int,
    layers:      list,
    device:      torch.device,
    dtype:       torch.dtype,
):
    kv_cache._hf_cache.clear()

    loaded = 0
    for layer_name in layers:
        safe_name = layer_name.replace(".", "_")
        fpath     = os.path.join(kv_store_dir, f"step{step_idx:03d}_{safe_name}.pt")

        if not os.path.exists(fpath):
            if step_idx == 0:
                print(f"[stage2] WARNING: kv_store missing {os.path.basename(fpath)} — layer skipped")
            continue

        kv = torch.load(fpath, map_location=device)
        kv_cache._hf_cache[layer_name] = (kv["k"].to(dtype=dtype), kv["v"].to(dtype=dtype))
        loaded += 1

    # load noisy latent saved by stage1b for this timestep
    latent_path   = os.path.join(kv_store_dir, f"step{step_idx:03d}_noisy_latent.pt")
    noisy_latent  = None
    if os.path.exists(latent_path):
        noisy_latent = torch.load(latent_path, map_location=device).to(dtype=dtype)

    return loaded, noisy_latent


def _get_shallow_layer_names(depth_map: dict) -> list:
    """
    Extract the list of shallow layer names from the depth map returned
    by patch_unet_shallow_only().

    Args:
        depth_map : {layer_name: "shallow" | "deep" | None}

    Returns:
        List of layer name strings where depth == "shallow"
    """
    return [name for name, depth in depth_map.items() if depth == "shallow"]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DENOISING LOOP
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_denoising_loop(
    pipe,
    artifacts: dict,
    cfg:       dict,
    depth_map: dict,
) -> torch.Tensor:
    """
    Inpainting denoising loop with blended latent anchoring and
    per-step PnP KV injection.

    Key difference from old pipeline:
        OLD: _shallow_store_pass() once before the loop → same KV every step
        NEW: load_kv_for_timestep() at the start of each step → timestep-
             matched KV from DDIM inversion, more faithful texture transfer

    Args:
        pipe      : Loaded + patched StableDiffusionInpaintPipeline
        artifacts : dict from load_artifacts()
        cfg       : Resolved config dict
        depth_map : {layer_name: depth} from patch_unet_shallow_only()

    Returns:
        latents : (1, 4, H/8, W/8) final denoised latent (pre-decode)
    """
    device = next(pipe.unet.parameters()).device

    # ── Config ────────────────────────────────────────────────────────────
    anneal         = cfg["ablation"].get("temporal_anneal",   False)  # flat is principled default
    blended_anchor = cfg["ablation"].get("blended_anchoring", True)
    shallow_inject = cfg["ablation"].get("shallow_injection", True)
    scale          = cfg["injection"]["injection_scale"]
    steps          = cfg["stage2"]["num_inference_steps"]
    g_scale        = cfg["stage2"]["guidance_scale"]
    seed           = cfg["stage2"]["seed"]
    prompt         = cfg["stage2"]["prompt"]
    neg            = cfg["stage2"].get("negative_prompt", "")
    verbose        = cfg.get("logging", {}).get("verbose", True)
    denoise_str    = float(cfg["stage2"].get("denoising_strength", 0.75))
    denoise_str    = max(0.0, min(1.0, denoise_str))
    feather_px     = int(cfg["stage2"].get("anchor_feather_px", 24))

    do_cfg         = g_scale > 1.0
    kv_store_dir   = artifacts["kv_store_dir"]
    shallow_layers = _get_shallow_layer_names(depth_map)

    # ── KV cache setup ────────────────────────────────────────────────────
    kv_cache               = KVCache()
    kv_cache.depth_routing = "correct"
    kv_cache.face_mask     = artifacts["face_mask"].to(device)

    # ── Encode images to latents ──────────────────────────────────────────
    _section("Encoding images to latents")

    # z_X0 — content image latent: denoising start point
    # (replaces masked_input_pil which no longer exists)
    z_X0 = _encode_to_latent(
        pipe.vae, pipe.image_processor, artifacts["content_pil"]
    )

    # z_S — clean source latent: blended latent anchoring reference
    z_S = _encode_to_latent(
        pipe.vae, pipe.image_processor, artifacts["content_pil"]
    )
    # z_X0 and z_S are identical since we no longer embed a prior into X0.
    # z_S is kept as a named variable for clarity in the anchoring logic.

    latent_h, latent_w = z_X0.shape[-2], z_X0.shape[-1]

    # z_S_masked — inpainting conditioning: source outside mask, black inside
    conditioning_pil = _build_inpaint_conditioning_pil(
        artifacts["content_pil"],
        artifacts["face_mask"],
    )
    z_S_masked = _encode_to_latent(pipe.vae, pipe.image_processor, conditioning_pil)

    print(
        f"[stage2] Encoded: z_X0={tuple(z_X0.shape)} | "
        f"z_S={tuple(z_S.shape)} | z_S_masked={tuple(z_S_masked.shape)}"
    )

    # ── Latent-space masks ────────────────────────────────────────────────
    Mz_edit = _downsample_mask(
        artifacts["face_mask"], latent_h, latent_w, binary=True
    ).to(device=device, dtype=z_X0.dtype)

    Mz_anchor = _downsample_mask(
        artifacts["face_mask"], latent_h, latent_w,
        binary=False, feather_px=feather_px,
    ).to(device=device, dtype=z_X0.dtype)

    # ── Prompt embeddings ─────────────────────────────────────────────────
    _section("Encoding prompts")

    prompt_embeds, neg_embeds = pipe.encode_prompt(
        prompt                      = prompt,
        device                      = device,
        num_images_per_prompt       = 1,
        do_classifier_free_guidance = do_cfg,
        negative_prompt             = neg,
    )

    text_embeds = torch.cat([neg_embeds, prompt_embeds]) if do_cfg \
                  else prompt_embeds
    ref_embeds  = neg_embeds if do_cfg else prompt_embeds

    print(f"[stage2] Prompt embeds: {tuple(prompt_embeds.shape)} | CFG={do_cfg}")

    # ── Scheduler timesteps ───────────────────────────────────────────────
    pipe.scheduler.set_timesteps(steps, device=device)
    all_timesteps = pipe.scheduler.timesteps

    init_timestep = min(int(steps * denoise_str), steps)
    t_start       = max(steps - init_timestep, 0)
    timesteps     = all_timesteps[t_start:]
    if len(timesteps) == 0:
        timesteps = all_timesteps[-1:]
    T = len(timesteps)

    # ── Initial noisy latent ──────────────────────────────────────────────
    gen   = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z_X0.shape, generator=gen, device=device, dtype=z_X0.dtype)
    latents = pipe.scheduler.add_noise(z_X0, noise, timesteps[:1])

    print(
        f"[stage2] Starting denoising | requested_steps={steps} | "
        f"active_steps={T} | denoising_strength={denoise_str:.2f} | seed={seed}"
    )
    print(
        f"[stage2] blended_anchoring={blended_anchor} | "
        f"shallow_injection={shallow_inject} | "
        f"temporal_anneal={anneal} | injection_scale={scale}"
    )
    print(
        f"[stage2] PnP KV mode: timestep-matched per-step load | "
        f"shallow_layers={len(shallow_layers)}"
    )

    _section(f"Denoising loop  [T={T} steps]")

    # ── Denoising loop ────────────────────────────────────────────────────
    for step_idx, t in enumerate(timesteps):

        # ── Lambda update ─────────────────────────────────────────────────
        # temporal_anneal is now optional — flat λ is the principled default
        # because KV tensors are already timestep-matched from DDIM inversion.
        # Annealing is kept as an ablation flag (A4) to compare against flat.
        progress  = step_idx / max(T - 1, 1)
        lambda_hf = (progress * scale) if anneal else scale
        lambda_hf = lambda_hf if shallow_inject else 0.0

        kv_cache.lambda_lf = 0.0
        kv_cache.lambda_hf = lambda_hf

        # ── Load PnP KV for this timestep  ★ ─────────────────────────────
        # Replaces the old static cache that was populated once before the loop.
        # kv_store files are indexed by denoising step (0-based), not by
        # scheduler timestep value, to match the inversion loop in stage1b.
        unet_dtype = next(pipe.unet.parameters()).dtype
        if shallow_inject:
            n_loaded, donor_noisy = load_kv_for_timestep(
                kv_cache, kv_store_dir, step_idx,
                shallow_layers, device, unet_dtype,
            )
            kv_cache.set_mode("inject")
        else:
            kv_cache.set_mode("bypass")
            n_loaded, donor_noisy = 0, None

        # initialise latents from saved donor noisy latent (PnP style)
        # blended anchoring then overwrites the outside-mask region
        if donor_noisy is not None:
            latents = donor_noisy

        if verbose and (step_idx % 5 == 0 or step_idx == T - 1):
            print(
                f"  step {step_idx + 1:3d}/{T} | t={int(t):4d} | "
                f"λ_HF={lambda_hf:.3f} | "
                f"kv_loaded={n_loaded} | "
                f"anchor={blended_anchor} | inject={shallow_inject}"
            )

        # ── Blended latent anchoring (pre-step) ───────────────────────────
        if blended_anchor:
            z_S_noised = pipe.scheduler.add_noise(z_S, noise, t.unsqueeze(0))
            latents    = _blend_latents(latents, z_S_noised, Mz_anchor)

        # ── Inject pass ───────────────────────────────────────────────────
        src_input = torch.cat([latents] * 2) if do_cfg else latents

        src_input  = src_input.to(dtype=unet_dtype)
        Mz_edit    = Mz_edit.to(dtype=unet_dtype)
        z_S_masked = z_S_masked.to(dtype=unet_dtype)

        mask_input    = torch.cat([Mz_edit]    * 2) if do_cfg else Mz_edit
        masked_lat_in = torch.cat([z_S_masked] * 2) if do_cfg else z_S_masked
        unet_input    = torch.cat([src_input, mask_input, masked_lat_in], dim=1)

        t_in = t.to(dtype=unet_dtype)
        t_in = t_in.unsqueeze(0) if t_in.ndim == 0 else t_in

        noise_pred = pipe.unet(
            unet_input,
            t_in,
            encoder_hidden_states  = text_embeds if do_cfg else prompt_embeds,
            cross_attention_kwargs = {"kv_cache": kv_cache},
        ).sample

        # CFG combine
        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + g_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        # ── Scheduler step ────────────────────────────────────────────────
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # ── Blended latent anchoring (post-step) ──────────────────────────
        if blended_anchor:
            if step_idx < T - 1:
                t_next       = timesteps[step_idx + 1]
                z_S_noised_n = pipe.scheduler.add_noise(
                    z_S, noise, t_next.unsqueeze(0)
                )
            else:
                z_S_noised_n = z_S
            latents = _blend_latents(latents, z_S_noised_n, Mz_anchor)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[stage2] Denoising complete. Final latent: {tuple(latents.shape)}")
    return latents


# ══════════════════════════════════════════════════════════════════════════════
# DECODE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def decode_latent(pipe, latents: torch.Tensor) -> Image.Image:
    vae_dtype = next(pipe.vae.parameters()).dtype
    latents   = (latents / pipe.vae.config.scaling_factor).to(dtype=vae_dtype)
    image     = pipe.vae.decode(latents).sample
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    result_pil:  Image.Image,
    artifacts:   dict,
    cfg:         dict,
    stage2_meta: dict,
    raw_pil:     Image.Image = None,
) -> str:
    """
    Save all outputs to the output directory.

    Files written:
        result.png          — final output image
        raw_diffusion.png   — pre-compositing decode (if compositing ran)
        comparison.png      — 4-panel: Content | Donor | Masked | Result
        meta_stage2.json    — complete Stage 2 config for reproducibility

    Comparison panel is 4 panels (was 5 — prior_pil removed).
    """
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    result_path = os.path.join(output_dir, "result.png")
    result_pil.save(result_path)

    if raw_pil is not None:
        raw_pil.save(os.path.join(output_dir, "raw_diffusion.png"))

    # ── 4-panel comparison (prior panel removed) ──────────────────────────
    panels = [
        artifacts["content_pil"],
        artifacts["donor_aligned_pil"],
        result_pil,
    ]
    labels = [
        "Content (S)",
        "Donor aligned (R̃)",
        "Result",
    ]

    W, H    = panels[0].size
    pad     = 8
    label_h = 36
    scale   = 3
    W, H    = W * scale, H * scale
    panels  = [p.resize((W, H), Image.LANCZOS) for p in panels]
    total_w = W * len(panels) + pad * (len(panels) + 1)
    total_h = H + label_h + pad * 2
    comparison = Image.new("RGB", (total_w, total_h), (30, 30, 30))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
            )
        except Exception:
            font = ImageFont.load_default()
    except ImportError:
        draw = None

    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = pad + i * (W + pad)
        y = pad
        comparison.paste(panel.resize((W, H)), (x, y))
        if draw is not None:
            draw.text((x + 4, y + H + 4), label, fill=(220, 220, 220), font=font)

    comparison.save(os.path.join(output_dir, "comparison.png"))

    # ── Stage 2 meta ───────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "meta_stage2.json"), "w") as f:
        json.dump(stage2_meta, f, indent=2)

    print(f"\n[stage2] Outputs saved → {os.path.abspath(output_dir)}/")
    for fname in ["result.png", "raw_diffusion.png", "comparison.png", "meta_stage2.json"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fname:<30}  {size_kb:7.1f} KB")

    return result_path


# ══════════════════════════════════════════════════════════════════════════════
# META BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_stage2_meta(cfg: dict, stage1_meta: dict) -> dict:
    """
    Build meta_stage2.json for full reproducibility.

    Prior-related fields (alpha, beta, gamma, decomp_method) removed.
    Stage 1b provenance (inversion_steps, kv_store_dir) added.
    temporal_anneal default changed to False (flat λ is now principled).
    """
    ablation_cfg = cfg.get("ablation", {})
    inj_cfg      = cfg.get("injection", {})

    return {
        # ── Stage 2 params ────────────────────────────────────────────────
        "model_id"            : cfg["stage2"]["model_id"],
        "pipeline_type"       : "inpainting",
        "scheduler"           : cfg["stage2"].get("scheduler", "DDIM"),
        "num_inference_steps" : cfg["stage2"]["num_inference_steps"],
        "denoising_strength"  : cfg["stage2"].get("denoising_strength", 0.75),
        "anchor_feather_px"   : cfg["stage2"].get("anchor_feather_px", 24),
        "guidance_scale"      : cfg["stage2"]["guidance_scale"],
        "seed"                : cfg["stage2"]["seed"],
        "prompt"              : cfg["stage2"]["prompt"],
        "negative_prompt"     : cfg["stage2"].get("negative_prompt", ""),
        "torch_dtype"         : cfg["stage2"]["torch_dtype"],

        # ── Injection params ──────────────────────────────────────────────
        "injection_scale"     : inj_cfg.get("injection_scale", 0.8),
        "injection_mode"      : "pnp_per_step",   # documents the new mechanism

        # ── Ablation flags ────────────────────────────────────────────────
        "ablation": {
            "blended_anchoring"  : ablation_cfg.get("blended_anchoring",  True),
            "shallow_injection"  : ablation_cfg.get("shallow_injection",  True),
            "temporal_anneal"    : ablation_cfg.get("temporal_anneal",    False),
            "compositing"        : ablation_cfg.get("compositing",        "freq"),
        },

        # ── Stage 1b provenance ───────────────────────────────────────────
        "stage1b": {
            "kv_store_dir"       : stage1_meta.get("kv_store_dir"),
            "num_inversion_steps": stage1_meta.get("num_inversion_steps"),
            "layers_stored"      : stage1_meta.get("layers_stored"),
        },

        # ── Stage 1 provenance ────────────────────────────────────────────
        "stage1_meta"         : stage1_meta,

        # ── Provenance ────────────────────────────────────────────────────
        "stage2_script"       : os.path.abspath(__file__),
        "timestamp"           : datetime.datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _print_final_summary(result_path: str, cfg: dict, stage1_meta: dict):
    print(f"\n{'═' * 60}")
    print(f"  Stage 2 complete — PnP KV Inpainting")
    print(f"{'═' * 60}")
    print(f"  result             : {os.path.abspath(result_path)}")
    print(f"  injection_mode     : pnp_per_step (timestep-matched KV)")
    print(f"  blended_anchoring  : {cfg['ablation'].get('blended_anchoring', True)}")
    print(f"  shallow_injection  : {cfg['ablation'].get('shallow_injection', True)}")
    print(f"  temporal_anneal    : {cfg['ablation'].get('temporal_anneal', False)}")
    print(f"  injection_scale    : {cfg['injection']['injection_scale']}")
    print(f"  denoising_strength : {cfg['stage2'].get('denoising_strength', 0.75)}")
    print(f"  compositing        : {cfg['ablation'].get('compositing', 'freq')}")
    print(f"  inversion_steps    : {stage1_meta.get('num_inversion_steps', '?')}")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_stage2(cfg: dict):
    """
    Execute the full Stage 2 pipeline.

    1. Load artifacts (content, donor, mask, kv_store)
    2. Load SD Inpainting model + patch shallow U-Net layers
    3. Run denoising loop with per-step PnP KV injection
    4. Decode latent → PIL image
    5. Compositing cleanup
    6. Save outputs
    """

    # ── Step 1: Load artifacts ────────────────────────────────────────────
    _section("Step 1 — Load artifacts")
    artifacts   = load_artifacts(cfg["paths"]["artifacts_dir"])
    stage1_meta = artifacts["meta"]

    # ── Step 2: Load model + patch UNet ──────────────────────────────────
    _section("Step 2 — Load SD Inpainting + patch shallow UNet layers")
    pipe = load_pipeline(cfg)

    use_shallow_inject = cfg["ablation"].get("shallow_injection", True)
    if use_shallow_inject:
        pipe.unet, depth_map = patch_unet_shallow_only(pipe.unet)
        n_shallow = sum(1 for v in depth_map.values() if v == "shallow")
        print(
            f"[stage2] UNet shallow-patched | "
            f"shallow KVI layers={n_shallow} | deep layers=standard attention"
        )
    else:
        pipe.unet, depth_map = patch_unet_attention(pipe.unet)
        print(
            "[stage2] shallow_injection=False — no KVI patching "
            "(pure inpainting baseline)"
        )

    # ── Step 3: Denoising loop ────────────────────────────────────────────
    _section("Step 3 — Denoising loop (PnP per-step KV injection)")
    latents = run_denoising_loop(pipe, artifacts, cfg, depth_map)

    # ── Step 4: Decode ────────────────────────────────────────────────────
    _section("Step 4 — Decode latent → image")
    raw_pil = decode_latent(pipe, latents)
    print(f"[stage2] Decoded: {raw_pil.size} {raw_pil.mode}")

    # ── Step 5: Compositing ───────────────────────────────────────────────
    _section("Step 5 — Boundary cleanup / compositing")
    result_pil = run_compositing(
        raw_pil,
        artifacts["content_pil"],
        artifacts["face_mask"],
        cfg,
    )

    # ── Step 6: Save ──────────────────────────────────────────────────────
    _section("Step 6 — Save outputs")
    stage2_meta = build_stage2_meta(cfg, stage1_meta)
    result_path = save_results(result_pil, artifacts, cfg, stage2_meta, raw_pil=raw_pil)

    return result_pil, result_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stage 2 — PnP KV Inpainting with Blended Latent Anchoring. "
            "Prior construction removed; texture via per-step KV injection."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help="Path to YAML config file")

    # ── Directory overrides ────────────────────────────────────────────────
    p.add_argument("--artifacts",  default=None,
                   help="Override paths.artifacts_dir")
    p.add_argument("--output-dir", default=None,
                   help="Override paths.output_dir")

    # ── Ablation overrides ─────────────────────────────────────────────────
    p.add_argument("--injection-scale",       type=float, default=None,
                   help="Override injection.injection_scale  (A6 sweep)")
    p.add_argument("--anneal",                action="store_true",
                   help="Enable temporal annealing — ramp λ_HF  (A4 ablation, off by default)")
    p.add_argument("--no-blended-anchoring",  action="store_true",
                   help="Disable blended latent anchoring  (A2 ablation)")
    p.add_argument("--no-shallow-injection",  action="store_true",
                   help="Disable shallow KV injection — pure inpainting baseline")

    # ── Diffusion overrides ────────────────────────────────────────────────
    p.add_argument("--steps",    type=int,   default=None,
                   help="Override stage2.num_inference_steps")
    p.add_argument("--guidance", type=float, default=None,
                   help="Override stage2.guidance_scale")
    p.add_argument("--seed",     type=int,   default=None,
                   help="Override stage2.seed")
    p.add_argument("--prompt",   default=None,
                   help="Override stage2.prompt")
    p.add_argument("--model-id", default=None,
                   help="Override stage2.model_id")

    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config and exit without running")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"\n[stage2] FATAL — {e}")
        sys.exit(1)

    cfg = apply_cli_overrides(cfg, args)

    artifacts_dir = cfg["paths"]["artifacts_dir"]
    meta_path     = os.path.join(artifacts_dir, "meta.json")
    stage1_meta   = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            stage1_meta = json.load(f)
    else:
        print(
            f"[stage2] WARNING: meta.json not found at {meta_path}. "
            f"Run stage1_segment.py and stage1b_invert.py first."
        )

    print_config(cfg, stage1_meta)

    if args.dry_run:
        print("[stage2] --dry-run: exiting without running.\n")
        sys.exit(0)

    try:
        result_pil, result_path = run_stage2(cfg)
    except FileNotFoundError as e:
        print(f"\n[stage2] FATAL — {e}\n")
        sys.exit(1)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(
                f"\n[stage2] FATAL — CUDA OOM: {e}\n"
                f"  Options:\n"
                f"  1. Set stage2.enable_cpu_offload: true in default.yaml\n"
                f"  2. Lower stage2.num_inference_steps\n"
                f"  3. Reduce image.target_size\n"
            )
        else:
            print(f"\n[stage2] FATAL — RuntimeError: {e}\n")
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[stage2] FATAL — unexpected error: {e}\n")
        traceback.print_exc()
        sys.exit(1)

    _print_final_summary(result_path, cfg, stage1_meta)
    sys.exit(0)


if __name__ == "__main__":
    main()