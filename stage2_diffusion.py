# stage2_diffusion.py
"""
Stage 2 — Mixed-Frequency Prior Guided Inpainting with Shallow KV Reinforcement.

Reads everything from artifacts/ (written by stage1_segment.py).
Loads Stable Diffusion Inpainting, patches shallow U-Net layers only,
runs blended-latent-anchored denoising with shallow KV injection,
saves the result.

Never imports SAM, MediaPipe, or anything from Stage 1.
All it needs from disk:
    artifacts/content_pil.png      — content image S (structure to keep)
    artifacts/donor_aligned_pil.png— aligned donor R̃ (texture source)
    artifacts/prior_pil.png        — mixed-frequency prior P = αS_LF + βR̃_LF + γR̃_HF
    artifacts/masked_input_pil.png — X₀ = (1-M)⊙S + M⊙P  (what diffusion sees)
    artifacts/face_mask.pt         — (1,1,S,S) float32 mask tensor M
    artifacts/meta.json            — ablation flags + stats written by Stage 1

Denoising loop (per timestep t):
    ┌─ Blended latent anchoring (pre-step) ────────────────────────────────────┐
    │  z_S_noised = add_noise(z_S, noise, t)                                   │
    │  latents    = Mz ⊙ latents + (1−Mz) ⊙ z_S_noised                       │
    │  → outside-mask region is reset to noisy source every step               │
    └──────────────────────────────────────────────────────────────────────────┘
    ┌─ Inject pass (shallow layers only) ──────────────────────────────────────┐
    │  noise_pred = unet(latents, t) with donor shallow KV blended in          │
    │  shallow layers ← donor_hf_cache × λ_HF(t)  [ramps 0→injection_scale]   │
    │  deep layers   ← standard attention, untouched                           │
    └──────────────────────────────────────────────────────────────────────────┘
    ┌─ Scheduler step ──────────────────────────────────────────────────────────┐
    │  latents = scheduler.step(noise_pred, t, latents).prev_sample            │
    └──────────────────────────────────────────────────────────────────────────┘
    ┌─ Blended latent anchoring (post-step) ───────────────────────────────────┐
    │  z_S_noised_next = add_noise(z_S, noise, t_next)                         │
    │  latents         = Mz ⊙ latents + (1−Mz) ⊙ z_S_noised_next             │
    │  → prevents scheduler drift outside the mask                             │
    └──────────────────────────────────────────────────────────────────────────┘

Shallow KV store (ONE pass, before loop):
    donor_aligned_pil → VAE encode → shallow U-Net forward → cache filled.
    This is done once, not per-step. The donor's fine surface structure is
    captured at 64×64 spatial resolution and injected throughout denoising.

Ablation flags (configs/default.yaml → ablation.*):
    blended_anchoring   : true  → per-step latent replacement (proposed)
                          false → one-time img2img init only (baseline)
    shallow_injection   : true  → shallow KV donor reinforcement (proposed)
                          false → pure inpainting, no KV (baseline)
    prior_construction  : "full"    → P = αS_LF + βR̃_LF + γR̃_HF (proposed)
                          "chimera" → P = S_LF + R̃_HF  (old chimera, α=1,β=0,γ=1)
                          "none"    → P = S (no prior, plain inpainting)
    histogram_match     : true  → histogram normalise P to source register
                          false → raw frequency sum
    temporal_anneal     : true  → λ_HF ramps 0 → injection_scale over steps
                          false → flat λ_HF = injection_scale throughout

Usage:
    # Default (reads configs/default.yaml):
    python stage2_diffusion.py

    # CLI overrides:
    python stage2_diffusion.py --injection-scale 0.5
    python stage2_diffusion.py --no-anneal
    python stage2_diffusion.py --no-blended-anchoring
    python stage2_diffusion.py --no-shallow-injection
    python stage2_diffusion.py --steps 50 --guidance 9.0 --seed 123
    python stage2_diffusion.py --artifacts artifacts/ --output-dir outputs/
    python stage2_diffusion.py --dry-run

Output:
    outputs/result.png           — final output image
    outputs/comparison.png       — 5-panel: Content | Donor | Prior | Masked | Result
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
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")
if os.path.isdir(_CORE):
    sys.path.insert(0, _HERE)

try:
    from core.kv_cache   import KVCache
    from core.patch_unet import patch_unet_attention, patch_unet_shallow_only
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
    """
    Apply CLI overrides on top of the yaml config.
    Only keys explicitly passed on the command line are overridden.
    """
    if args.artifacts is not None:
        cfg["paths"]["artifacts_dir"] = args.artifacts
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = args.output_dir
    if args.injection_scale is not None:
        cfg["injection"]["injection_scale"] = args.injection_scale
    if args.no_anneal:
        cfg["ablation"]["temporal_anneal"] = False
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
    print("\n┌─ Stage 2 config ──────────────────────────────────────────")
    print(f"│  model_id          : {cfg['stage2']['model_id']}")
    print(f"│  target_size       : {cfg['image']['target_size']}px")
    print(f"│  steps             : {cfg['stage2']['num_inference_steps']}")
    print(f"│  guidance          : {cfg['stage2']['guidance_scale']}")
    print(f"│  seed              : {cfg['stage2']['seed']}")
    print(f"│  prompt            : {cfg['stage2']['prompt'][:60]}...")
    print(f"│")
    print(f"│  [ablation]")
    print(f"│  prior_construction: {cfg['ablation'].get('prior_construction', 'full')}")
    print(f"│  blended_anchoring : {cfg['ablation'].get('blended_anchoring', True)}")
    print(f"│  shallow_injection : {cfg['ablation'].get('shallow_injection', True)}")
    print(f"│  histogram_match   : {cfg['ablation'].get('histogram_match', True)}")
    print(f"│  temporal_anneal   : {cfg['ablation'].get('temporal_anneal', True)}")
    print(f"│  injection_scale   : {cfg['injection']['injection_scale']}")
    print(f"│")
    print(f"│  [prior coefficients from stage1]")
    print(f"│  alpha (S_LF)      : {meta.get('alpha', '?')}")
    print(f"│  beta  (R_LF)      : {meta.get('beta', '?')}")
    print(f"│  gamma (R_HF)      : {meta.get('gamma', '?')}")
    print(f"│  alignment_method  : {meta.get('alignment_method', '?')}")
    print(f"│  mask_coverage     : {meta.get('mask_coverage_pct', '?')}%")
    print(f"└───────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_artifacts(artifacts_dir: str) -> dict:
    """
    Load all Stage 1 artifacts from disk.

    Required files (must all be present):
        content_pil.png       — content image S
        donor_aligned_pil.png — aligned donor R̃
        prior_pil.png         — mixed-frequency prior P
        masked_input_pil.png  — X₀ = (1-M)⊙S + M⊙P
        face_mask.pt          — (1,1,S,S) float32 mask tensor M
        meta.json             — stage1 config + stats

    Returns:
        dict with PIL images, mask tensor, and meta dict.

    Raises FileNotFoundError with a clear message if any artifact is missing.
    The most common cause is running Stage 2 before Stage 1 completes.
    """
    required = {
        "content_pil.png"       : "PIL",
        "donor_aligned_pil.png" : "PIL",
        "prior_pil.png"         : "PIL",
        "masked_input_pil.png"  : "PIL",
        "face_mask.pt"          : "tensor",
        "meta.json"             : "json",
    }

    missing = [
        fname for fname in required
        if not os.path.exists(os.path.join(artifacts_dir, fname))
    ]
    if missing:
        raise FileNotFoundError(
            f"[stage2] Missing artifacts in '{artifacts_dir}': {missing}\n"
            f"  Run stage1_segment.py first to generate these files.\n"
            f"  If you used a custom artifacts_dir in Stage 1, pass it here:\n"
            f"    python stage2_diffusion.py --artifacts <your_artifacts_dir>"
        )

    def _path(fname):
        return os.path.join(artifacts_dir, fname)

    content_pil       = Image.open(_path("content_pil.png")).convert("RGB")
    donor_aligned_pil = Image.open(_path("donor_aligned_pil.png")).convert("RGB")
    prior_pil         = Image.open(_path("prior_pil.png")).convert("RGB")
    masked_input_pil  = Image.open(_path("masked_input_pil.png")).convert("RGB")

    # map_location="cpu" so it loads regardless of GPU state
    face_mask = torch.load(_path("face_mask.pt"), map_location="cpu")

    with open(_path("meta.json"), "r") as f:
        meta = json.load(f)

    print(f"[stage2] Artifacts loaded from '{os.path.abspath(artifacts_dir)}'")
    print(
        f"[stage2]   content={content_pil.size} | donor={donor_aligned_pil.size} | "
        f"prior={prior_pil.size} | mask={tuple(face_mask.shape)}"
    )

    return {
        "content_pil"       : content_pil,
        "donor_aligned_pil" : donor_aligned_pil,
        "prior_pil"         : prior_pil,
        "masked_input_pil"  : masked_input_pil,
        "face_mask"         : face_mask,
        "meta"              : meta,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline(cfg: dict):
    """
    Load Stable Diffusion Inpainting pipeline and set up the scheduler.

    Key change from the old pipeline:
        Old: StableDiffusionPipeline (img2img)
        New: StableDiffusionInpaintPipeline (inpainting)

    The inpainting checkpoint (sd-2-inpainting) is specifically trained to
    complete masked regions conditioned on surrounding pixel context. This
    is exactly the task we are now solving — the masked_input X₀ gives the
    model realistic surrounding context + a mixed-frequency hint inside the
    hole.

    Returns:
        pipe : StableDiffusionInpaintPipeline on device with:
               - unet.eval()
               - DDIM scheduler (or as configured)
               - safety_checker disabled
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
            "This will be very slow (~30 min per run). "
            "Using float32 on CPU."
        )
        dtype = torch.float32

    print(f"[stage2] Loading {model_id} ({dtype_str}) as InpaintPipeline ...")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype   = dtype,
        safety_checker= None,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────
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
        print(
            f"[stage2] Unknown or unavailable scheduler '{sched_name}', "
            f"falling back to DDIM."
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # ── CPU offload (low VRAM mode) ────────────────────────────────────────
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
def _encode_to_latent(
    vae,
    image_processor,
    pil_image: Image.Image,
) -> torch.Tensor:
    """
    Encode a PIL RGB image to a VAE latent tensor.

    Uses pipe.image_processor so normalisation ([-1,1] range) matches
    exactly what the pipeline uses internally.

    Args:
        vae             : pipe.vae
        image_processor : pipe.image_processor
        pil_image       : PIL RGB image at the target resolution

    Returns:
        (1, 4, H/8, W/8) latent tensor, scaled by vae.config.scaling_factor
    """
    device    = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    pixel     = image_processor.preprocess(pil_image).to(device, dtype=vae_dtype)
    latent    = vae.encode(pixel).latent_dist.sample()
    return latent * vae.config.scaling_factor


@torch.no_grad()
def _encode_source_latent(
    vae,
    image_processor,
    content_pil: Image.Image,
) -> torch.Tensor:
    """
    Encode the content image S to a latent for blended latent anchoring.

    This is a dedicated function (separate from _encode_to_latent) to make
    the blended anchoring usage explicit and easy to trace. The returned
    latent z_S is used at every denoising step to overwrite the non-mask
    region of the evolving latent.

    Why a separate encode pass?
        The masked_input X₀ latent (z_X0) is the denoising starting point.
        The source latent z_S is the anchoring reference — it must encode S
        without the prior pixels, so outside-mask regions stay pixel-identical
        to S after blended latent replacement.

    Args:
        vae             : pipe.vae
        image_processor : pipe.image_processor
        content_pil     : Content image S (PIL RGB, target resolution)

    Returns:
        (1, 4, H/8, W/8) latent tensor for S
    """
    return _encode_to_latent(vae, image_processor, content_pil)


# ══════════════════════════════════════════════════════════════════════════════
# MASK DOWNSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def _downsample_mask(face_mask: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    """
    Downsample the pixel-space mask to latent spatial resolution.

    The VAE compresses by 8× in each dimension. For a 512×512 image,
    latent spatial size is 64×64. The mask must match this for the
    element-wise blended latent operation.

    Args:
        face_mask : (1, 1, H, W) float32 mask, values in [0, 1]
        latent_h  : latent height (H/8)
        latent_w  : latent width  (W/8)

    Returns:
        (1, 1, latent_h, latent_w) float32 mask in [0, 1]
        Threshold at 0.5 so boundary pixels are cleanly binary.
    """
    m = F.interpolate(
        face_mask,
        size=(latent_h, latent_w),
        mode="bilinear",
        align_corners=False,
    )
    return (m > 0.5).float()


# ══════════════════════════════════════════════════════════════════════════════
# BLENDED LATENT ANCHORING
# ══════════════════════════════════════════════════════════════════════════════

def _blend_latents(
    latents:      torch.Tensor,   # (1, 4, H/8, W/8) current denoising latent
    z_S_noised:   torch.Tensor,   # (1, 4, H/8, W/8) noised source latent at t
    Mz:           torch.Tensor,   # (1, 1, H/8, W/8) latent-space mask {0, 1}
) -> torch.Tensor:
    """
    Blended latent anchoring — enforce source structure outside the mask.

    This is the single most important operation in the pipeline.
    At every denoising step, the non-mask region of the latent is
    overwritten with the noised source latent z_S_noised.

    Effect:
        Inside mask  (Mz = 1): latent is unchanged — diffusion fills freely
        Outside mask (Mz = 0): latent ← z_S_noised — source is frozen

    This guarantees that the content image is preserved exactly outside
    the editable region by construction, not by relying on the model to
    respect KV constraints globally.

    Applied TWICE per step:
        1. Before the inject pass  — so the inject pass sees a correctly
           anchored starting point
        2. After scheduler.step()  — to correct any drift the scheduler
           step itself might introduce outside the mask

    Args:
        latents    : Current denoising latent (mask region evolving freely)
        z_S_noised : Noised source latent at the current or next timestep
        Mz         : Binary latent-space mask (1=editable, 0=frozen)

    Returns:
        Blended latent tensor — inside mask unchanged, outside replaced with z_S_noised
    """
    return Mz * latents + (1.0 - Mz) * z_S_noised


# ══════════════════════════════════════════════════════════════════════════════
# SHALLOW KV STORE PASS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _shallow_store_pass(
    unet,
    donor_lat:   torch.Tensor,   # (1, 4, H/8, W/8) donor aligned latent
    ref_embeds:  torch.Tensor,   # uncond text embeds (no CFG needed here)
    kv_cache:    KVCache,
    timestep:    torch.Tensor,   # representative mid-noise timestep
):
    """
    Single U-Net forward pass to populate shallow KV cache with donor features.

    This is done ONCE before the denoising loop starts, not per-step.
    The stored K,V tensors represent the donor's fine surface structure at
    64×64 spatial resolution (down_blocks.0 and up_blocks.3). These will
    be injected into every denoising step via shallow-layer KV blending.

    Why once (not per-step)?
        Per-step store passes (the old design) were needed to track the
        evolving noise level of the reference. In the new design, the
        donor's texture information is static — we just need the shallow
        attention features that encode R̃'s surface detail. A single
        forward at a representative mid-noise timestep gives stable,
        clean features without the cost of 30 additional U-Net passes.

    Why only shallow layers?
        Deep layers (mid_block, down_blocks.2, up_blocks.1) at 8×8–16×16
        resolution encode coarse structure. Structure preservation is now
        handled by blended latent anchoring — injecting deep layers would
        conflict with that mechanism and potentially push geometry toward
        the donor rather than the content. Shallow layers encode fine
        surface detail (texture, microstructure) — exactly the donor
        attribute we want to transfer.

    The cache mode is set to "store" with freq_mode="hf" so that only
    KVInjectionAttention layers classified as "shallow" write to _hf_cache.
    Deep-layer KVInjectionAttention instances are absent (patch_unet_shallow_only
    leaves them with _KwargSafeProcessor), so they contribute nothing.

    Args:
        unet       : Shallow-patched U-Net (only down_blocks.0 + up_blocks.3 are KVI)
        donor_lat  : VAE-encoded aligned donor R̃ latent
        ref_embeds : Uncond text embeddings (single-batch, no CFG)
        kv_cache   : KVCache in which to write shallow HF features
        timestep   : Representative timestep (mid-noise, e.g. 500/999 × max_t)
    """
    kv_cache.set_freq_mode("hf")
    kv_cache.set_mode("store")

    t = timestep.unsqueeze(0) if timestep.ndim == 0 else timestep
    unet(
        donor_lat,
        t,
        encoder_hidden_states  = ref_embeds,
        cross_attention_kwargs = {"kv_cache": kv_cache},
    )
    # Output discarded — cache population is the only side effect.
    # The kv_cache._hf_cache now holds shallow K,V from the donor.


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DENOISING LOOP
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_denoising_loop(
    pipe,
    artifacts:  dict,
    cfg:        dict,
) -> torch.Tensor:
    """
    Inpainting denoising loop with blended latent anchoring and shallow KV injection.

    Architecture:
        - No dual store passes. Donor KV is captured ONCE before the loop.
        - Blended latent anchoring at every step (pre + post scheduler).
        - Shallow layers inject donor surface texture via λ_HF (ramps up).
        - Deep layers are untouched — structure preserved by anchoring.
        - CFG guidance applied normally.

    Ablation conditions (all controlled via default.yaml → ablation.*):
        blended_anchoring=false : one-time img2img init only (old baseline)
        shallow_injection=false : pure inpainting, no donor KV
        temporal_anneal=false   : flat λ_HF = injection_scale throughout
        prior_construction=none : P = S (plain inpainting baseline)

    Args:
        pipe      : Loaded + patched StableDiffusionInpaintPipeline
        artifacts : dict from load_artifacts()
        cfg       : Resolved config dict

    Returns:
        latents : (1, 4, H/8, W/8) final denoised latent tensor (pre-decode)
    """
    device = next(pipe.unet.parameters()).device

    # ── Config ────────────────────────────────────────────────────────────
    anneal           = cfg["ablation"].get("temporal_anneal",   True)
    blended_anchor   = cfg["ablation"].get("blended_anchoring", True)
    shallow_inject   = cfg["ablation"].get("shallow_injection", True)
    scale            = cfg["injection"]["injection_scale"]
    steps            = cfg["stage2"]["num_inference_steps"]
    g_scale          = cfg["stage2"]["guidance_scale"]
    seed             = cfg["stage2"]["seed"]
    prompt           = cfg["stage2"]["prompt"]
    neg              = cfg["stage2"].get("negative_prompt", "")
    verbose          = cfg["logging"].get("verbose", True)

    do_cfg = g_scale > 1.0

    # ── KV cache setup ────────────────────────────────────────────────────
    # depth_routing is not used for shallow-only injection but kept in the
    # cache for interface compatibility with KVInjectionAttention.
    kv_cache = KVCache()
    kv_cache.depth_routing = "correct"    # shallow layers always read "hf"
    kv_cache.face_mask     = artifacts["face_mask"].to(device)

    # ── Encode images to latents ──────────────────────────────────────────
    _section("Encoding images to latents")

    # z_X0 — masked prior latent: initialisation for denoising
    z_X0 = _encode_to_latent(
        pipe.vae, pipe.image_processor, artifacts["masked_input_pil"]
    )

    # z_S — clean source latent: blended latent anchoring reference
    z_S = _encode_source_latent(
        pipe.vae, pipe.image_processor, artifacts["content_pil"]
    )

    # donor_lat — aligned donor latent: for shallow KV store pass
    donor_lat = _encode_to_latent(
        pipe.vae, pipe.image_processor, artifacts["donor_aligned_pil"]
    )

    latent_h, latent_w = z_X0.shape[-2], z_X0.shape[-1]

    print(
        f"[stage2] Encoded: z_X0={tuple(z_X0.shape)} | "
        f"z_S={tuple(z_S.shape)} | donor={tuple(donor_lat.shape)}"
    )

    # ── Latent-space mask ─────────────────────────────────────────────────
    # Mz: (1, 1, H/8, W/8) float32 {0, 1}
    # 1 = editable region (inside mask — diffusion fills this)
    # 0 = frozen region  (outside mask — blended anchoring enforces S)
    Mz = _downsample_mask(artifacts["face_mask"], latent_h, latent_w).to(device)

    # ── Prompt embeddings ─────────────────────────────────────────────────
    _section("Encoding prompts")

    prompt_embeds, neg_embeds = pipe.encode_prompt(
        prompt                     = prompt,
        device                     = device,
        num_images_per_prompt      = 1,
        do_classifier_free_guidance= do_cfg,
        negative_prompt            = neg,
    )

    # CFG: concat [uncond, cond] for a single batched forward pass
    text_embeds = torch.cat([neg_embeds, prompt_embeds]) if do_cfg \
                  else prompt_embeds

    # Store and KV passes always use uncond — halves VRAM cost
    ref_embeds = neg_embeds if do_cfg else prompt_embeds

    print(f"[stage2] Prompt embeds: {tuple(prompt_embeds.shape)} | CFG={do_cfg}")

    # ── Scheduler timesteps ───────────────────────────────────────────────
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps   # (T,) descending, e.g. 999→0
    T         = len(timesteps)

    # ── Generator for reproducibility ────────────────────────────────────
    gen   = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z_X0.shape, generator=gen, device=device, dtype=z_X0.dtype)

    # ── Initial noisy latent ──────────────────────────────────────────────
    # Start from a noisy version of X₀ (the masked prior image).
    # Inside the mask: prior P is the starting signal.
    # Outside the mask: S pixels are present but will be overwritten
    #   at every step anyway by blended latent anchoring.
    latents = pipe.scheduler.add_noise(z_X0, noise, timesteps[:1])

    print(f"[stage2] Starting denoising | steps={T} | seed={seed}")
    print(f"[stage2] blended_anchoring={blended_anchor} | "
          f"shallow_injection={shallow_inject} | "
          f"temporal_anneal={anneal} | injection_scale={scale}")

    # ── Shallow KV store pass (ONCE, before loop) ─────────────────────────
    # Capture donor fine surface features at 64×64 spatial resolution.
    # Only runs when shallow_injection is enabled.
    if shallow_inject:
        _section("Shallow KV store pass (donor surface features — once before loop)")
        # Use a representative mid-noise timestep for stable feature capture
        mid_idx = T // 2
        mid_t   = timesteps[mid_idx]
        _shallow_store_pass(pipe.unet, donor_lat, ref_embeds, kv_cache, mid_t)
        # Cache is now populated. Mode is reset to bypass before the loop.
        kv_cache.set_mode("bypass")
        print(
            f"[stage2] Shallow KV stored at t={int(mid_t)} | "
            f"HF keys={len(kv_cache._hf_cache)}"
        )
    else:
        print("[stage2] shallow_injection=False — skipping KV store (pure inpainting).")

    _section(f"Denoising loop  [T={T} steps]")

    # ── Denoising loop ─────────────────────────────────────────────────────
    for step_idx, t in enumerate(timesteps):

        # ── Lambda update ─────────────────────────────────────────────────
        # Only λ_HF is used — deep layer injection is gone.
        # λ_HF ramps from 0 → injection_scale over the denoising trajectory
        # so texture detail is applied late (when diffusion is filling fine
        # detail), not early (when it is deciding coarse structure).
        progress    = step_idx / max(T - 1, 1)
        lambda_hf   = (progress * scale) if anneal else scale
        lambda_hf   = lambda_hf if shallow_inject else 0.0

        # Update KVCache lambdas — only hf matters; lf zeroed out
        kv_cache.lambda_lf = 0.0
        kv_cache.lambda_hf = lambda_hf

        if verbose and (step_idx % 5 == 0 or step_idx == T - 1):
            print(
                f"  step {step_idx + 1:3d}/{T} | t={int(t):4d} | "
                f"λ_HF={lambda_hf:.3f} | "
                f"anchor={blended_anchor} | inject={shallow_inject}"
            )

        # ── Blended latent anchoring (pre-step) ───────────────────────────
        # Overwrite the non-mask region with the noised source latent.
        # This is the core structure preservation mechanism.
        # Must happen BEFORE the inject pass so the U-Net never sees
        # source-drifted content in the background region.
        if blended_anchor:
            z_S_noised = pipe.scheduler.add_noise(z_S, noise, t.unsqueeze(0))
            latents    = _blend_latents(latents, z_S_noised, Mz)

        # ── Inject pass ───────────────────────────────────────────────────
        # Shallow layers will blend donor K,V into the current attention.
        # Deep layers run standard attention (untouched by patching).
        # CFG doubles the batch: [uncond_latent, cond_latent]
        if shallow_inject:
            kv_cache.set_mode("inject")
        else:
            kv_cache.set_mode("bypass")

        src_input  = torch.cat([latents] * 2) if do_cfg else latents
        noise_pred = pipe.unet(
            src_input,
            t.unsqueeze(0) if t.ndim == 0 else t,
            encoder_hidden_states  = text_embeds if do_cfg else prompt_embeds,
            cross_attention_kwargs = {"kv_cache": kv_cache},
        ).sample

        # CFG: combine uncond and cond noise predictions
        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + g_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        # ── Scheduler step ────────────────────────────────────────────────
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # ── Blended latent anchoring (post-step) ──────────────────────────
        # Correct any drift outside the mask that the scheduler step
        # itself may have introduced. Uses the noise level of the NEXT
        # timestep (t_next) so the replacement matches where the denoiser
        # will operate on the next iteration.
        if blended_anchor:
            if step_idx < T - 1:
                t_next       = timesteps[step_idx + 1]
                z_S_noised_n = pipe.scheduler.add_noise(
                    z_S, noise, t_next.unsqueeze(0)
                )
            else:
                # Final step: use clean z_S (no noise)
                z_S_noised_n = z_S
            latents = _blend_latents(latents, z_S_noised_n, Mz)

        # Free KV VRAM after each step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[stage2] Denoising complete. Final latent: {tuple(latents.shape)}")
    return latents


# ══════════════════════════════════════════════════════════════════════════════
# DECODE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def decode_latent(pipe, latents: torch.Tensor) -> Image.Image:
    """
    Decode a latent tensor to a PIL RGB image.

    Divides by scaling_factor before decoding (inverse of encode).
    pipe.image_processor.postprocess() handles [-1,1] → [0,255] and uint8 cast.
    """
    latents = latents / pipe.vae.config.scaling_factor
    image   = pipe.vae.decode(latents).sample
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    result_pil:  Image.Image,
    artifacts:   dict,
    cfg:         dict,
    stage2_meta: dict,
) -> str:
    """
    Save all outputs to the output directory.

    Files written:
        result.png          — final output image
        comparison.png      — 5-panel: Content | Donor | Prior | Masked Input | Result
        meta_stage2.json    — complete Stage 2 config for reproducibility

    The comparison panel now shows the full pipeline: content structure source,
    donor texture source, mixed-frequency prior, the masked input X₀ that
    diffusion sees, and the final result. This makes every design decision
    visually auditable in one image.
    """
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ── Result image ───────────────────────────────────────────────────────
    result_path = os.path.join(output_dir, "result.png")
    result_pil.save(result_path)

    # ── 5-panel comparison ─────────────────────────────────────────────────
    panels = [
        artifacts["content_pil"],
        artifacts["donor_aligned_pil"],
        artifacts["prior_pil"],
        artifacts["masked_input_pil"],
        result_pil,
    ]
    labels = [
        "Content (S)",
        "Donor aligned (R̃)",
        "Prior (αS_LF + βR̃_LF + γR̃_HF)",
        "Masked input (X₀)",
        "Result",
    ]

    W, H      = panels[0].size
    pad       = 4
    label_h   = 22
    total_w   = W * len(panels) + pad * (len(panels) + 1)
    total_h   = H + label_h + pad * 2
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

    comp_path = os.path.join(output_dir, "comparison.png")
    comparison.save(comp_path)

    # ── Stage 2 meta ───────────────────────────────────────────────────────
    meta_path = os.path.join(output_dir, "meta_stage2.json")
    with open(meta_path, "w") as f:
        json.dump(stage2_meta, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n[stage2] Outputs saved → {os.path.abspath(output_dir)}/")
    for fname in ["result.png", "comparison.png", "meta_stage2.json"]:
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
    Build the meta_stage2.json dict for full reproducibility.

    Contains every Stage 2 parameter used in this run plus Stage 1 provenance.
    The ablation flags are recorded verbatim so any run can be replicated
    from this JSON alone (combined with the default.yaml snapshot).

    Key additions vs the old meta builder:
        blended_anchoring, shallow_only, alpha, beta, gamma — the new
        pipeline's design choices that are not present in the face-swap version.
    """
    ablation_cfg = cfg.get("ablation", {})
    inj_cfg      = cfg.get("injection", {})
    prior_cfg    = cfg.get("prior", {})

    return {
        # ── Stage 2 params ────────────────────────────────────────────────
        "model_id"           : cfg["stage2"]["model_id"],
        "pipeline_type"      : "inpainting",
        "scheduler"          : cfg["stage2"].get("scheduler", "DDIM"),
        "num_inference_steps": cfg["stage2"]["num_inference_steps"],
        "guidance_scale"     : cfg["stage2"]["guidance_scale"],
        "seed"               : cfg["stage2"]["seed"],
        "prompt"             : cfg["stage2"]["prompt"],
        "negative_prompt"    : cfg["stage2"].get("negative_prompt", ""),
        "torch_dtype"        : cfg["stage2"]["torch_dtype"],

        # ── Injection params ──────────────────────────────────────────────
        "injection_scale"    : inj_cfg.get("injection_scale", 0.8),
        "shallow_only"       : True,   # always shallow-only in new pipeline

        # ── Ablation flags (all six) ──────────────────────────────────────
        "ablation": {
            "prior_construction" : ablation_cfg.get("prior_construction", "full"),
            "blended_anchoring"  : ablation_cfg.get("blended_anchoring",  True),
            "shallow_injection"  : ablation_cfg.get("shallow_injection",  True),
            "histogram_match"    : ablation_cfg.get("histogram_match",    True),
            "temporal_anneal"    : ablation_cfg.get("temporal_anneal",    True),
            "compositing"        : ablation_cfg.get("compositing",        "simple"),
        },

        # ── Prior construction coefficients (from Stage 1) ────────────────
        # These are read from stage1_meta so the meta is self-consistent
        # even if the user changes the yaml between stages.
        "prior": {
            "alpha" : stage1_meta.get("alpha",  prior_cfg.get("alpha",  0.6)),
            "beta"  : stage1_meta.get("beta",   prior_cfg.get("beta",   0.5)),
            "gamma" : stage1_meta.get("gamma",  prior_cfg.get("gamma",  0.8)),
        },

        # ── Stage 1 provenance ────────────────────────────────────────────
        "stage1_meta"        : stage1_meta,

        # ── Provenance ────────────────────────────────────────────────────
        "stage2_script"      : os.path.abspath(__file__),
        "timestamp"          : datetime.datetime.now().isoformat(),
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
    print(f"  Stage 2 complete — Mixed-Frequency Prior Guided Inpainting")
    print(f"{'═' * 60}")
    print(f"  result             : {os.path.abspath(result_path)}")
    print(f"  pipeline_type      : inpainting")
    print(f"  shallow_only       : True")
    print(f"  blended_anchoring  : {cfg['ablation'].get('blended_anchoring', True)}")
    print(f"  shallow_injection  : {cfg['ablation'].get('shallow_injection', True)}")
    print(f"  injection_scale    : {cfg['injection']['injection_scale']}")
    print(f"  temporal_anneal    : {cfg['ablation'].get('temporal_anneal', True)}")
    print(f"  alpha / beta / gamma: "
          f"{stage1_meta.get('alpha', '?')} / "
          f"{stage1_meta.get('beta', '?')} / "
          f"{stage1_meta.get('gamma', '?')}")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_stage2(cfg: dict):
    """
    Execute the full Stage 2 pipeline.

    1. Load artifacts from disk
    2. Load SD Inpainting model + patch shallow U-Net layers
    3. Run blended-latent-anchored denoising with shallow KV injection
    4. Decode latent → PIL image
    5. Save outputs

    Args:
        cfg : Fully resolved config dict (yaml + CLI overrides applied)

    Returns:
        result_pil  : PIL RGB output image
        result_path : Path where result was saved
    """

    # ── Step 1: Load artifacts ────────────────────────────────────────────
    _section("Step 1 — Load artifacts")
    artifacts   = load_artifacts(cfg["paths"]["artifacts_dir"])
    stage1_meta = artifacts["meta"]

    # ── Step 2: Load model + patch UNet ──────────────────────────────────
    _section("Step 2 — Load SD Inpainting + patch shallow UNet layers")
    pipe = load_pipeline(cfg)

    # Shallow-only patching: only down_blocks.0 and up_blocks.3 get
    # KVInjectionAttention. Deep layers get _KwargSafeProcessor only.
    # Structure preservation is handled by blended latent anchoring —
    # deep injection would conflict with that mechanism.
    use_shallow_inject = cfg["ablation"].get("shallow_injection", True)
    if use_shallow_inject:
        pipe.unet, depth_map = patch_unet_shallow_only(pipe.unet)
        n_shallow = sum(1 for v in depth_map.values() if v == "shallow")
        print(
            f"[stage2] UNet shallow-patched | "
            f"shallow KVI layers={n_shallow} | deep layers=standard attention"
        )
    else:
        # No injection — just apply safe cross-attention processors
        # to absorb kv_cache kwarg without warnings
        pipe.unet, depth_map = patch_unet_attention(pipe.unet)
        print(
            "[stage2] shallow_injection=False — no KVI patching "
            "(pure inpainting baseline)"
        )

    # ── Step 3: Denoising loop ────────────────────────────────────────────
    _section("Step 3 — Denoising loop")
    latents = run_denoising_loop(pipe, artifacts, cfg)

    # ── Step 4: Decode ────────────────────────────────────────────────────
    _section("Step 4 — Decode latent → image")
    result_pil = decode_latent(pipe, latents)
    print(f"[stage2] Decoded: {result_pil.size} {result_pil.mode}")

    # ── Step 5: Save ──────────────────────────────────────────────────────
    _section("Step 5 — Save outputs")
    stage2_meta = build_stage2_meta(cfg, stage1_meta)
    result_path = save_results(result_pil, artifacts, cfg, stage2_meta)

    return result_pil, result_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stage 2 — Mixed-Frequency Prior Guided Inpainting "
            "with Shallow KV Reinforcement"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file"
    )

    # ── Directory overrides ────────────────────────────────────────────────
    p.add_argument("--artifacts",  default=None,
                   help="Override paths.artifacts_dir")
    p.add_argument("--output-dir", default=None,
                   help="Override paths.output_dir")

    # ── Ablation overrides ─────────────────────────────────────────────────
    p.add_argument("--injection-scale", type=float, default=None,
                   help="Override injection.injection_scale  (A6 sweep)")
    p.add_argument("--no-anneal",       action="store_true",
                   help="Disable temporal annealing — flat λ_HF  (A4 ablation)")
    p.add_argument("--no-blended-anchoring", action="store_true",
                   help="Disable blended latent anchoring — one-time init only  (A2 ablation)")
    p.add_argument("--no-shallow-injection", action="store_true",
                   help="Disable shallow KV injection — pure inpainting baseline  (A3 ablation)")

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

    # ── Utility ────────────────────────────────────────────────────────────
    p.add_argument("--dry-run",  action="store_true",
                   help="Print resolved config and exit without running")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"\n[stage2] FATAL — {e}")
        sys.exit(1)

    cfg = apply_cli_overrides(cfg, args)

    # ── Load Stage 1 meta for display (before dry-run exit) ───────────────
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    meta_path     = os.path.join(artifacts_dir, "meta.json")
    stage1_meta   = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            stage1_meta = json.load(f)
    else:
        print(
            f"[stage2] WARNING: meta.json not found at {meta_path}. "
            f"Run stage1_segment.py first."
        )

    print_config(cfg, stage1_meta)

    if args.dry_run:
        print("[stage2] --dry-run: exiting without running.\n")
        sys.exit(0)

    # ── Run pipeline ──────────────────────────────────────────────────────
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
                f"  2. Lower stage2.num_inference_steps (e.g. 20)\n"
                f"  3. Use a smaller model or reduce image target_size\n"
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