# stage2_diffusion.py
"""
Stage 2 — KV-injection diffusion pipeline.

Reads everything from artifacts/ (written by stage1_segment.py).
Loads Stable Diffusion, patches the U-Net, runs the 3-pass denoising loop,
saves the result.

Never imports SAM, MediaPipe, or anything from Stage 1.
All it needs from disk is:
    artifacts/source_pil.png     — source image (pose/background to keep)
    artifacts/aligned_pil.png    — aligned reference (phase2 mode only)
    artifacts/lf_pil.png         — LF component of aligned reference
    artifacts/hf_pil.png         — HF component (shifted to [0,255])
    artifacts/face_mask.pt       — (1,1,S,S) float32 mask tensor
    artifacts/meta.json          — ablation flags + stats written by Stage 1

3-pass denoising loop (per timestep):
    Pass 1 — store LF  : noisy LF ref → U-Net (deep layers store structure KV)
    Pass 2 — store HF  : noisy HF ref → U-Net (shallow layers store texture KV)
    Pass 3 — inject    : noisy source latent → U-Net with reference KV blended
                         deep layers   ← LF cache × λ_LF(t)
                         shallow layers← HF cache × λ_HF(t)

Phase 2 baseline (ablation.mode = "phase2"):
    Single store pass with aligned reference, no freq split.
    Both LF and HF caches receive identical KV.
    Flat lambdas (no temporal annealing).

Usage:
    # Default config:
    python stage2_diffusion.py

    # Override ablation flags without editing yaml:
    python stage2_diffusion.py --mode phase2
    python stage2_diffusion.py --depth-routing swapped
    python stage2_diffusion.py --injection-scale 0.5
    python stage2_diffusion.py --no-anneal
    python stage2_diffusion.py --steps 50 --guidance 9.0

    # Custom artifacts and output dirs:
    python stage2_diffusion.py --artifacts artifacts/ --output-dir outputs/

    # Dry run — prints resolved config and exits:
    python stage2_diffusion.py --dry-run

Output:
    outputs/result.png                   — final face swap result
    outputs/result_phase2.png            — if mode=phase2
    outputs/comparison.png               — side-by-side: source | aligned | result
    outputs/meta_stage2.json             — all stage2 params for reproducibility

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
import yaml
from PIL import Image
from core.compositing import composite_result

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")
if os.path.isdir(_CORE):
    sys.path.insert(0, _HERE)   # insert project root so 'core.X' imports work

try:
    from core.kv_cache   import KVCache
    from core.patch_unet import patch_unet_attention
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
        cfg["paths"]["artifacts_dir"] = args.artifacts_dir
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = args.output_dir
    if args.mode is not None:
        cfg["ablation"]["mode"] = args.mode
    if args.depth_routing is not None:
        cfg["ablation"]["depth_routing"] = args.depth_routing
    if args.injection_scale is not None:
        cfg["injection"]["injection_scale"] = args.injection_scale
    if args.no_anneal:
        cfg["ablation"]["temporal_anneal"] = False
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
    print(f"│  model_id        : {cfg['stage2']['model_id']}")
    print(f"│  target_size     : {cfg['image']['target_size']}px")
    print(f"│  steps           : {cfg['stage2']['num_inference_steps']}")
    print(f"│  guidance        : {cfg['stage2']['guidance_scale']}")
    print(f"│  seed            : {cfg['stage2']['seed']}")
    print(f"│  prompt          : {cfg['stage2']['prompt'][:60]}...")
    print(f"│")
    print(f"│  [ablation]")
    print(f"│  mode            : {cfg['ablation']['mode']}")
    print(f"│  depth_routing   : {cfg['ablation']['depth_routing']}")
    print(f"│  temporal_anneal : {cfg['ablation']['temporal_anneal']}")
    print(f"│  injection_scale : {cfg['injection']['injection_scale']}")
    print(f"│")
    print(f"│  [from stage1 meta.json]")
    print(f"│  decomp_method   : {meta.get('decomp_method', 'unknown')}")
    print(f"│  mask_type       : {meta.get('mask_type', 'unknown')}")
    print(f"│  yaw_diff        : {meta.get('yaw_diff_deg', '?'):.1f}°")
    print(f"│  HF_std          : {meta.get('hf_std', '?'):.2f}")
    print(f"│  mask_coverage   : {meta.get('mask_coverage_pct', '?'):.1f}%")
    print(f"└───────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_artifacts(artifacts_dir: str) -> dict:
    """
    Load all Stage 1 artifacts from disk.

    Returns a dict with keys:
        source_pil    : PIL RGB
        aligned_pil   : PIL RGB
        lf_pil        : PIL RGB
        hf_pil        : PIL RGB
        face_mask     : (1,1,S,S) float32 tensor (CPU)
        meta          : dict from meta.json

    Raises FileNotFoundError with a clear message if any artifact is missing.
    The most common cause is running Stage 2 before Stage 1 has completed.
    """
    required = {
        "source_pil.png"  : "PIL",
        "aligned_pil.png" : "PIL",
        "lf_pil.png"      : "PIL",
        "hf_pil.png"      : "PIL",
        "face_mask.pt"    : "tensor",
        "meta.json"       : "json",
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

    source_pil  = Image.open(_path("source_pil.png")).convert("RGB")
    aligned_pil = Image.open(_path("aligned_pil.png")).convert("RGB")
    lf_pil      = Image.open(_path("lf_pil.png")).convert("RGB")
    hf_pil      = Image.open(_path("hf_pil.png")).convert("RGB")

    # Load mask tensor — map_location="cpu" so it loads regardless of GPU state
    face_mask = torch.load(_path("face_mask.pt"), map_location="cpu")

    with open(_path("meta.json"), "r") as f:
        meta = json.load(f)

    print(f"[stage2] Artifacts loaded from '{os.path.abspath(artifacts_dir)}'")
    print(
        f"[stage2]   source={source_pil.size} | lf={lf_pil.size} | "
        f"hf={hf_pil.size} | mask={tuple(face_mask.shape)}"
    )

    return {
        "source_pil"  : source_pil,
        "aligned_pil" : aligned_pil,
        "lf_pil"      : lf_pil,
        "hf_pil"      : hf_pil,
        "face_mask"   : face_mask,
        "meta"        : meta,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline(cfg: dict):
    """
    Load Stable Diffusion pipeline and set up the scheduler.

    Returns:
        pipe : StableDiffusionPipeline on device, with:
               - unet.eval()
               - DDIM scheduler (or as configured)
               - safety_checker disabled (we're operating on face images,
                 not generating content; checker would block valid outputs)

    Memory note:
        SD 2.1-base in float16 uses ~3.5GB VRAM for the full pipeline.
        After Stage 1 freed SAM (~6GB), a T4 (15GB) has plenty of headroom.
        If VRAM is still tight (< 8GB), set stage2.enable_cpu_offload: true
        in default.yaml to use accelerate's CPU offloading.
    """
    from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler
    try:
        from diffusers import DPMSolverMultistepScheduler
        _has_dpm = True
    except ImportError:
        _has_dpm = False

    model_id   = cfg["stage2"]["model_id"]
    dtype_str  = cfg["stage2"]["torch_dtype"]
    dtype      = torch.float16 if dtype_str == "float16" else torch.float32
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print(
            "[stage2] WARNING: No GPU detected — running on CPU. "
            "This will be very slow (~30 min per run). "
            "Using float32 on CPU."
        )
        dtype = torch.float32

    print(f"[stage2] Loading {model_id} ({dtype_str}) ...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
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
# VAE ENCODING HELPER
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
    exactly what the pipeline uses internally. Returns a latent on the
    same device and dtype as the VAE.

    Args:
        vae             : pipe.vae
        image_processor : pipe.image_processor
        pil_image       : PIL RGB image at the target resolution

    Returns:
        (1, 4, H/8, W/8) latent tensor
    """
    device    = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    pixel = image_processor.preprocess(pil_image).to(device, dtype=vae_dtype)
    latent = vae.encode(pixel).latent_dist.sample()
    return latent * vae.config.scaling_factor


# ══════════════════════════════════════════════════════════════════════════════
# STORE PASS HELPER
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _store_pass(
    unet,
    latent:        torch.Tensor,   # (1, 4, H/8, W/8) noisy latent
    timestep:      torch.Tensor,   # scalar or (1,) tensor
    ref_embeds:    torch.Tensor,   # uncond text embeds (no CFG during store)
    kv_cache:      KVCache,
):
    """
    Single U-Net forward pass for KV population.

    Output is discarded — the only side effect is filling the KV cache.
    Always runs without CFG (single batch, uncond embeds) to halve the
    memory cost of the two store passes per step.

    The timestep is passed unsqueezed to match the shape U-Net expects:
        (1,) tensor for a single-sample batch.
    """
    t = timestep.unsqueeze(0) if timestep.ndim == 0 else timestep
    unet(
        latent,
        t,
        encoder_hidden_states=ref_embeds,
        cross_attention_kwargs={"kv_cache": kv_cache},
    )
    # Return value deliberately ignored — cache population is the side effect


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DENOISING LOOP
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_denoising_loop(
    pipe,
    artifacts:       dict,
    cfg:             dict,
) -> torch.Tensor:
    """
    Full denoising loop with 3-pass KV injection per timestep.

    Args:
        pipe      : Loaded + patched StableDiffusionPipeline
        artifacts : dict from load_artifacts()
        cfg       : Resolved config dict

    Returns:
        latents : (1, 4, H/8, W/8) final denoised latent tensor (pre-decode)

    Loop structure per timestep t:
        ┌─ Pass 1 (mode=store, freq=lf) ──────────────────────────────────────┐
        │  noisy_lf  = add_noise(lf_latent, noise, t)                         │
        │  unet(noisy_lf) → fills _lf_cache in deep layers                   │
        └──────────────────────────────────────────────────────────────────────┘
        ┌─ Pass 2 (mode=store, freq=hf) ──────────────────────────────────────┐
        │  noisy_hf  = add_noise(hf_latent, noise, t)                         │
        │  unet(noisy_hf) → fills _hf_cache in shallow layers                │
        └──────────────────────────────────────────────────────────────────────┘
        ┌─ Pass 3 (mode=inject) ──────────────────────────────────────────────┐
        │  src_input = [latents, latents] for CFG                             │
        │  noise_pred = unet(src_input) with LF/HF KV blended in             │
        │  latents = scheduler.step(noise_pred, t, latents)                  │
        └──────────────────────────────────────────────────────────────────────┘

    Phase 2 mode (ablation.mode = "phase2"):
        Single store pass with aligned_pil (whole reference, no freq split).
        LF and HF caches receive identical KV.
        Flat lambdas (injection_scale constant, no temporal annealing).
    """
    device  = next(pipe.unet.parameters()).device
    mode    = cfg["ablation"]["mode"]         # "phase3" | "phase2"
    anneal  = cfg["ablation"]["temporal_anneal"]
    scale   = cfg["injection"]["injection_scale"]
    steps   = cfg["stage2"]["num_inference_steps"]
    g_scale = cfg["stage2"]["guidance_scale"]
    seed    = cfg["stage2"]["seed"]
    prompt  = cfg["stage2"]["prompt"]
    neg     = cfg["stage2"].get("negative_prompt", "")
    verbose = cfg["logging"].get("verbose", True)

    do_cfg = g_scale > 1.0

    # ── Set depth_routing on cache (Bug 1 fix — attribute must be set here) ─
    kv_cache = KVCache()
    kv_cache.depth_routing = cfg["ablation"]["depth_routing"]
    kv_cache.face_mask     = artifacts["face_mask"].to(device)

    # ── Encode all reference images to latents ─────────────────────────────
    _section("Encoding images to latents")

    src_lat = _encode_to_latent(pipe.vae, pipe.image_processor,
                                artifacts["source_pil"])

    if mode == "phase3":
        lf_lat = _encode_to_latent(pipe.vae, pipe.image_processor,
                                   artifacts["lf_pil"])
        hf_lat = _encode_to_latent(pipe.vae, pipe.image_processor,
                                   artifacts["hf_pil"])
        print(
            f"[stage2] Encoded: src={tuple(src_lat.shape)} | "
            f"lf={tuple(lf_lat.shape)} | hf={tuple(hf_lat.shape)}"
        )
    else:  # phase2
        ref_lat = _encode_to_latent(pipe.vae, pipe.image_processor,
                                    artifacts["aligned_pil"])
        print(
            f"[stage2] Encoded: src={tuple(src_lat.shape)} | "
            f"ref={tuple(ref_lat.shape)}  [phase2 — single reference]"
        )

    # ── Prompt embeddings ──────────────────────────────────────────────────
    _section("Encoding prompts")

    prompt_embeds, neg_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=neg,
    )

    # CFG: concat [uncond, cond] for a single batched forward pass
    text_embeds = torch.cat([neg_embeds, prompt_embeds]) if do_cfg \
                  else prompt_embeds

    # Store passes always use uncond — no guidance needed and it halves cost
    ref_embeds = neg_embeds if do_cfg else prompt_embeds

    print(f"[stage2] Prompt embeds: {tuple(prompt_embeds.shape)} | CFG={do_cfg}")

    # ── Scheduler timesteps ────────────────────────────────────────────────
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps   # (T,) descending, e.g. 999→0
    T         = len(timesteps)

    # ── Initial noisy latent ───────────────────────────────────────────────
    # Start from noisy source (img2img-like initialisation) so the source
    # pose, background, and lighting are preserved as a prior.
    gen   = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn_like(src_lat, generator=gen)
    latents = pipe.scheduler.add_noise(src_lat, noise, timesteps[:1])

    print(f"[stage2] Starting denoising | steps={T} | seed={seed}")
    _section(f"Denoising loop  [mode={mode}]")

    # ── Denoising loop ─────────────────────────────────────────────────────
    for step_idx, t in enumerate(timesteps):

        # ── Lambda update (once per step, before store passes) ────────────
        # Phase 2 always uses flat lambdas regardless of temporal_anneal flag.
        use_anneal = anneal and (mode == "phase3")
        kv_cache.set_lambdas(
            step_idx        = step_idx,
            total_steps     = T,
            temporal_anneal = use_anneal,
            injection_scale = scale,
        )

        # Clear stale KV from previous step
        kv_cache.clear()

        if verbose and (step_idx % 5 == 0 or step_idx == T - 1):
            print(f"  step {step_idx+1:3d}/{T} | t={int(t):4d} | "
                  f"{kv_cache.summary()}")

        # ── Phase 3: two store passes (LF then HF) ────────────────────────
        if mode == "phase3":

            # Pass 1 — store LF: captures global structure into deep layers
            # Add fresh noise at timestep t to the LF latent so the reference
            # and source are at the same noise level when features are compared.
            lf_noise = torch.randn_like(lf_lat, generator=gen)
            noisy_lf = pipe.scheduler.add_noise(lf_lat, lf_noise, t.unsqueeze(0))

            kv_cache.set_freq_mode("lf")
            kv_cache.set_mode("store")
            _store_pass(pipe.unet, noisy_lf, t, ref_embeds, kv_cache)

            # Pass 2 — store HF: captures texture detail into shallow layers
            hf_noise = torch.randn_like(hf_lat, generator=gen)
            noisy_hf = pipe.scheduler.add_noise(hf_lat, hf_noise, t.unsqueeze(0))

            kv_cache.set_freq_mode("hf")
            kv_cache.set_mode("store")
            _store_pass(pipe.unet, noisy_hf, t, ref_embeds, kv_cache)

        # ── Phase 2: single store pass (whole aligned reference) ──────────
        else:
            ref_noise = torch.randn_like(ref_lat, generator=gen)
            noisy_ref = pipe.scheduler.add_noise(ref_lat, ref_noise, t.unsqueeze(0))

            kv_cache.set_freq_mode("lf")
            kv_cache.set_mode("store")
            _store_pass(pipe.unet, noisy_ref, t, ref_embeds, kv_cache)

            # Mirror: both depth levels find cached features
            # (shallow layers will read from _hf_cache which is now a copy of _lf_cache)
            kv_cache._hf_cache = dict(kv_cache._lf_cache)

        # ── Pass 3 — inject: source denoising step ────────────────────────
        kv_cache.set_mode("inject")

        # Double batch for CFG: [uncond_latent, cond_latent]
        src_input  = torch.cat([latents] * 2) if do_cfg else latents
        noise_pred = pipe.unet(
            src_input,
            t.unsqueeze(0) if t.ndim == 0 else t,
            encoder_hidden_states  = text_embeds if do_cfg else prompt_embeds,
            cross_attention_kwargs = {"kv_cache": kv_cache},
        ).sample

        # CFG: linearly combine uncond and cond predictions
        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + g_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        # Scheduler step: predict previous (less noisy) latent
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Free attention cache VRAM after each step — the 3 U-Net passes per
        # step accumulate cached KV tensors; clearing avoids OOM on T4.
        torch.cuda.empty_cache()

    print(f"\n[stage2] Denoising complete. Final latent: {tuple(latents.shape)}")
    return latents


# ══════════════════════════════════════════════════════════════════════════════
# DECODE + SAVE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def decode_latent(pipe, latents: torch.Tensor) -> Image.Image:
    """
    Decode a latent tensor to a PIL RGB image.

    Divides by scaling_factor before decoding (inverse of what encode does).
    pipe.image_processor.postprocess() handles the [-1,1] → [0,255] clamp
    and uint8 cast.
    """
    latents = latents / pipe.vae.config.scaling_factor
    image   = pipe.vae.decode(latents).sample
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def save_results(
    result_pil:  Image.Image,
    artifacts:   dict,
    cfg:         dict,
    stage2_meta: dict,
):
    """
    Save all outputs to the output directory.

    Files written:
        result.png          — final face swap output
        comparison.png      — 4-panel: source | aligned | LF | result
        meta_stage2.json    — complete Stage 2 config for reproducibility
    """
    output_dir = cfg["paths"]["output_dir"]
    mode       = cfg["ablation"]["mode"]
    os.makedirs(output_dir, exist_ok=True)

    # ── Result image ───────────────────────────────────────────────────────
    suffix      = "_phase2" if mode == "phase2" else ""
    result_path = os.path.join(output_dir, f"result{suffix}.png")
    result_pil.save(result_path)

    # ── Comparison panel ───────────────────────────────────────────────────
    # 4 panels side-by-side: Source | Aligned Ref | LF component | Result
    # Makes it easy to visually verify each stage of the pipeline at a glance.
    panels      = [
        artifacts["source_pil"],
        artifacts["aligned_pil"],
        artifacts["lf_pil"],
        result_pil,
    ]
    labels      = ["Source", "Aligned Ref", "LF (structure)", f"Result ({mode})"]
    W, H        = panels[0].size
    pad         = 4
    label_h     = 20
    total_w     = W * len(panels) + pad * (len(panels) + 1)
    total_h     = H + label_h + pad * 2

    comparison  = Image.new("RGB", (total_w, total_h), (30, 30, 30))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
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

    comp_path = os.path.join(output_dir, f"comparison{suffix}.png")
    comparison.save(comp_path)

    # ── Stage 2 meta ───────────────────────────────────────────────────────
    meta_path = os.path.join(output_dir, "meta_stage2.json")
    with open(meta_path, "w") as f:
        json.dump(stage2_meta, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n[stage2] Outputs saved → {os.path.abspath(output_dir)}/")
    for fname in [f"result{suffix}.png", f"comparison{suffix}.png", "meta_stage2.json"]:
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
    Contains every parameter used in this run — config + stage1 provenance.
    """
    return {
        # ── Stage 2 params ────────────────────────────────────────────────
        "model_id"          : cfg["stage2"]["model_id"],
        "scheduler"         : cfg["stage2"].get("scheduler", "DDIM"),
        "num_inference_steps": cfg["stage2"]["num_inference_steps"],
        "guidance_scale"    : cfg["stage2"]["guidance_scale"],
        "seed"              : cfg["stage2"]["seed"],
        "prompt"            : cfg["stage2"]["prompt"],
        "negative_prompt"   : cfg["stage2"].get("negative_prompt", ""),
        "torch_dtype"       : cfg["stage2"]["torch_dtype"],

        # ── Injection / ablation params ───────────────────────────────────
        "injection_scale"   : cfg["injection"]["injection_scale"],
        "ablation"          : {
            "mode"           : cfg["ablation"]["mode"],
            "depth_routing"  : cfg["ablation"]["depth_routing"],
            "temporal_anneal": cfg["ablation"]["temporal_anneal"],
            "decomposition"  : cfg["ablation"]["decomposition"],
            "mask_type"      : cfg["ablation"]["mask_type"],
        },

        # ── Stage 1 provenance ────────────────────────────────────────────
        "stage1_meta"       : stage1_meta,

        # ── Provenance ────────────────────────────────────────────────────
        "stage2_script"     : os.path.abspath(__file__),
        "timestamp"         : datetime.datetime.now().isoformat(),
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
    print(f"  Stage 2 complete")
    print(f"{'═' * 60}")
    print(f"  result           : {os.path.abspath(result_path)}")
    print(f"  mode             : {cfg['ablation']['mode']}")
    print(f"  depth_routing    : {cfg['ablation']['depth_routing']}")
    print(f"  injection_scale  : {cfg['injection']['injection_scale']}")
    print(f"  temporal_anneal  : {cfg['ablation']['temporal_anneal']}")
    print(f"  yaw_diff (s1)    : {stage1_meta.get('yaw_diff_deg', '?'):.1f}°")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_stage2(cfg: dict):
    """
    Execute the full Stage 2 pipeline.

    1. Load artifacts from disk
    2. Load SD model + patch UNet
    3. Run denoising loop (3-pass KV injection per step)
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
    artifacts = load_artifacts(cfg["paths"]["artifacts_dir"])
    stage1_meta = artifacts["meta"]

    # ── Step 2: Load model + patch UNet ──────────────────────────────────
    _section("Step 2 — Load Stable Diffusion + patch UNet")
    pipe = load_pipeline(cfg)

    # patch_unet_attention is idempotent — safe to call even if already patched
    pipe.unet, depth_map = patch_unet_attention(pipe.unet)

    n_deep    = sum(1 for v in depth_map.values() if v == "deep")
    n_shallow = sum(1 for v in depth_map.values() if v == "shallow")
    n_skip    = sum(1 for v in depth_map.values() if v is None)
    print(
        f"[stage2] UNet patched | "
        f"deep={n_deep} shallow={n_shallow} skipped={n_skip} | "
        f"depth_routing={cfg['ablation']['depth_routing']}"
    )

    # ── Step 3: Denoising loop ────────────────────────────────────────────
    latents = run_denoising_loop(pipe, artifacts, cfg)

    # ── Step 4: Decode ────────────────────────────────────────────────────
    _section("Step 4 — Decode latent → image")
    generated_pil = decode_latent(pipe, latents)
    print(f"[stage2] Decoded: {generated_pil.size} {generated_pil.mode}")
 
    # ── Step 4b: Composite ────────────────────────────────────────────────
    # Paste generated face onto source background using the face mask.
    # This preserves the source image exactly outside the mask region —
    # background, hair, neck, clothing are all taken directly from source.
    # Poisson blending removes the color seam at the mask boundary.
    # Skip compositing only when mask_type="none" (global injection mode).
    _section("Step 4b — Composite result onto source")
    if cfg["ablation"]["mask_type"] == "none":
        result_pil = generated_pil
        print("[stage2] mask_type=none — skipping compositing (global injection mode).")
    else:
        result_pil = composite_result(
            generated_pil    = generated_pil,
            source_pil       = artifacts["source_pil"],
            face_mask_tensor = artifacts["face_mask"],
            cfg              = cfg,
        )
        print(f"[stage2] Compositing complete: {result_pil.size}")
 
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
        description="Stage 2 — KV-injection diffusion pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file"
    )

    # ── Directory overrides ────────────────────────────────────────────────
    p.add_argument("--artifacts",   default=None,
                   help="Override paths.artifacts_dir")
    p.add_argument("--output-dir",  default=None,
                   help="Override paths.output_dir")

    # ── Ablation overrides ─────────────────────────────────────────────────
    p.add_argument("--mode",        default=None,
                   choices=["phase3", "phase2"],
                   help="Override ablation.mode")
    p.add_argument("--depth-routing", default=None,
                   choices=["correct", "swapped", "uniform"],
                   help="Override ablation.depth_routing  (A2 ablation)")
    p.add_argument("--injection-scale", type=float, default=None,
                   help="Override injection.injection_scale  (A6 sweep)")
    p.add_argument("--no-anneal",   action="store_true",
                   help="Disable temporal annealing  (A4 ablation, flat lambdas)")

    # ── Diffusion overrides ────────────────────────────────────────────────
    p.add_argument("--steps",       type=int,   default=None,
                   help="Override stage2.num_inference_steps")
    p.add_argument("--guidance",    type=float, default=None,
                   help="Override stage2.guidance_scale")
    p.add_argument("--seed",        type=int,   default=None,
                   help="Override stage2.seed")
    p.add_argument("--prompt",      default=None,
                   help="Override stage2.prompt")
    p.add_argument("--model-id",    default=None,
                   help="Override stage2.model_id")

    # ── Utility ────────────────────────────────────────────────────────────
    p.add_argument("--dry-run",     action="store_true",
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
                f"  3. Use a smaller SAM model (vit_b) and re-run Stage 1\n"
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