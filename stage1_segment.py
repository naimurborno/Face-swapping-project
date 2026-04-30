# stage1_segment.py
"""
Stage 1 — Donor alignment, segmentation, and mixed-frequency prior construction.

Entry point for the first stage of the Mixed-Frequency Prior Guided Inpainting
pipeline. Reads configs/default.yaml, runs all Stage 1 steps, and saves every
artifact the diffusion stage needs into the artifacts/ directory.

Pipeline steps (in order):
    1. Load content + donor images from disk
    2. Align donor to content spatial coordinate system
    3. Object mask: SAM / none  (ablation.mask_type)
    4. Frequency decomposition of both content and aligned donor
    5. Build mixed-frequency prior  P = α·S_LF + β·R̃_LF + γ·R̃_HF
    6. Embed prior into masked input  X₀ = (1−M)⊙S + M⊙P
    7. Format conversion: numpy BGR → PIL RGB + mask tensor
    8. Save all artifacts to artifacts/

After this script completes:
    - Stage 2 (stage2_diffusion.py) can be run immediately
    - SAM is fully unloaded from memory before this script exits
    - Artifacts contain everything Stage 2 needs — no original images required
      at diffusion time

The key artifact produced here is masked_input.png  (X₀): the content image
with the mixed-frequency prior embedded inside the mask region. This is what
the inpainting model receives as its starting point.

Usage:
    # Default config:
    python stage1_segment.py

    # Custom config file:
    python stage1_segment.py --config configs/my_experiment.yaml

    # Override specific values without editing the yaml:
    python stage1_segment.py --mask-type none
    python stage1_segment.py --decomp fft
    python stage1_segment.py --content inputs/wall.jpg --donor inputs/marble.jpg
    python stage1_segment.py --alpha 0.7 --beta 0.4 --gamma 0.9

    # Dry run — prints resolved config and exits without running:
    python stage1_segment.py --dry-run

Output (all saved to artifacts/ by default):
    content_pil.png       — content image S resized to target_size, RGB
    donor_aligned_pil.png — donor R̃ aligned to content, resized, RGB
    s_lf_pil.png          — S_LF (content low-frequency component), RGB
    r_hf_pil.png          — R̃_HF shifted to [0,255] for visualization, RGB
    prior_pil.png         — prior P (the mixed signal inside the mask), RGB
    masked_input_pil.png  — X₀ = (1−M)⊙S + M⊙P  (inpainting model input), RGB
    object_mask.pt        — (1, 1, S, S) float32 mask tensor for shallow KV gating + blended latent anchoring
    meta.json             — all parameters + stats, read by Stage 2 at startup

Dependencies:
    pip install opencv-python-headless torch pillow pyyaml numpy
    pip install git+https://github.com/facebookresearch/segment-anything.git
    Download SAM checkpoint (see README for links)
"""

import sys
import os
import json
import argparse
import datetime
import traceback

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
# Support running from project root OR from inside the project directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")
if os.path.isdir(_CORE):
    sys.path.insert(0, _CORE)
else:
    sys.path.insert(0, _HERE)

try:
    from alignment   import run_donor_alignment, AlignmentResult
    from segmentation import load_sam_model, get_object_mask, mask_to_tensor
    from decomposition import (
        decompose,
        build_prior,
        prior_result_to_pil,
        PriorResult,
    )
except ImportError as e:
    print(f"\n[stage1] FATAL — could not import core modules: {e}")
    print("  Make sure core/ exists and contains alignment.py, segmentation.py,")
    print("  decomposition.py. Run from the project root directory.")
    traceback.print_exc()
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADER
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG_PATH = os.path.join(_HERE, "configs", "default.yaml")


def load_config(config_path: str) -> dict:
    """
    Load and return the YAML config as a nested dict.

    Raises FileNotFoundError if the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[stage1] Config file not found: {config_path}\n"
            f"  Expected at: {os.path.abspath(config_path)}"
        )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """
    Apply CLI argument overrides on top of the loaded config.

    Only overrides keys that were explicitly provided on the command line
    (i.e. not None). This allows ablations without editing the yaml:
        python stage1_segment.py --mask-type none --decomp fft --alpha 0.8
    """
    # ── Paths ─────────────────────────────────────────────────────────────
    if args.content is not None:
        cfg["paths"]["content_image"] = args.content
    if args.donor is not None:
        cfg["paths"]["donor_image"] = args.donor
    if args.artifacts_dir is not None:
        cfg["paths"]["artifacts_dir"] = args.artifacts_dir
    if args.sam_checkpoint is not None:
        cfg["paths"]["sam_checkpoint"] = args.sam_checkpoint
    if args.sam_type is not None:
        cfg["paths"]["sam_model_type"] = args.sam_type

    # ── Ablation / method overrides ────────────────────────────────────────
    if args.mask_type is not None:
        cfg["ablation"]["mask_type"] = args.mask_type
    if args.decomp is not None:
        cfg["ablation"]["decomposition"] = args.decomp
    if args.align_method is not None:
        cfg["alignment"]["method"] = args.align_method

    # ── Prior coefficients ─────────────────────────────────────────────────
    if args.alpha is not None:
        cfg["prior"]["alpha"] = args.alpha
    if args.beta is not None:
        cfg["prior"]["beta"] = args.beta
    if args.gamma is not None:
        cfg["prior"]["gamma"] = args.gamma

    # ── Image ──────────────────────────────────────────────────────────────
    if args.target_size is not None:
        cfg["image"]["target_size"] = args.target_size

    return cfg


def print_config(cfg: dict):
    """Print the fully resolved config that will actually be used."""
    print("\n┌─ Resolved config ─────────────────────────────────────────")
    print(f"│  content_image  : {cfg['paths']['content_image']}")
    print(f"│  donor_image    : {cfg['paths']['donor_image']}")
    print(f"│  artifacts_dir  : {cfg['paths']['artifacts_dir']}")
    print(f"│  target_size    : {cfg['image']['target_size']}px")
    print(f"│")
    print(f"│  [prior]")
    print(f"│  alpha          : {cfg['prior']['alpha']}  (content LF weight)")
    print(f"│  beta           : {cfg['prior']['beta']}  (donor LF weight)")
    print(f"│  gamma          : {cfg['prior']['gamma']}  (donor HF weight)")
    print(f"│  histogram_match: {cfg['prior']['histogram_match']}")
    print(f"│")
    print(f"│  [ablation]")
    print(f"│  mask_type      : {cfg['ablation']['mask_type']}")
    print(f"│  decomposition  : {cfg['ablation']['decomposition']}")
    print(f"│")
    print(f"│  [alignment]")
    print(f"│  method         : {cfg['alignment']['method']}")
    print(f"│")
    if cfg["ablation"]["mask_type"] == "sam":
        print(f"│  [SAM]")
        print(f"│  sam_checkpoint : {cfg['paths']['sam_checkpoint']}")
        print(f"│  sam_model_type : {cfg['paths']['sam_model_type']}")
        print(f"│  prompt_strategy: {cfg['stage1']['sam']['prompt_strategy']}")
        print(f"│  pred_iou_thresh: {cfg['stage1']['sam']['pred_iou_thresh']}")
        print(f"│")
    if cfg["ablation"]["decomposition"] == "gaussian":
        print(f"│  [Gaussian decomposition]")
        print(f"│  kernel         : {cfg['stage1']['gaussian']['kernel']}")
        print(f"│  sigma          : {cfg['stage1']['gaussian']['sigma']}")
    elif cfg["ablation"]["decomposition"] == "fft":
        print(f"│  [FFT decomposition]")
        print(f"│  cutoff_ratio   : {cfg['stage1']['fft']['cutoff_ratio']}")
    print(f"└───────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP RUNNERS
# Each step is a named function so failures have clear traceback origins.
# ══════════════════════════════════════════════════════════════════════════════

def step_load_images(cfg: dict):
    """
    Step 1: Load content and donor images from disk.

    Both images are read as-is (any resolution). Resizing to target_size
    happens later in step_convert() so all intermediate operations work
    at original resolution.

    Returns:
        content_bgr : (H_c, W_c, 3) uint8 BGR — content image S
        donor_bgr   : (H_d, W_d, 3) uint8 BGR — donor image R
    """
    content_path = cfg["paths"]["content_image"]
    donor_path   = cfg["paths"]["donor_image"]

    for label, path, flag in [
        ("content", content_path, "--content"),
        ("donor",   donor_path,   "--donor"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[stage1] {label} image not found: {path}\n"
                f"  Update paths.{label}_image in configs/default.yaml\n"
                f"  or pass {flag} <path>"
            )

    content_bgr = cv2.imread(content_path)
    donor_bgr   = cv2.imread(donor_path)

    if content_bgr is None:
        raise ValueError(f"[stage1] cv2.imread failed for content: {content_path}")
    if donor_bgr is None:
        raise ValueError(f"[stage1] cv2.imread failed for donor: {donor_path}")

    print(f"[stage1] Content : {content_path}  ({content_bgr.shape[1]}×{content_bgr.shape[0]}px)")
    print(f"[stage1] Donor   : {donor_path}  ({donor_bgr.shape[1]}×{donor_bgr.shape[0]}px)")

    return content_bgr, donor_bgr


def step_align(cfg: dict, content_bgr: np.ndarray, donor_bgr: np.ndarray) -> AlignmentResult:
    """
    Step 2: Align donor image into content image's spatial coordinate system.

    The aligned donor R̃ has the same spatial dimensions as the content image S.
    This is required because build_prior() computes P = α·S_LF + β·R̃_LF + γ·R̃_HF
    pixel-by-pixel — misaligned inputs produce a spatially incoherent prior that
    the inpainting model will treat as noise and ignore.

    Method is read from cfg["alignment"]["method"]:
        "resize"  — simple resize, default, no extra dependencies
        "tile"    — tile donor to fill content canvas (for tileable textures)
        "match"   — ORB feature matching + affine RANSAC
        "flow"    — dense optical flow warp (Farneback)
        "affine"  — explicit affine from user-supplied point pairs

    Returns:
        AlignmentResult with aligned_donor (BGR), method_used, fallback, meta
    """
    method     = cfg["alignment"]["method"]
    content_hw = content_bgr.shape[:2]   # (H, W)

    alignment_result = run_donor_alignment(
        donor_bgr   = donor_bgr,
        content_bgr = content_bgr,
        method      = method,
    )

    fallback_note = "  [fallback]" if alignment_result.fallback else ""
    print(
        f"[stage1] Alignment done | "
        f"method={alignment_result.method_used}{fallback_note} | "
        f"donor → content: {donor_bgr.shape[1]}×{donor_bgr.shape[0]}px "
        f"→ {content_hw[1]}×{content_hw[0]}px"
    )
    if alignment_result.meta:
        for k, v in alignment_result.meta.items():
            print(f"           {k}: {v}")

    return alignment_result


def step_mask(cfg: dict, content_bgr: np.ndarray) -> np.ndarray:
    """
    Step 3: Produce the binary object mask for the content image.

    The mask defines exactly two regions:
        mask=255  — editable region: mixed-frequency prior is applied here;
                    diffusion fills this area guided by the prior + shallow KV.
        mask=0    — frozen region: blended latent anchoring guarantees this
                    region is pixel-identical to the content image in the output.

    mask_type is read from ablation.mask_type:
        "sam"  → SAM segmentation with center_point or bbox prompt (default)
        "none" → full-image mask — entire image is editable (ablation baseline)

    SAM is loaded, used, and immediately deleted inside this function so its
    VRAM is free before Stage 2 loads the inpainting diffusion model.

    Returns:
        object_mask_uint8 : (H, W) uint8, values 0/255
    """
    mask_type = cfg["ablation"]["mask_type"]
    sam_cfg   = cfg["stage1"]["sam"]
    predictor = None

    if mask_type == "sam":
        sam_checkpoint = cfg["paths"]["sam_checkpoint"]
        sam_model_type = cfg["paths"]["sam_model_type"]

        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(
                f"[stage1] SAM checkpoint not found: {sam_checkpoint}\n"
                f"  Update paths.sam_checkpoint in configs/default.yaml\n"
                f"  or pass --sam-checkpoint <path>\n"
                f"  Download SAM checkpoints from:\n"
                f"  https://github.com/facebookresearch/segment-anything#model-checkpoints\n"
                f"\n"
                f"  To skip SAM, use ablation.mask_type: none\n"
                f"  or pass --mask-type none"
            )

        predictor = load_sam_model(
            checkpoint_path = sam_checkpoint,
            model_type      = sam_model_type,
        )

    object_mask_uint8 = get_object_mask(
        image_bgr              = content_bgr,
        mask_type              = mask_type,
        predictor              = predictor,
        prompt_strategy        = sam_cfg["prompt_strategy"],
        pred_iou_thresh        = sam_cfg["pred_iou_thresh"],
        stability_score_thresh = sam_cfg["stability_score_thresh"],
        dilation_px            = sam_cfg.get("dilation_px", 0),
    )

    # Free SAM from GPU memory immediately — Stage 2 needs that VRAM
    if predictor is not None:
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[stage1] SAM predictor deleted — VRAM freed for Stage 2")

    coverage = float(object_mask_uint8.astype(bool).mean() * 100)
    print(
        f"[stage1] Mask ready | "
        f"mask_type={mask_type} | "
        f"coverage={coverage:.1f}%  "
        f"{'✓' if 5.0 < coverage < 95.0 else '⚠ unusual coverage — check mask quality'}"
    )

    return object_mask_uint8


def step_decompose(cfg: dict, content_bgr: np.ndarray, donor_aligned_bgr: np.ndarray):
    """
    Step 4: Frequency decompose both content S and aligned donor R̃.

    Produces four frequency bands:
        S_LF  — content low-frequency:  coarse geometry, depth gradients,
                 illumination field. Governs spatial structure in the prior.
        S_HF  — content high-frequency: fine surface detail of S (not used
                 in prior construction, kept for logging and ablation A1).
        R̃_LF  — donor low-frequency:   coarse material color and tone.
                 Contributes donor semantic appearance to the prior.
        R̃_HF  — donor high-frequency:  fine surface detail, texture,
                 microstructure. The primary transfer signal.

    The separation method (Gaussian / FFT / none) is read from
    ablation.decomposition and applied identically to both images so their
    frequency bands are computed at the same cutoff frequency.

    Returns:
        content_decomp : DecomposeResult for S  (S_LF, S_HF, method, stats)
        donor_decomp   : DecomposeResult for R̃  (R̃_LF, R̃_HF, method, stats)
    """
    method    = cfg["ablation"]["decomposition"]
    gauss_cfg = cfg["stage1"]["gaussian"]
    fft_cfg   = cfg["stage1"]["fft"]

    decomp_kwargs = dict(
        method       = method,
        kernel       = gauss_cfg["kernel"],
        sigma        = gauss_cfg["sigma"],
        cutoff_ratio = fft_cfg["cutoff_ratio"],
    )

    content_decomp = decompose(content_bgr,   **decomp_kwargs)
    donor_decomp   = decompose(donor_aligned_bgr, **decomp_kwargs)

    print(
        f"[stage1] Decomposition done | method={method}\n"
        f"         Content : LF_energy={content_decomp.lf_energy:.1f}  "
        f"HF_std={content_decomp.hf_std:.2f}"
        f"  {'✓' if content_decomp.hf_std > 5 else '⚠ low HF — flat content?'}\n"
        f"         Donor   : LF_energy={donor_decomp.lf_energy:.1f}  "
        f"HF_std={donor_decomp.hf_std:.2f}"
        f"  {'✓' if donor_decomp.hf_std > 5 else '⚠ low HF — flat donor?'}"
    )

    return content_decomp, donor_decomp


def step_build_prior(
    cfg:               dict,
    content_bgr:       np.ndarray,
    donor_aligned_bgr: np.ndarray,
    object_mask_uint8: np.ndarray,
) -> "PriorResult":
    """
    Step 5: Build the mixed-frequency prior P and embed it into X₀.

    Prior construction:
        P_raw = α·S_LF + β·R̃_LF + γ·R̃_HF
        P     = histogram_match(P_raw, S_masked)   # photometric plausibility
        X₀    = (1−M)⊙S + M⊙P

    The three coefficients determine what each frequency band contributes:
        α (content LF weight) — how much of S's spatial structure survives
          inside the mask. High = strict geometry preservation (retexturing).
          Low = donor geometry can reshape the region (face/object swap).
        β (donor LF weight)   — how much of R̃'s coarse material tone enters
          the prior. Sets the color palette register inside the mask.
        γ (donor HF weight)   — donor fine detail strength. Always high —
          this is the primary transfer signal. The inpainting model amplifies it.

    Histogram matching normalizes P's tone to match the surrounding source
    context so the inpainting model does not treat the prior as corruption.

    Reads coefficients from cfg["prior"]: alpha, beta, gamma, histogram_match.

    Returns:
        PriorResult(P, X0, S_LF, S_HF, R_LF, R_HF, alpha, beta, gamma)
    """
    prior_cfg = cfg["prior"]

    # Pass donor_aligned_bgr directly — build_prior() runs its own decomposition
    # internally. Reconstructing from decomp.LF + decomp.HF would introduce
    # float→uint8 rounding errors that alter the frequency split on the re-decomp.
    prior_result = build_prior(
        source_bgr        = content_bgr,
        donor_aligned_bgr = donor_aligned_bgr,
        mask              = object_mask_uint8,
        alpha             = prior_cfg["alpha"],
        beta              = prior_cfg["beta"],
        gamma             = prior_cfg["gamma"],
        histogram_match   = prior_cfg["histogram_match"],
        method            = cfg["ablation"]["decomposition"],
        kernel            = cfg["stage1"]["gaussian"]["kernel"],
        sigma             = cfg["stage1"]["gaussian"]["sigma"],
        cutoff_ratio      = cfg["stage1"]["fft"]["cutoff_ratio"],
    )

    # Verify X₀ correctness: outside-mask pixels must be identical to content
    mask_bool = object_mask_uint8.astype(bool)
    if mask_bool.any() and not mask_bool.all():
        outside_content = content_bgr[~mask_bool]
        outside_x0      = prior_result.X0[~mask_bool]
        if not np.array_equal(outside_content, outside_x0):
            raise RuntimeError(
                "[stage1] FATAL — X₀ outside-mask pixels do not match content image.\n"
                "  This means build_prior() has modified pixels outside the mask.\n"
                "  Blended latent anchoring relies on X₀ = S outside the mask.\n"
                "  Check decomposition.build_masked_input()."
            )

    print(
        f"[stage1] Prior ready | "
        f"α={prior_cfg['alpha']}  β={prior_cfg['beta']}  γ={prior_cfg['gamma']} | "
        f"histogram_match={prior_cfg['histogram_match']} | "
        f"P range=[{prior_result.P.min()},{prior_result.P.max()}] | "
        f"X₀ outside-mask integrity: ✓"
    )

    return prior_result


def step_convert(
    cfg:              dict,
    content_bgr:      np.ndarray,
    donor_aligned_bgr: np.ndarray,
    content_decomp,
    donor_decomp,
    prior_result:     "PriorResult",
    object_mask_uint8: np.ndarray,
):
    """
    Step 6: Convert all numpy outputs to pipeline-ready PIL images and tensors.

    Resizes everything to target_size×target_size (SD2.1-base: 512px).
    All Stage 2 inputs are at this resolution.

    Returns:
        content_pil        : PIL RGB  — content image S
        donor_aligned_pil  : PIL RGB  — aligned donor R̃
        s_lf_pil           : PIL RGB  — S_LF (content low-frequency)
        r_hf_pil           : PIL RGB  — R̃_HF shifted to [0,255]
        prior_pil          : PIL RGB  — prior P (inside-mask only, for visualization)
        masked_input_pil   : PIL RGB  — X₀ = (1−M)⊙S + M⊙P (inpainting input)
        object_mask_tensor : (1,1,target_size,target_size) float32 [0,1]
    """
    target_size = cfg["image"]["target_size"]

    def _to_pil(bgr: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).resize((target_size, target_size), Image.LANCZOS)

    def _hf_to_pil(hf: np.ndarray) -> Image.Image:
        """Shift HF residual from [-128,127] to [0,255] for saving."""
        shifted = np.clip(hf.astype(np.float32) + 128.0, 0, 255).astype(np.uint8)
        rgb     = cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).resize((target_size, target_size), Image.LANCZOS)

    content_pil       = _to_pil(content_bgr)
    donor_aligned_pil = _to_pil(donor_aligned_bgr)
    s_lf_pil          = _to_pil(content_decomp.LF.astype(np.uint8))
    r_hf_pil          = _hf_to_pil(donor_decomp.HF)
    prior_pil         = _to_pil(prior_result.P)
    masked_input_pil  = _to_pil(prior_result.X0)

    object_mask_tensor = mask_to_tensor(object_mask_uint8)

    print(
        f"[stage1] Conversion done | "
        f"PIL size={content_pil.size} | "
        f"mask tensor shape={tuple(object_mask_tensor.shape)} "
        f"range=[{float(object_mask_tensor.min()):.2f}, {float(object_mask_tensor.max()):.2f}]"
    )

    return (
        content_pil,
        donor_aligned_pil,
        s_lf_pil,
        r_hf_pil,
        prior_pil,
        masked_input_pil,
        object_mask_tensor,
    )


def step_save(
    cfg:               dict,
    content_pil:       "Image.Image",
    donor_aligned_pil: "Image.Image",
    s_lf_pil:          "Image.Image",
    r_hf_pil:          "Image.Image",
    prior_pil:         "Image.Image",
    masked_input_pil:  "Image.Image",
    object_mask_tensor: "torch.Tensor",
    meta:              dict,
):
    """
    Step 7: Save all artifacts to the artifacts directory.

    Files written:
        content_pil.png        — content image S (resized, RGB)
        donor_aligned_pil.png  — aligned donor R̃ (resized, RGB)
        s_lf_pil.png           — S_LF component (resized, RGB)
        r_hf_pil.png           — R̃_HF shifted [0,255] (resized, RGB)
        prior_pil.png          — prior P — what's inside the mask (resized, RGB)
        masked_input_pil.png   — X₀ = (1−M)⊙S + M⊙P — inpainting model input
        object_mask.pt         — (1,1,S,S) float32 tensor
        meta.json              — all metadata Stage 2 reads at startup

    All files are required by Stage 2. Do not manually delete any of them.
    object_mask.pt is used for both kv_cache.face_mask (shallow KV token gating)
    and the latent-space mask Mz (blended latent anchoring every denoising step).
    """
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    saves = {
        "content_pil.png"       : ("pil",    content_pil),
        "donor_aligned_pil.png" : ("pil",    donor_aligned_pil),
        "s_lf_pil.png"          : ("pil",    s_lf_pil),
        "r_hf_pil.png"          : ("pil",    r_hf_pil),
        "prior_pil.png"         : ("pil",    prior_pil),
        "masked_input_pil.png"  : ("pil",    masked_input_pil),
        "object_mask.pt"        : ("tensor", object_mask_tensor),
        "meta.json"             : ("json",   meta),
    }

    print(f"\n[stage1] Saving artifacts → {os.path.abspath(artifacts_dir)}/")
    for fname, (kind, obj) in saves.items():
        fpath = os.path.join(artifacts_dir, fname)
        if kind == "pil":
            obj.save(fpath)
        elif kind == "tensor":
            torch.save(obj, fpath)
        elif kind == "json":
            with open(fpath, "w") as f:
                json.dump(obj, f, indent=2)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname:<28}  {size_kb:7.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# META BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_meta(
    cfg:               dict,
    alignment_result:  AlignmentResult,
    content_decomp,
    donor_decomp,
    prior_result:      "PriorResult",
    object_mask_uint8: np.ndarray,
) -> dict:
    """
    Build the meta.json dict that Stage 2 reads at startup.

    Contains every parameter and stat needed to:
        - Reproduce the run exactly (decomp method, mask type, α/β/γ, etc.)
        - Verify artifact quality (HF_std, mask_coverage, prior stats)
        - Support the ablation runner (all ablation flags recorded)
        - Audit the pipeline (provenance, timestamp, absolute paths)

    Stage 2 (stage2_diffusion.py) reads this at load time and logs a
    summary of what Stage 1 produced before starting the denoising loop.
    """
    coverage = float(object_mask_uint8.astype(bool).mean() * 100)

    return {
        # ── Image paths ──────────────────────────────────────────────────
        "content_path"       : os.path.abspath(cfg["paths"]["content_image"]),
        "donor_path"         : os.path.abspath(cfg["paths"]["donor_image"]),

        # ── Alignment ────────────────────────────────────────────────────
        "alignment_method"   : alignment_result.method_used,
        "alignment_fallback" : alignment_result.fallback,
        "alignment_meta"     : alignment_result.meta,

        # ── Decomposition ─────────────────────────────────────────────────
        "decomp_method"      : cfg["ablation"]["decomposition"],
        "gauss_kernel"       : cfg["stage1"]["gaussian"]["kernel"],
        "gauss_sigma"        : cfg["stage1"]["gaussian"]["sigma"],
        "fft_cutoff"         : cfg["stage1"]["fft"]["cutoff_ratio"],

        # Content decomposition stats
        "content_lf_energy"  : round(float(content_decomp.lf_energy), 3),
        "content_hf_std"     : round(float(content_decomp.hf_std),    3),

        # Donor decomposition stats
        "donor_lf_energy"    : round(float(donor_decomp.lf_energy),   3),
        "donor_hf_std"       : round(float(donor_decomp.hf_std),      3),

        # ── Prior construction ────────────────────────────────────────────
        "alpha"              : float(prior_result.alpha),
        "beta"               : float(prior_result.beta),
        "gamma"              : float(prior_result.gamma),
        "histogram_match"    : cfg["prior"]["histogram_match"],
        "prior_mean"         : round(float(prior_result.P.mean()), 3),
        "prior_std"          : round(float(prior_result.P.std()),  3),

        # ── Mask ─────────────────────────────────────────────────────────
        "mask_type"          : cfg["ablation"]["mask_type"],
        "sam_model_type"     : cfg["paths"].get("sam_model_type", "n/a"),
        "prompt_strategy"    : cfg["stage1"]["sam"].get("prompt_strategy", "n/a"),
        "mask_coverage_pct"  : round(coverage, 2),

        # ── Output format ─────────────────────────────────────────────────
        "target_size"        : cfg["image"]["target_size"],

        # ── Ablation flags (Stage 2 and ablation_runner read these) ──────
        "ablation": {
            "decomposition"      : cfg["ablation"]["decomposition"],
            "mask_type"          : cfg["ablation"]["mask_type"],
            "prior_construction" : cfg["ablation"].get("prior_construction", "full"),
            "blended_anchoring"  : cfg["ablation"].get("blended_anchoring",  True),
            "shallow_injection"  : cfg["ablation"].get("shallow_injection",  True),
            "histogram_match"    : cfg["ablation"].get("histogram_match",    True),
            "temporal_anneal"    : cfg["ablation"].get("temporal_anneal",    True),
            "compositing"        : cfg["ablation"].get("compositing",        "simple"),
        },

        # ── Provenance ────────────────────────────────────────────────────
        "stage1_script"      : os.path.abspath(__file__),
        "timestamp"          : datetime.datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_stage1(cfg: dict):
    """
    Execute all Stage 1 steps in order.

    All side effects (printing, file I/O) are contained inside each step
    function. This function is the single sequence:
        load → align → mask → decompose → prior → convert → meta → save

    Args:
        cfg : Fully resolved config dict (yaml + CLI overrides applied).

    Returns:
        artifacts_dir : Absolute path to the artifacts directory.
        meta          : The meta dict written to meta.json.
    """

    # ── Step 1: Load images ───────────────────────────────────────────────
    _section("Step 1 — Load images")
    content_bgr, donor_bgr = step_load_images(cfg)

    # ── Step 2: Donor alignment ───────────────────────────────────────────
    _section(f"Step 2 — Donor alignment  [method={cfg['alignment']['method']}]")
    alignment_result  = step_align(cfg, content_bgr, donor_bgr)
    donor_aligned_bgr = alignment_result.aligned_donor

    # ── Step 3: Object mask ───────────────────────────────────────────────
    _section(f"Step 3 — Object mask  [mask_type={cfg['ablation']['mask_type']}]")
    object_mask_uint8 = step_mask(cfg, content_bgr)

    # ── Step 4: Frequency decomposition ──────────────────────────────────
    _section(f"Step 4 — Frequency decomposition  [method={cfg['ablation']['decomposition']}]")
    content_decomp, donor_decomp = step_decompose(cfg, content_bgr, donor_aligned_bgr)

    # ── Step 5: Build mixed-frequency prior ───────────────────────────────
    _section(
        f"Step 5 — Build prior  "
        f"[α={cfg['prior']['alpha']}  β={cfg['prior']['beta']}  γ={cfg['prior']['gamma']}]"
    )
    prior_result = step_build_prior(
        cfg, content_bgr, donor_aligned_bgr, object_mask_uint8
    )

    # ── Step 6: Format conversion ─────────────────────────────────────────
    _section("Step 6 — Format conversion (numpy → PIL + tensor)")
    (
        content_pil,
        donor_aligned_pil,
        s_lf_pil,
        r_hf_pil,
        prior_pil,
        masked_input_pil,
        object_mask_tensor,
    ) = step_convert(
        cfg, content_bgr, donor_aligned_bgr,
        content_decomp, donor_decomp,
        prior_result, object_mask_uint8,
    )

    # ── Build meta ────────────────────────────────────────────────────────
    meta = build_meta(
        cfg, alignment_result,
        content_decomp, donor_decomp,
        prior_result, object_mask_uint8,
    )

    # ── Step 7: Save artifacts ────────────────────────────────────────────
    _section("Step 7 — Save artifacts")
    step_save(
        cfg,
        content_pil,
        donor_aligned_pil,
        s_lf_pil,
        r_hf_pil,
        prior_pil,
        masked_input_pil,
        object_mask_tensor,
        meta,
    )

    return cfg["paths"]["artifacts_dir"], meta


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _print_summary(artifacts_dir: str, meta: dict):
    """Print a human-readable summary after all steps complete."""
    coverage   = meta["mask_coverage_pct"]
    hf_std     = meta["donor_hf_std"]
    prior_std  = meta["prior_std"]

    print(f"\n{'═' * 60}")
    print(f"  Stage 1 complete")
    print(f"{'═' * 60}")
    print(f"  artifacts_dir    : {os.path.abspath(artifacts_dir)}/")
    print(f"")
    print(f"  alignment_method : {meta['alignment_method']}"
          + ("  [fallback]" if meta["alignment_fallback"] else ""))
    print(f"  mask_type        : {meta['mask_type']}")
    print(f"  coverage         : {coverage:.1f}%  "
          f"{'✓' if 5.0 < coverage < 95.0 else '⚠ check mask'}")
    print(f"  decomp_method    : {meta['decomp_method']}")
    print(f"  donor HF_std     : {hf_std:.2f}  "
          f"{'✓' if hf_std > 5 else '⚠ low — donor may be flat / blurry'}")
    print(f"")
    print(f"  prior  α={meta['alpha']}  β={meta['beta']}  γ={meta['gamma']} | "
          f"histogram_match={meta['histogram_match']}")
    print(f"  prior_std        : {prior_std:.2f}  "
          f"{'✓' if prior_std > 5 else '⚠ low — prior may be too smooth'}")
    print(f"  target_size      : {meta['target_size']}px")
    print(f"  timestamp        : {meta['timestamp'][:19]}")
    print(f"")
    print(f"  Key artifact for Stage 2:")
    print(f"    artifacts/masked_input_pil.png  — inpainting model input X₀")
    print(f"    artifacts/object_mask.pt        — mask tensor for KV gating + anchoring")
    print(f"")
    print(f"  Stage 2 is ready to run:")
    print(f"    python stage2_diffusion.py")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stage 1 — donor alignment, object segmentation, "
            "and mixed-frequency prior construction"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file"
    )

    # ── Image path overrides ───────────────────────────────────────────────
    p.add_argument(
        "--content", default=None,
        help="Override paths.content_image (the image whose structure is preserved)"
    )
    p.add_argument(
        "--donor", default=None,
        help="Override paths.donor_image (the image whose texture/attributes are transferred)"
    )
    p.add_argument(
        "--artifacts-dir", default=None,
        help="Override paths.artifacts_dir"
    )

    # ── SAM model overrides ────────────────────────────────────────────────
    p.add_argument(
        "--sam-checkpoint", default=None,
        help="Override paths.sam_checkpoint"
    )
    p.add_argument(
        "--sam-type", default=None,
        choices=["vit_h", "vit_l", "vit_b", "mobile"],
        help="Override paths.sam_model_type"
    )

    # ── Ablation / method overrides ────────────────────────────────────────
    p.add_argument(
        "--mask-type", default=None,
        choices=["sam", "none"],
        help="Override ablation.mask_type"
    )
    p.add_argument(
        "--decomp", default=None,
        choices=["gaussian", "fft", "none"],
        help="Override ablation.decomposition"
    )
    p.add_argument(
        "--align-method", default=None,
        choices=["resize", "tile", "match", "flow", "affine"],
        help="Override alignment.method"
    )

    # ── Prior coefficient overrides ────────────────────────────────────────
    p.add_argument(
        "--alpha", type=float, default=None,
        help="Override prior.alpha (content LF weight, default 0.6)"
    )
    p.add_argument(
        "--beta", type=float, default=None,
        help="Override prior.beta (donor LF weight, default 0.5)"
    )
    p.add_argument(
        "--gamma", type=float, default=None,
        help="Override prior.gamma (donor HF weight, default 0.8)"
    )

    # ── Image resolution override ──────────────────────────────────────────
    p.add_argument(
        "--target-size", type=int, default=None,
        choices=[512, 768],
        help="Override image.target_size (512 for sd-2-inpainting, 768 for sd-2-inpainting-768)"
    )

    # ── Utility ────────────────────────────────────────────────────────────
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved config and exit without running"
    )

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
        print(f"\n[stage1] FATAL — {e}")
        sys.exit(1)

    # Apply any CLI overrides on top of the yaml
    cfg = apply_cli_overrides(cfg, args)

    # Print resolved config (always — short and useful for debugging)
    print_config(cfg)

    # Dry run: show config and exit
    if args.dry_run:
        print("[stage1] --dry-run: exiting without running.\n")
        sys.exit(0)

    # ── Run pipeline ──────────────────────────────────────────────────────
    try:
        artifacts_dir, meta = run_stage1(cfg)
    except FileNotFoundError as e:
        print(f"\n[stage1] FATAL — {e}\n")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[stage1] FATAL — {e}\n")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n[stage1] FATAL — {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n[stage1] FATAL — unexpected error: {e}\n")
        traceback.print_exc()
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────
    _print_summary(artifacts_dir, meta)
    sys.exit(0)


if __name__ == "__main__":
    main()