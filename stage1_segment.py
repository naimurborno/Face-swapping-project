# stage1_segment.py
"""
Stage 1 — Face alignment, segmentation, and frequency decomposition.

Entry point for the first stage of the face swap pipeline.
Reads configs/default.yaml, runs the full Stage 1 pipeline, and saves
all artifacts to the artifacts/ directory for Stage 2 to consume.

Pipeline steps (in order):
    1. Load source + reference images from disk
    2. MediaPipe landmark detection on both images
    3. Affine alignment: warp reference → source pose
    4. Face mask: SAM / convex_hull / none  (ablation.mask_type)
    5. Frequency decomposition of aligned reference  (ablation.decomposition)
    6. Format conversion: numpy BGR → PIL RGB + mask tensor
    7. Save all artifacts to artifacts/

After this script completes:
    - Stage 2 can be run immediately (pipe loads diffusion model fresh)
    - SAM is fully unloaded from memory before this script exits
    - Artifacts contain everything Stage 2 needs — no images needed at runtime

Usage:
    # Default config:
    python stage1_segment.py

    # Custom config file:
    python stage1_segment.py --config configs/my_experiment.yaml

    # Override specific values without editing the yaml:
    python stage1_segment.py --mask-type convex_hull
    python stage1_segment.py --decomp fft
    python stage1_segment.py --source inputs/alice.jpg --reference inputs/bob.jpg

    # Dry run — prints resolved config and exits without running:
    python stage1_segment.py --dry-run

Output (all in artifacts/ by default):
    source_pil.png    — source image resized to target_size, RGB
    aligned_pil.png   — reference warped into source pose, resized
    lf_pil.png        — LF component of aligned reference, resized
    hf_pil.png        — HF component shifted to [0,255], resized
    face_mask.pt      — (1,1,S,S) float32 mask tensor for KVCache.face_mask
    meta.json         — all parameters + stats, read by Stage 2 at startup

Dependencies:
    pip install mediapipe opencv-python-headless torch pillow pyyaml
    pip install git+https://github.com/facebookresearch/segment-anything.git
    Download face_landmarker.task (see README)
    Download SAM checkpoint (see README)
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
    # Fallback: assume core modules are in the same directory
    sys.path.insert(0, _HERE)

try:
    from alignment import run_alignment
    from segmentation import load_sam_model, get_face_mask
    from decomposition import decompose, to_pil_inputs, mask_to_tensor
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
    (i.e. not None). This lets you run ablations without editing the yaml:
        python stage1_segment.py --mask-type convex_hull --decomp fft
    """
    if args.source is not None:
        cfg["paths"]["source_image"] = args.source
    if args.reference is not None:
        cfg["paths"]["reference_image"] = args.reference
    if args.artifacts_dir is not None:
        cfg["paths"]["artifacts_dir"] = args.artifacts_dir
    if args.landmark_model is not None:
        cfg["paths"]["landmark_model"] = args.landmark_model
    if args.sam_checkpoint is not None:
        cfg["paths"]["sam_checkpoint"] = args.sam_checkpoint
    if args.sam_type is not None:
        cfg["paths"]["sam_model_type"] = args.sam_type
    if args.mask_type is not None:
        cfg["ablation"]["mask_type"] = args.mask_type
    if args.decomp is not None:
        cfg["ablation"]["decomposition"] = args.decomp
    if args.target_size is not None:
        cfg["image"]["target_size"] = args.target_size
    return cfg


def print_config(cfg: dict):
    """Print the resolved config that will actually be used."""
    print("\n┌─ Resolved config ─────────────────────────────────────────")
    print(f"│  source_image   : {cfg['paths']['source_image']}")
    print(f"│  reference_image: {cfg['paths']['reference_image']}")
    print(f"│  artifacts_dir  : {cfg['paths']['artifacts_dir']}")
    print(f"│  landmark_model : {cfg['paths']['landmark_model']}")
    print(f"│  target_size    : {cfg['image']['target_size']}px")
    print(f"│")
    print(f"│  [ablation]")
    print(f"│  mask_type      : {cfg['ablation']['mask_type']}")
    print(f"│  decomposition  : {cfg['ablation']['decomposition']}")
    print(f"│")
    if cfg["ablation"]["mask_type"] == "sam":
        print(f"│  [SAM]")
        print(f"│  sam_checkpoint : {cfg['paths']['sam_checkpoint']}")
        print(f"│  sam_model_type : {cfg['paths']['sam_model_type']}")
        print(f"│  prompt_strategy: {cfg['stage1']['sam']['prompt_strategy']}")
        print(f"│  pred_iou_thresh: {cfg['stage1']['sam']['pred_iou_thresh']}")
        print(f"│")
    if cfg["ablation"]["decomposition"] == "gaussian":
        print(f"│  [Gaussian]")
        print(f"│  kernel         : {cfg['stage1']['gaussian']['kernel']}")
        print(f"│  sigma          : {cfg['stage1']['gaussian']['sigma']}")
    elif cfg["ablation"]["decomposition"] == "fft":
        print(f"│  [FFT]")
        print(f"│  cutoff_ratio   : {cfg['stage1']['fft']['cutoff_ratio']}")
    print(f"└───────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP RUNNERS
# Each step is a named function so failures have clear traceback origins.
# ══════════════════════════════════════════════════════════════════════════════

def step_load_images(cfg: dict):
    """
    Step 1: Load source and reference images from disk.

    Returns:
        source_bgr    : (H, W, 3) uint8 BGR
        reference_bgr : (H, W, 3) uint8 BGR
    """
    src_path = cfg["paths"]["source_image"]
    ref_path = cfg["paths"]["reference_image"]

    for label, path in [("source", src_path), ("reference", ref_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[stage1] {label} image not found: {path}\n"
                f"  Update paths.{label}_image in configs/default.yaml\n"
                f"  or pass --{'source' if label == 'source' else 'reference'} <path>"
            )

    source_bgr    = cv2.imread(src_path)
    reference_bgr = cv2.imread(ref_path)

    if source_bgr is None:
        raise ValueError(f"[stage1] cv2.imread failed for source: {src_path}")
    if reference_bgr is None:
        raise ValueError(f"[stage1] cv2.imread failed for reference: {ref_path}")

    print(f"[stage1] Source   : {src_path}  ({source_bgr.shape[1]}×{source_bgr.shape[0]}px)")
    print(f"[stage1] Reference: {ref_path}  ({reference_bgr.shape[1]}×{reference_bgr.shape[0]}px)")

    return source_bgr, reference_bgr


def step_align(cfg: dict, source_bgr: np.ndarray, reference_bgr: np.ndarray):
    """
    Step 2+3: MediaPipe detection on both images + affine warp.

    Returns:
        alignment_result : AlignmentResult (src_result, aligned_face, yaw_diff, ...)
    """
    mp_cfg         = cfg["stage1"]["mediapipe"]
    landmark_model = cfg["paths"]["landmark_model"]

    if not os.path.exists(landmark_model):
        raise FileNotFoundError(
            f"[stage1] MediaPipe model not found: {landmark_model}\n"
            "  Download with:\n"
            "  wget -O face_landmarker.task https://storage.googleapis.com/"
            "mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    alignment_result = run_alignment(
        source_bgr    = source_bgr,
        reference_bgr = reference_bgr,
        model_path    = landmark_model,
        min_face_detection_confidence = mp_cfg["min_face_detection_confidence"],
        min_face_presence_confidence  = mp_cfg["min_face_presence_confidence"],
        max_yaw_warning_deg           = mp_cfg["max_yaw_warning_deg"],
    )

    print(
        f"[stage1] Alignment done | "
        f"yaw_diff={alignment_result.yaw_diff:.1f}°  "
        f"{'✓ ok' if alignment_result.yaw_diff <= mp_cfg['max_yaw_warning_deg'] else '⚠ large — warp may degrade'}"
    )

    return alignment_result


def step_mask(cfg: dict, source_bgr: np.ndarray, src_lm) -> np.ndarray:
    """
    Step 4: Produce the face mask for the source image.

    mask_type is read from ablation.mask_type:
        "sam"         → load SAM, run with MediaPipe point prompt, delete SAM
        "convex_hull" → convex hull of 68 landmarks, dilated
        "none"        → full-image mask (global injection, no spatial gating)

    SAM is loaded and deleted inside this function so its VRAM is free
    before Stage 2 loads the diffusion model.

    Returns:
        face_mask_uint8 : (H, W) uint8, values 0/255
    """
    mask_type = cfg["ablation"]["mask_type"]
    sam_cfg   = cfg["stage1"]["sam"]
    ch_cfg    = cfg["stage1"]["convex_hull"]

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
                f"  To skip SAM entirely, use ablation.mask_type: convex_hull\n"
                f"  or pass --mask-type convex_hull"
            )

        predictor = load_sam_model(
            checkpoint_path = sam_checkpoint,
            model_type      = sam_model_type,
        )

    face_mask_uint8 = get_face_mask(
        image_bgr               = source_bgr,
        lm_result               = src_lm,
        mask_type               = mask_type,
        predictor               = predictor,
        prompt_strategy         = sam_cfg["prompt_strategy"],
        pred_iou_thresh         = sam_cfg["pred_iou_thresh"],
        stability_score_thresh  = sam_cfg["stability_score_thresh"],
        dilation_px             = 0,                      # SAM mask is already precise
        convex_hull_dilation_px = ch_cfg["dilation_px"],
    )

    # Free SAM from GPU memory immediately — Stage 2 needs that VRAM
    if predictor is not None:
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[stage1] SAM predictor deleted — VRAM freed for Stage 2")

    coverage = float(face_mask_uint8.astype(bool).mean() * 100)
    print(
        f"[stage1] Mask ready | "
        f"mask_type={mask_type} | "
        f"coverage={coverage:.1f}%  "
        f"{'✓' if 5.0 < coverage < 80.0 else '⚠ unusual coverage — check mask quality'}"
    )

    return face_mask_uint8


def step_decompose(cfg: dict, aligned_face: np.ndarray):
    """
    Step 5: Frequency decomposition of the aligned reference face.

    method is read from ablation.decomposition:
        "gaussian" → Gaussian blur split (default, no ringing)
        "fft"      → FFT rectangular mask split (ablation)
        "none"     → whole image used as both LF and HF (Phase 2 equivalent)

    Returns:
        decomp_result : DecomposeResult(LF, HF, method, lf_energy, hf_std)
    """
    method       = cfg["ablation"]["decomposition"]
    gauss_cfg    = cfg["stage1"]["gaussian"]
    fft_cfg      = cfg["stage1"]["fft"]

    decomp_result = decompose(
        aligned_bgr  = aligned_face,
        method       = method,
        kernel       = gauss_cfg["kernel"],
        sigma        = gauss_cfg["sigma"],
        cutoff_ratio = fft_cfg["cutoff_ratio"],
    )

    print(
        f"[stage1] Decomposition done | "
        f"method={method} | "
        f"LF_energy={decomp_result.lf_energy:.1f} | "
        f"HF_std={decomp_result.hf_std:.2f}  "
        f"{'✓' if decomp_result.hf_std > 10 else '⚠ low HF std — check image texture'}"
    )

    return decomp_result


def step_convert(cfg: dict, decomp_result, aligned_face: np.ndarray,
                 face_mask_uint8: np.ndarray, source_bgr: np.ndarray):
    """
    Step 6: Convert all numpy outputs to pipeline-ready formats.

    Returns:
        source_pil       : PIL RGB (target_size × target_size)
        aligned_pil      : PIL RGB (target_size × target_size)
        lf_pil           : PIL RGB (target_size × target_size)
        hf_pil           : PIL RGB (target_size × target_size)  HF shifted to [0,255]
        face_mask_tensor : (1, 1, target_size, target_size) float32 [0,1]
    """
    target_size = cfg["image"]["target_size"]

    aligned_pil, lf_pil, hf_pil = to_pil_inputs(
        result      = decomp_result,
        aligned_bgr = aligned_face,
        target_size = target_size,
    )

    face_mask_tensor = mask_to_tensor(face_mask_uint8, target_size=target_size)

    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    source_pil = Image.fromarray(source_rgb).resize(
        (target_size, target_size), Image.LANCZOS
    )

    print(
        f"[stage1] Conversion done | "
        f"PIL size={source_pil.size} | "
        f"mask tensor shape={tuple(face_mask_tensor.shape)} "
        f"range=[{face_mask_tensor.min():.2f}, {face_mask_tensor.max():.2f}]"
    )

    return source_pil, aligned_pil, lf_pil, hf_pil, face_mask_tensor


def step_save(cfg: dict, source_pil, aligned_pil, lf_pil, hf_pil,
              face_mask_tensor, meta: dict):
    """
    Step 7: Save all artifacts to the artifacts directory.

    Files written:
        source_pil.png    — source image (resized, RGB)
        aligned_pil.png   — aligned reference (resized, RGB)
        lf_pil.png        — LF component (resized, RGB)
        hf_pil.png        — HF component shifted [0,255] (resized, RGB)
        face_mask.pt      — (1,1,S,S) float32 tensor
        meta.json         — all metadata Stage 2 reads at startup
    """
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    saves = {
        "source_pil.png"  : ("pil",    source_pil),
        "aligned_pil.png" : ("pil",    aligned_pil),
        "lf_pil.png"      : ("pil",    lf_pil),
        "hf_pil.png"      : ("pil",    hf_pil),
        "face_mask.pt"    : ("tensor", face_mask_tensor),
        "meta.json"       : ("json",   meta),
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
        print(f"  {fname:<22}  {size_kb:7.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# META BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_meta(cfg: dict, yaw_diff: float, decomp_result,
               face_mask_uint8: np.ndarray) -> dict:
    """
    Build the meta.json dict that Stage 2 reads at startup.

    Contains every parameter and stat needed to:
        - Reproduce the run exactly (decomp method, mask type, kernel, etc.)
        - Verify artifact quality (hf_std, mask_coverage, yaw_diff)
        - Support the ablation runner (all ablation flags recorded)
    """
    coverage = float(face_mask_uint8.astype(bool).mean() * 100)

    return {
        # ── Image paths ──────────────────────────────────────────────────
        "source_path"       : os.path.abspath(cfg["paths"]["source_image"]),
        "reference_path"    : os.path.abspath(cfg["paths"]["reference_image"]),

        # ── Alignment ────────────────────────────────────────────────────
        "yaw_diff_deg"      : round(float(yaw_diff), 3),

        # ── Decomposition ─────────────────────────────────────────────────
        "decomp_method"     : cfg["ablation"]["decomposition"],
        "gauss_kernel"      : cfg["stage1"]["gaussian"]["kernel"],
        "gauss_sigma"       : cfg["stage1"]["gaussian"]["sigma"],
        "fft_cutoff"        : cfg["stage1"]["fft"]["cutoff_ratio"],
        "lf_energy"         : round(float(decomp_result.lf_energy), 3),
        "hf_std"            : round(float(decomp_result.hf_std),    3),

        # ── Mask ─────────────────────────────────────────────────────────
        "mask_type"         : cfg["ablation"]["mask_type"],
        "sam_model_type"    : cfg["paths"]["sam_model_type"],
        "prompt_strategy"   : cfg["stage1"]["sam"]["prompt_strategy"],
        "mask_coverage_pct" : round(coverage, 2),

        # ── Output format ─────────────────────────────────────────────────
        "target_size"       : cfg["image"]["target_size"],

        # ── Ablation flags (all of them — Stage 2 and ablation_runner read these) ──
        "ablation" : {
            "decomposition"  : cfg["ablation"]["decomposition"],
            "mask_type"      : cfg["ablation"]["mask_type"],
            "depth_routing"  : cfg["ablation"]["depth_routing"],
            "temporal_anneal": cfg["ablation"]["temporal_anneal"],
            "mode"           : cfg["ablation"]["mode"],
        },

        # ── Provenance ────────────────────────────────────────────────────
        "stage1_script"     : os.path.abspath(__file__),
        "timestamp"         : datetime.datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_stage1(cfg: dict):
    """
    Execute all 7 Stage 1 steps in order.

    All side effects (printing, file I/O) are in the step functions.
    This function is the single sequence: load → align → mask → decompose
    → convert → build meta → save.

    Args:
        cfg : Fully resolved config dict (yaml + CLI overrides applied).

    Returns:
        artifacts_dir : Path to the directory where artifacts were saved.
        meta          : The meta dict that was written to meta.json.
    """

    # ── Step 1: Load images ───────────────────────────────────────────────
    _section("Step 1 — Load images")
    source_bgr, reference_bgr = step_load_images(cfg)

    # ── Step 2+3: Landmark detection + affine alignment ───────────────────
    _section("Step 2+3 — Landmark detection + affine alignment")
    alignment_result = step_align(cfg, source_bgr, reference_bgr)
    src_lm       = alignment_result.src_result
    aligned_face = alignment_result.aligned_face   # (H, W, 3) uint8 BGR
    yaw_diff     = alignment_result.yaw_diff

    # ── Step 4: Face mask ─────────────────────────────────────────────────
    _section(f"Step 4 — Face mask  [mask_type={cfg['ablation']['mask_type']}]")
    face_mask_uint8 = step_mask(cfg, source_bgr, src_lm)

    # ── Step 5: Frequency decomposition ──────────────────────────────────
    _section(f"Step 5 — Frequency decomposition  [method={cfg['ablation']['decomposition']}]")
    decomp_result = step_decompose(cfg, aligned_face)

    # ── Step 6: Format conversion ─────────────────────────────────────────
    _section("Step 6 — Format conversion (numpy → PIL + tensor)")
    source_pil, aligned_pil, lf_pil, hf_pil, face_mask_tensor = step_convert(
        cfg, decomp_result, aligned_face, face_mask_uint8, source_bgr
    )

    # ── Build meta ────────────────────────────────────────────────────────
    meta = build_meta(cfg, yaw_diff, decomp_result, face_mask_uint8)

    # ── Step 7: Save artifacts ────────────────────────────────────────────
    _section("Step 7 — Save artifacts")
    step_save(cfg, source_pil, aligned_pil, lf_pil, hf_pil, face_mask_tensor, meta)

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
    print(f"\n{'═' * 60}")
    print(f"  Stage 1 complete")
    print(f"{'═' * 60}")
    print(f"  artifacts_dir  : {os.path.abspath(artifacts_dir)}/")
    print(f"  yaw_diff       : {meta['yaw_diff_deg']:.1f}°"
          f"  {'✓ ok' if meta['yaw_diff_deg'] <= 35 else '⚠ large'}")
    print(f"  mask_type      : {meta['mask_type']}")
    print(f"  coverage       : {meta['mask_coverage_pct']:.1f}%")
    print(f"  decomp_method  : {meta['decomp_method']}")
    print(f"  HF_std         : {meta['hf_std']:.2f}"
          f"  {'✓' if meta['hf_std'] > 10 else '⚠ low — check image texture'}")
    print(f"  target_size    : {meta['target_size']}px")
    print(f"  timestamp      : {meta['timestamp'][:19]}")
    print(f"\n  Stage 2 is ready to run:")
    print(f"    python stage2_diffusion.py")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1 — face alignment, segmentation, decomposition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file"
    )

    # ── Image overrides ────────────────────────────────────────────────────
    p.add_argument("--source",    default=None,
                   help="Override paths.source_image")
    p.add_argument("--reference", default=None,
                   help="Override paths.reference_image")
    p.add_argument("--artifacts-dir", default=None,
                   help="Override paths.artifacts_dir")

    # ── Model overrides ────────────────────────────────────────────────────
    p.add_argument("--landmark-model", default=None,
                   help="Override paths.landmark_model")
    p.add_argument("--sam-checkpoint", default=None,
                   help="Override paths.sam_checkpoint")
    p.add_argument("--sam-type", default=None,
                   choices=["vit_h", "vit_l", "vit_b", "mobile"],
                   help="Override paths.sam_model_type")

    # ── Ablation overrides ─────────────────────────────────────────────────
    p.add_argument("--mask-type", default=None,
                   choices=["sam", "convex_hull", "none"],
                   help="Override ablation.mask_type")
    p.add_argument("--decomp", default=None,
                   choices=["gaussian", "fft", "none"],
                   help="Override ablation.decomposition")
    p.add_argument("--target-size", type=int, default=None,
                   choices=[512, 768],
                   help="Override image.target_size")

    # ── Utility ────────────────────────────────────────────────────────────
    p.add_argument("--dry-run", action="store_true",
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
        print(f"\n[stage1] FATAL — {e}")
        sys.exit(1)

    # Apply any CLI overrides
    cfg = apply_cli_overrides(cfg, args)

    # Print resolved config always (short, useful)
    print_config(cfg)

    # Dry run: just show config and exit
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
    except Exception as e:
        print(f"\n[stage1] FATAL — unexpected error: {e}\n")
        traceback.print_exc()
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────
    _print_summary(artifacts_dir, meta)
    sys.exit(0)


if __name__ == "__main__":
    main()