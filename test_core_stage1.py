"""
test_stage1_real.py
===================
Real-image, real-SAM end-to-end Stage 1 test.

Takes two face photos, runs the FULL Stage 1 pipeline:
    1. MediaPipe landmark detection + affine alignment
    2. SAM segmentation (or convex_hull / none as fallback)
    3. Gaussian / FFT / none frequency decomposition
    4. Saves ALL artifacts to artifacts/
    5. Opens a multi-panel visualization window

Usage (from project root):
    # With real SAM checkpoint:
    python test_stage1_real.py --source inputs/source.jpg --reference inputs/reference.jpg

    # With a different SAM variant:
    python test_stage1_real.py \\
        --source inputs/source.jpg \\
        --reference inputs/reference.jpg \\
        --sam-checkpoint checkpoints/mobile_sam.pt \\
        --sam-type mobile

    # Skip SAM, use convex hull mask (no checkpoint needed):
    python test_stage1_real.py \\
        --source inputs/source.jpg \\
        --reference inputs/reference.jpg \\
        --mask-type convex_hull

    # Test all three decomposition methods:
    python test_stage1_real.py \\
        --source inputs/source.jpg \\
        --reference inputs/reference.jpg \\
        --decomp fft

    # Save visualization to file instead of displaying:
    python test_stage1_real.py \\
        --source inputs/source.jpg \\
        --reference inputs/reference.jpg \\
        --save-viz artifacts/stage1_viz.png

Project layout:
    project/
    ├── configs/default.yaml
    ├── core/
    │   ├── alignment.py
    │   ├── segmentation.py
    │   ├── decomposition.py
    │   └── mask_utils.py
    ├── inputs/
    │   ├── source.jpg
    │   └── reference.jpg
    ├── artifacts/          ← created automatically
    └── test_stage1_real.py ← this file

Requirements:
    pip install mediapipe opencv-python numpy torch pillow matplotlib pyyaml
    pip install git+https://github.com/facebookresearch/segment-anything.git  # for SAM1
    pip install git+https://github.com/ChaoningZhang/MobileSAM.git            # for MobileSAM
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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")
sys.path.insert(0, _CORE if os.path.isdir(_CORE) else _HERE)

try:
    from alignment import (
        FaceLandmarkDetector,
        run_alignment,
        build_convex_hull_mask,
    )
    from segmentation import load_sam_model, get_face_mask
    from decomposition import decompose, to_pil_inputs, mask_to_tensor
    from mask_utils import mask_to_token_mask
except ImportError as e:
    print(f"\n[FATAL] Could not import core modules: {e}")
    print("  Make sure you are running from the project root and core/ exists.")
    traceback.print_exc()
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1 real-image end-to-end test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Images ─────────────────────────────────────────────────────────────
    p.add_argument("--source",    required=True,
                   help="Path to source image (pose/background to keep)")
    p.add_argument("--reference", required=True,
                   help="Path to reference image (identity to transfer)")

    # ── MediaPipe ──────────────────────────────────────────────────────────
    p.add_argument("--landmark-model", default="face_landmarker.task",
                   help="Path to face_landmarker.task file")

    # ── SAM ────────────────────────────────────────────────────────────────
    p.add_argument("--sam-checkpoint", default=None,
                   help="Path to SAM checkpoint (.pth file). "
                        "Not needed when --mask-type=convex_hull or none.")
    p.add_argument("--sam-type", default="vit_h",
                   choices=["vit_h", "vit_l", "vit_b", "mobile"],
                   help="SAM model variant (must match checkpoint)")
    p.add_argument("--sam-device", default=None,
                   help="Device for SAM: 'cuda' or 'cpu'. Auto-detects if not set.")
    p.add_argument("--prompt-strategy", default="nose_tip",
                   choices=["nose_tip", "eye_center", "bbox"],
                   help="SAM point prompt strategy")
    p.add_argument("--pred-iou-thresh", type=float, default=0.88,
                   help="SAM predicted IoU threshold")
    p.add_argument("--stability-thresh", type=float, default=0.95,
                   help="SAM stability score threshold")
    p.add_argument("--sam-dilation", type=int, default=5,
                   help="Post-process dilation of SAM mask in pixels")

    # ── Mask type (ablation) ────────────────────────────────────────────────
    p.add_argument("--mask-type", default="sam",
                   choices=["sam", "convex_hull", "none"],
                   help="Mask type: sam (full pipeline) | convex_hull (no SAM needed) | none")
    p.add_argument("--hull-dilation", type=int, default=10,
                   help="Dilation for convex hull mask (pixels)")

    # ── Decomposition (ablation) ────────────────────────────────────────────
    p.add_argument("--decomp", default="gaussian",
                   choices=["gaussian", "fft", "none"],
                   help="Decomposition method")
    p.add_argument("--gauss-kernel", type=int, default=31,
                   help="Gaussian kernel size (odd number)")
    p.add_argument("--gauss-sigma", type=float, default=5.0,
                   help="Gaussian sigma")
    p.add_argument("--fft-cutoff", type=float, default=0.1,
                   help="FFT cutoff ratio (fraction of frequency radius)")

    # ── Output ─────────────────────────────────────────────────────────────
    p.add_argument("--target-size", type=int, default=512,
                   help="Resize all outputs to this square size (512 or 768)")
    p.add_argument("--artifacts-dir", default="artifacts",
                   help="Directory where artifacts are saved")
    p.add_argument("--save-viz", default=None,
                   help="Save visualization to this path instead of displaying")
    p.add_argument("--no-display", action="store_true",
                   help="Skip interactive display (use with --save-viz)")
    p.add_argument("--yaw-warn", type=float, default=35.0,
                   help="Warn if yaw difference exceeds this (degrees)")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path: str, label: str) -> np.ndarray:
    """Load a BGR image from disk with a clear error message."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[test] {label} image not found: {path}\n"
            f"  Make sure the file exists or use a different --{label.lower()} path."
        )
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"[test] cv2.imread returned None for: {path}")
    print(f"[test] Loaded {label}: {path}  ({img.shape[1]}×{img.shape[0]}px)")
    return img


def _mask_overlay(bgr_img: np.ndarray, mask_uint8: np.ndarray,
                  color_bgr=(0, 220, 80), alpha=0.40) -> np.ndarray:
    """Draw a semi-transparent coloured overlay on the masked region."""
    overlay = bgr_img.copy().astype(np.float32)
    color   = np.array(color_bgr, dtype=np.float32)
    face    = mask_uint8 > 127

    overlay[face] = (
        overlay[face] * (1 - alpha) + color * alpha
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _bgr_to_rgb_pil(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 numpy to RGB for matplotlib."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _hf_display(hf_float32: np.ndarray) -> np.ndarray:
    """Shift HF [-255,255] → [0,255] for display."""
    return _clip_uint8(hf_float32 + 128.0)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_stage1(args) -> dict:
    """
    Full Stage 1 pipeline with real images and (optionally) real SAM.

    Returns artifacts dict:
        source_pil, aligned_pil, lf_pil, hf_pil  — PIL RGB images
        face_mask                                  — (1,1,S,S) float32 tensor
        meta                                       — metadata dict
        _debug                                     — raw intermediate arrays for viz
    """

    # ── 1. Load images ──────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Stage 1 — Step 1: Load images")
    print("═" * 60)
    source_bgr    = load_image(args.source,    "Source")
    reference_bgr = load_image(args.reference, "Reference")

    # ── 2. MediaPipe landmark detection + affine alignment ──────────────────
    print("\n" + "═" * 60)
    print("  Stage 1 — Step 2: Landmark detection + alignment")
    print("═" * 60)

    if not os.path.exists(args.landmark_model):
        raise FileNotFoundError(
            f"[test] face_landmarker.task not found: {args.landmark_model}\n"
            "  Download with:\n"
            "  wget -O face_landmarker.task https://storage.googleapis.com/"
            "mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    alignment_result = run_alignment(
        source_bgr    = source_bgr,
        reference_bgr = reference_bgr,
        model_path    = args.landmark_model,
        max_yaw_warning_deg = args.yaw_warn,
    )

    src_lm       = alignment_result.src_result
    aligned_face = alignment_result.aligned_face
    yaw_diff     = alignment_result.yaw_diff

    print(f"\n[test] ✓ Alignment done | yaw_diff={yaw_diff:.1f}°")

    # ── 3. SAM / mask ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  Stage 1 — Step 3: Face mask (mask_type='{args.mask_type}')")
    print("═" * 60)

    predictor = None
    if args.mask_type == "sam":
        if args.sam_checkpoint is None:
            print(
                "\n[test] WARNING: --mask-type=sam but --sam-checkpoint not provided.\n"
                "  Falling back to convex_hull. Pass --sam-checkpoint <path> to use SAM.\n"
            )
            args.mask_type = "convex_hull"
        elif not os.path.exists(args.sam_checkpoint):
            raise FileNotFoundError(
                f"[test] SAM checkpoint not found: {args.sam_checkpoint}\n"
                "  Download from https://github.com/facebookresearch/segment-anything"
                "#model-checkpoints"
            )
        else:
            print(f"[test] Loading SAM model_type='{args.sam_type}' from {args.sam_checkpoint}")
            predictor = load_sam_model(
                checkpoint_path = args.sam_checkpoint,
                model_type      = args.sam_type,
                device          = args.sam_device,
            )

    face_mask_uint8 = get_face_mask(
        image_bgr               = source_bgr,
        lm_result               = src_lm,
        mask_type               = args.mask_type,
        predictor               = predictor,
        prompt_strategy         = args.prompt_strategy,
        pred_iou_thresh         = args.pred_iou_thresh,
        stability_score_thresh  = args.stability_thresh,
        dilation_px             = args.sam_dilation,
        convex_hull_dilation_px = args.hull_dilation,
    )

    coverage = float(face_mask_uint8.astype(bool).mean() * 100)
    print(f"\n[test] ✓ Mask ready | mask_type={args.mask_type} | coverage={coverage:.1f}%")

    # Free SAM VRAM before diffusion stage
    if predictor is not None:
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[test] SAM predictor deleted — VRAM freed")

    # ── 4. Frequency decomposition ───────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  Stage 1 — Step 4: Decomposition (method='{args.decomp}')")
    print("═" * 60)

    decomp_result = decompose(
        aligned_bgr  = aligned_face,
        method       = args.decomp,
        kernel       = args.gauss_kernel,
        sigma        = args.gauss_sigma,
        cutoff_ratio = args.fft_cutoff,
    )

    print(f"\n[test] ✓ Decomposition done")
    print(f"  LF energy : {decomp_result.lf_energy:.2f}")
    print(f"  HF std    : {decomp_result.hf_std:.2f}"
          + ("  ✓" if decomp_result.hf_std > 10 else "  ⚠ low — check image texture"))

    # ── 5. Convert to PIL + mask tensor ─────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Stage 1 — Step 5: PIL conversion + mask tensor")
    print("═" * 60)

    aligned_pil, lf_pil, hf_pil = to_pil_inputs(
        result      = decomp_result,
        aligned_bgr = aligned_face,
        target_size = args.target_size,
    )

    face_mask_tensor = mask_to_tensor(face_mask_uint8, target_size=args.target_size)

    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    source_pil = Image.fromarray(source_rgb).resize(
        (args.target_size, args.target_size), Image.LANCZOS
    )

    print(f"[test] PIL sizes: source={source_pil.size} aligned={aligned_pil.size} "
          f"lf={lf_pil.size} hf={hf_pil.size}")
    print(f"[test] Mask tensor: shape={tuple(face_mask_tensor.shape)} "
          f"range=[{face_mask_tensor.min():.3f}, {face_mask_tensor.max():.3f}]")

    # ── 6. Build artifact bundle ─────────────────────────────────────────────
    meta = {
        "yaw_diff_deg"      : round(float(yaw_diff), 3),
        "decomp_method"     : args.decomp,
        "gauss_kernel"      : args.gauss_kernel,
        "gauss_sigma"       : args.gauss_sigma,
        "fft_cutoff"        : args.fft_cutoff,
        "mask_type"         : args.mask_type,
        "sam_type"          : args.sam_type,
        "prompt_strategy"   : args.prompt_strategy,
        "target_size"       : args.target_size,
        "hf_std"            : round(float(decomp_result.hf_std), 3),
        "lf_energy"         : round(float(decomp_result.lf_energy), 3),
        "mask_coverage_pct" : round(coverage, 2),
        "source_path"       : os.path.abspath(args.source),
        "reference_path"    : os.path.abspath(args.reference),
        "timestamp"         : datetime.datetime.now().isoformat(),
    }

    artifacts = {
        "source_pil"  : source_pil,
        "aligned_pil" : aligned_pil,
        "lf_pil"      : lf_pil,
        "hf_pil"      : hf_pil,
        "face_mask"   : face_mask_tensor,
        "meta"        : meta,
        # Raw intermediates for visualization (not saved to disk)
        "_debug": {
            "source_bgr"       : source_bgr,
            "reference_bgr"    : reference_bgr,
            "aligned_face_bgr" : aligned_face,
            "face_mask_uint8"  : face_mask_uint8,
            "LF_float"         : decomp_result.LF,
            "HF_float"         : decomp_result.HF,
            "src_lm"           : src_lm,
            "yaw_diff"         : yaw_diff,
        },
    }

    return artifacts


# ══════════════════════════════════════════════════════════════════════════════
# SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(artifacts: dict, artifacts_dir: str):
    """Save all artifacts to disk exactly as stage1_segment.py will."""
    os.makedirs(artifacts_dir, exist_ok=True)

    print("\n" + "═" * 60)
    print(f"  Saving artifacts → {os.path.abspath(artifacts_dir)}/")
    print("═" * 60)

    saves = {
        "source_pil.png"  : ("pil",    artifacts["source_pil"]),
        "aligned_pil.png" : ("pil",    artifacts["aligned_pil"]),
        "lf_pil.png"      : ("pil",    artifacts["lf_pil"]),
        "hf_pil.png"      : ("pil",    artifacts["hf_pil"]),
        "face_mask.pt"    : ("tensor", artifacts["face_mask"]),
        "meta.json"       : ("json",   artifacts["meta"]),
    }

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

    print(f"\n[test] ✓ All artifacts saved")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION  (2 rows × 5 columns)
#
#  Row 1 — Spatial overview:
#    Source | Reference | Aligned Ref | Mask overlay | Token mask (64×64)
#
#  Row 2 — Frequency decomposition:
#    Aligned Ref | LF (structure) | HF (texture) | HF histogram | Meta table
# ══════════════════════════════════════════════════════════════════════════════

def visualize(artifacts: dict, args, save_path: str = None):
    """
    Draw a 2-row, 5-column visualization panel.

    Row 1: Spatial overview of what Stage 1 produced.
    Row 2: Frequency decomposition quality check.

    Args:
        artifacts : Output of run_stage1()
        args      : CLI args (for titles)
        save_path : If not None, save figure to this path instead of showing.
    """
    dbg = artifacts["_debug"]
    meta = artifacts["meta"]

    # ── Prepare display images ─────────────────────────────────────────────
    source_rgb    = _bgr_to_rgb_pil(dbg["source_bgr"])
    reference_rgb = _bgr_to_rgb_pil(dbg["reference_bgr"])
    aligned_rgb   = _bgr_to_rgb_pil(dbg["aligned_face_bgr"])
    mask_u8       = dbg["face_mask_uint8"]
    LF            = dbg["LF_float"]
    HF            = dbg["HF_float"]

    # Mask overlay on source
    mask_overlay_rgb = _bgr_to_rgb_pil(
        _mask_overlay(dbg["source_bgr"], mask_u8)
    )

    # LF display — float [0,255] → uint8
    lf_rgb = cv2.cvtColor(_clip_uint8(LF), cv2.COLOR_BGR2RGB)

    # HF display — shift [-255,255] → [0,255]
    hf_display = _hf_display(HF)
    hf_rgb     = cv2.cvtColor(hf_display, cv2.COLOR_BGR2RGB)

    # Token mask at 64×64 (SD2.1-base shallow layer resolution)
    face_mask_t  = artifacts["face_mask"]
    token_mask64 = mask_to_token_mask(face_mask_t, spatial_size=64)
    token_img64  = token_mask64.reshape(64, 64).numpy().astype(np.float32)
    token_mask16 = mask_to_token_mask(face_mask_t, spatial_size=16)
    token_img16  = token_mask16.reshape(16, 16).numpy().astype(np.float32)
    token_mask8  = mask_to_token_mask(face_mask_t, spatial_size=8)
    token_img8   = token_mask8.reshape(8, 8).numpy().astype(np.float32)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor("#1a1a2e")

    TITLE_COLOR  = "#e0e0f0"
    LABEL_COLOR  = "#a0a0c0"
    ACCENT_GREEN = "#4caf50"
    ACCENT_AMBER = "#ffb300"

    gs = fig.add_gridspec(
        2, 5,
        hspace=0.35, wspace=0.18,
        left=0.04, right=0.97, top=0.91, bottom=0.05
    )

    # ── Main title ─────────────────────────────────────────────────────────
    yaw_color = ACCENT_GREEN if meta["yaw_diff_deg"] <= args.yaw_warn else ACCENT_AMBER
    fig.suptitle(
        f"Stage 1 — Alignment + Segmentation + Decomposition\n"
        f"mask={meta['mask_type']}  |  decomp={meta['decomp_method']}  |  "
        f"yaw_diff={meta['yaw_diff_deg']:.1f}°  |  "
        f"coverage={meta['mask_coverage_pct']:.1f}%  |  "
        f"HF_std={meta['hf_std']:.2f}",
        fontsize=13, color=TITLE_COLOR, fontweight="bold", y=0.97
    )

    # ─────────────────────────────────────────────────────────────────────
    # ROW 1 — Spatial overview
    # ─────────────────────────────────────────────────────────────────────

    def _ax(row, col, title, img_rgb, cmap=None):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#0d0d1a")
        if cmap:
            ax.imshow(img_rgb, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(img_rgb)
        ax.set_title(title, color=TITLE_COLOR, fontsize=10, pad=4)
        ax.axis("off")
        return ax

    # Col 0: Source
    ax00 = _ax(0, 0, "Source\n(pose + background to keep)", source_rgb)

    # Col 1: Reference
    ax01 = _ax(0, 1, "Reference\n(identity to transfer)", reference_rgb)

    # Col 2: Aligned reference
    ax02 = _ax(0, 2, f"Aligned Reference\n(warped into source pose)", aligned_rgb)

    # Annotate yaw
    yaw_txt = f"yaw Δ ≈ {meta['yaw_diff_deg']:.1f}°"
    ax02.text(0.02, 0.02, yaw_txt, transform=ax02.transAxes,
              color=yaw_color, fontsize=9, va="bottom",
              bbox=dict(boxstyle="round,pad=0.2", fc="#0d0d1a", alpha=0.7))

    # Col 3: Mask overlay
    ax03 = _ax(0, 3,
               f"Face Mask Overlay\n(mask_type={meta['mask_type']}  cov={meta['mask_coverage_pct']:.1f}%)",
               mask_overlay_rgb)

    # Col 4: Token masks at three resolutions
    ax04 = fig.add_subplot(gs[0, 4])
    ax04.set_facecolor("#0d0d1a")
    ax04.set_title("Token Masks\n(64×64 / 16×16 / 8×8 layers)", color=TITLE_COLOR, fontsize=10, pad=4)
    ax04.axis("off")

    # Stack the three token mask images with labels
    sub_gs = gs[0, 4].subgridspec(1, 3, wspace=0.08)
    for sub_col, (timg, label, true_pct) in enumerate([
        (token_img64, "64×64\nshallow", float(token_mask64.float().mean()*100)),
        (token_img16, "16×16\ndeep",    float(token_mask16.float().mean()*100)),
        (token_img8,  "8×8\ndeep",      float(token_mask8.float().mean()*100)),
    ]):
        sub_ax = fig.add_subplot(sub_gs[0, sub_col])
        sub_ax.imshow(timg, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
        sub_ax.set_title(f"{label}\n{true_pct:.0f}% face", color=LABEL_COLOR, fontsize=7.5)
        sub_ax.axis("off")
        sub_ax.set_facecolor("#0d0d1a")

    # ─────────────────────────────────────────────────────────────────────
    # ROW 2 — Decomposition quality
    # ─────────────────────────────────────────────────────────────────────

    # Col 0: Aligned reference (again — reference point for LF/HF comparison)
    ax10 = _ax(1, 0, "Aligned Reference\n(input to decomposition)", aligned_rgb)

    # Col 1: LF component
    lf_title = (f"LF Component\n(structure / skin tone | "
                f"method={meta['decomp_method']})")
    ax11 = _ax(1, 1, lf_title, lf_rgb)

    # Col 2: HF component (shifted to [0,255])
    hf_std_str = f"HF std={meta['hf_std']:.1f}"
    hf_color   = ACCENT_GREEN if meta["hf_std"] > 10 else ACCENT_AMBER
    ax12 = _ax(1, 2,
               f"HF Component\n(texture detail, shifted +128 | {hf_std_str})",
               hf_rgb)
    ax12.text(0.02, 0.02, hf_std_str, transform=ax12.transAxes,
              color=hf_color, fontsize=9, va="bottom",
              bbox=dict(boxstyle="round,pad=0.2", fc="#0d0d1a", alpha=0.7))

    # Col 3: HF histogram (inside face mask only)
    ax13 = fig.add_subplot(gs[1, 3])
    ax13.set_facecolor("#0d0d1a")
    ax13.set_title("HF Histogram\n(inside face mask — expect ~N(0,σ))",
                   color=TITLE_COLOR, fontsize=10, pad=4)

    # Resize mask to match HF resolution
    H_hf, W_hf = HF.shape[:2]
    mask_resized = cv2.resize(mask_u8, (W_hf, H_hf), interpolation=cv2.INTER_LINEAR)
    face_region  = mask_resized > 127

    hf_vals_inside  = HF[face_region].ravel()
    hf_vals_outside = HF[~face_region].ravel()

    if len(hf_vals_inside) > 0:
        ax13.hist(hf_vals_inside,  bins=80, range=(-80, 80),
                  color="#4caf50", alpha=0.75, label="inside face",  density=True)
    if len(hf_vals_outside) > 0:
        ax13.hist(hf_vals_outside, bins=80, range=(-80, 80),
                  color="#2196f3", alpha=0.50, label="outside face", density=True)

    ax13.axvline(0, color="#ffffff", linewidth=0.8, linestyle="--")
    ax13.set_xlabel("HF pixel value", color=LABEL_COLOR, fontsize=8)
    ax13.set_ylabel("density", color=LABEL_COLOR, fontsize=8)
    ax13.tick_params(colors=LABEL_COLOR, labelsize=7)
    ax13.spines[["top", "right"]].set_visible(False)
    for spine in ax13.spines.values():
        spine.set_color("#444466")
    ax13.legend(fontsize=7, facecolor="#1a1a2e", labelcolor=LABEL_COLOR)

    # Ideal: mean near 0, std > 10
    mean_in = float(hf_vals_inside.mean()) if len(hf_vals_inside) > 0 else 0.0
    std_in  = float(hf_vals_inside.std())  if len(hf_vals_inside) > 0 else 0.0
    ax13.text(0.97, 0.95,
              f"μ={mean_in:.1f}\nσ={std_in:.1f}",
              transform=ax13.transAxes, ha="right", va="top",
              color=ACCENT_GREEN if abs(mean_in) < 5 and std_in > 10 else ACCENT_AMBER,
              fontsize=8,
              bbox=dict(boxstyle="round,pad=0.3", fc="#0d0d1a", alpha=0.8))

    # Col 4: Meta summary table
    ax14 = fig.add_subplot(gs[1, 4])
    ax14.set_facecolor("#0d0d1a")
    ax14.axis("off")
    ax14.set_title("Artifact Metadata", color=TITLE_COLOR, fontsize=10, pad=4)

    display_meta = [
        ("source",         os.path.basename(meta["source_path"])),
        ("reference",      os.path.basename(meta["reference_path"])),
        ("target_size",    f"{meta['target_size']}px"),
        ("yaw_diff",       f"{meta['yaw_diff_deg']:.1f}°"),
        ("mask_type",      meta["mask_type"]),
        ("coverage",       f"{meta['mask_coverage_pct']:.1f}%"),
        ("decomp_method",  meta["decomp_method"]),
        ("gauss_kernel",   str(meta["gauss_kernel"])),
        ("gauss_sigma",    str(meta["gauss_sigma"])),
        ("LF_energy",      f"{meta['lf_energy']:.1f}"),
        ("HF_std",         f"{meta['hf_std']:.2f}"),
        ("timestamp",      meta["timestamp"][:19]),
    ]

    y = 0.96
    dy = 0.074
    for key, val in display_meta:
        val_color = (ACCENT_GREEN
                     if (key == "HF_std" and float(val) > 10)
                     or (key == "yaw_diff" and float(val.rstrip("°")) <= args.yaw_warn)
                     else (ACCENT_AMBER
                           if key in ("HF_std", "yaw_diff") else LABEL_COLOR))
        ax14.text(0.02, y, key,        transform=ax14.transAxes,
                  color="#8888aa", fontsize=8, va="top", fontfamily="monospace")
        ax14.text(0.48, y, val,        transform=ax14.transAxes,
                  color=val_color, fontsize=8, va="top", fontfamily="monospace",
                  fontweight="bold")
        y -= dy

    # ── Final touches ───────────────────────────────────────────────────────
    # Add row labels on left edge
    fig.text(0.005, 0.73, "SPATIAL\nOVERVIEW",
             color=LABEL_COLOR, fontsize=8, va="center", ha="left",
             rotation=90, fontweight="bold")
    fig.text(0.005, 0.25, "FREQUENCY\nDECOMP",
             color=LABEL_COLOR, fontsize=8, va="center", ha="left",
             rotation=90, fontweight="bold")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n[test] ✓ Visualization saved → {save_path}")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION  (quick sanity checks before returning)
# ══════════════════════════════════════════════════════════════════════════════

def verify(artifacts: dict, target_size: int):
    """Quick sanity checks — prints ✓ or ✗ for each."""
    print("\n" + "═" * 60)
    print("  Verification")
    print("═" * 60)

    ok   = True
    meta = artifacts["meta"]

    def chk(label, cond, hint=""):
        nonlocal ok
        if cond:
            print(f"  ✓  {label}")
        else:
            print(f"  ✗  {label}  →  {hint}")
            ok = False

    S = target_size

    for name in ["source_pil", "aligned_pil", "lf_pil", "hf_pil"]:
        img = artifacts[name]
        chk(f"{name}: PIL RGB {S}×{S}",
            isinstance(img, Image.Image) and img.mode == "RGB" and img.size == (S, S),
            f"got mode={getattr(img,'mode','?')} size={getattr(img,'size','?')}")

        arr = np.array(img)
        chk(f"{name}: pixel range [0,255]",
            int(arr.min()) >= 0 and int(arr.max()) <= 255,
            f"min={arr.min()} max={arr.max()}")

    mask = artifacts["face_mask"]
    chk("face_mask shape (1,1,S,S)",
        mask.shape == (1, 1, S, S), f"got {tuple(mask.shape)}")
    chk("face_mask dtype float32",
        mask.dtype == torch.float32, f"got {mask.dtype}")
    chk("face_mask range [0,1]",
        float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0,
        f"min={mask.min():.3f} max={mask.max():.3f}")
    chk("face_mask has face tokens (>0.5)",
        bool((mask > 0.5).any()), "all zeros — mask is empty")
    chk("face_mask has background tokens (<0.5)",
        bool((mask < 0.5).any()), "all ones — mask covers entire image")

    chk("HF_std > 10 (textured face)",
        meta["hf_std"] > 10.0, f"got {meta['hf_std']:.2f} — may be over-blurred")
    chk("yaw_diff < 35° (alignment reliable)",
        meta["yaw_diff_deg"] < 35.0, f"got {meta['yaw_diff_deg']:.1f}°")
    chk("mask_coverage in (5%, 80%)",
        5.0 < meta["mask_coverage_pct"] < 80.0,
        f"got {meta['mask_coverage_pct']:.1f}%")

    # Token mask checks at each U-Net spatial resolution
    for spatial in [64, 16, 8]:
        tm = mask_to_token_mask(mask, spatial_size=spatial)
        chk(f"token_mask spatial={spatial}: has face tokens",
            tm.any().item(), "all False")
        chk(f"token_mask spatial={spatial}: has background",
            (~tm).any().item(), "all True — no spatial gating")

    print()
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("\n" + "═" * 60)
    print("  Stage 1 Real-Image Test")
    print("═" * 60)
    print(f"  source      : {args.source}")
    print(f"  reference   : {args.reference}")
    print(f"  mask_type   : {args.mask_type}")
    print(f"  decomp      : {args.decomp}")
    print(f"  target_size : {args.target_size}px")
    print(f"  artifacts   : {os.path.abspath(args.artifacts_dir)}/")
    if args.sam_checkpoint:
        print(f"  sam_ckpt    : {args.sam_checkpoint}")
        print(f"  sam_type    : {args.sam_type}")

    # ── Run Stage 1 ──────────────────────────────────────────────────────────
    try:
        artifacts = run_stage1(args)
    except FileNotFoundError as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] Stage 1 crashed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Save artifacts ───────────────────────────────────────────────────────
    save_artifacts(artifacts, args.artifacts_dir)

    # ── Verify ───────────────────────────────────────────────────────────────
    ok = verify(artifacts, args.target_size)

    # ── Visualize ────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Visualization")
    print("═" * 60)

    save_viz = args.save_viz
    if save_viz is None:
        # Default: save alongside artifacts
        save_viz = os.path.join(args.artifacts_dir, "stage1_visualization.png")

    fig = visualize(artifacts, args, save_path=save_viz)

    if not args.no_display:
        print("[test] Opening visualization window (close to exit)...")
        plt.show()
    else:
        plt.close(fig)

    # ── Final result ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    if ok:
        print("  ✓  Stage 1 complete — all checks passed")
        print(f"  Artifacts → {os.path.abspath(args.artifacts_dir)}/")
        print(f"  Viz       → {os.path.abspath(save_viz)}")
    else:
        print("  ✗  Stage 1 complete — some checks failed (see above)")
    print("═" * 60 + "\n")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()