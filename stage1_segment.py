# stage1_segment.py
"""
Stage 1 — Donor alignment and segmentation.

Entry point for the first stage of the PnP-KV Inpainting pipeline.
Reads configs/default.yaml, runs alignment + segmentation, and saves the
artifacts that Stage 1b (DDIM inversion) and Stage 2 (denoising) need.

Prior construction and frequency decomposition have been removed.
Texture transfer is now handled entirely by PnP KV injection in Stage 1b/2.

Pipeline steps (in order):
    1. Load content + donor images from disk
    2. Align donor to content spatial coordinate system
    3. Object mask: SAM / none  (ablation.mask_type)
    4. Format conversion: numpy BGR → PIL RGB + mask tensor
    5. Save artifacts to artifacts/

After this script completes:
    - Stage 1b (stage1b_invert.py) runs DDIM inversion on the donor
    - Stage 2 (stage2_diffusion.py) runs the denoising loop
    - SAM is fully unloaded from memory before this script exits

Artifacts produced:
    content_pil.png       — content image S resized to target_size, RGB
    donor_aligned_pil.png — donor R̃ aligned + resized to target_size, RGB
    object_mask.pt        — (1, 1, S, S) float32 mask tensor
    meta.json             — parameters + stats read by Stage 1b and Stage 2

Usage:
    python stage1_segment.py
    python stage1_segment.py --config configs/my_experiment.yaml
    python stage1_segment.py --mask-type none
    python stage1_segment.py --content inputs/wall.jpg --donor inputs/marble.jpg
    python stage1_segment.py --align-method flow
    python stage1_segment.py --dry-run

Dependencies:
    pip install opencv-python-headless torch pillow pyyaml numpy
    pip install git+https://github.com/facebookresearch/segment-anything.git
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
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")
if os.path.isdir(_CORE):
    sys.path.insert(0, _CORE)
else:
    sys.path.insert(0, _HERE)

try:
    from alignment    import run_donor_alignment, AlignmentResult
    from segmentation import load_sam_model, get_object_mask, mask_to_tensor
except ImportError as e:
    print(f"\n[stage1] FATAL — could not import core modules: {e}")
    print("  Make sure core/ exists and contains alignment.py, segmentation.py.")
    print("  Run from the project root directory.")
    traceback.print_exc()
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADER
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG_PATH = os.path.join(_HERE, "configs", "default.yaml")


def load_config(config_path: str) -> dict:
    """Load and return the YAML config as a nested dict."""
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
    Only overrides keys that were explicitly provided on the command line.
    """
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
    if args.mask_type is not None:
        cfg["ablation"]["mask_type"] = args.mask_type
    if args.align_method is not None:
        cfg["alignment"]["method"] = args.align_method
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
    print(f"│  [alignment]")
    print(f"│  method         : {cfg['alignment']['method']}")
    print(f"│")
    print(f"│  [ablation]")
    print(f"│  mask_type      : {cfg['ablation']['mask_type']}")
    print(f"│")
    if cfg["ablation"]["mask_type"] == "sam":
        print(f"│  [SAM]")
        print(f"│  sam_checkpoint : {cfg['paths']['sam_checkpoint']}")
        print(f"│  sam_model_type : {cfg['paths']['sam_model_type']}")
        print(f"│  prompt_strategy: {cfg['stage1']['sam']['prompt_strategy']}")
        print(f"│  pred_iou_thresh: {cfg['stage1']['sam']['pred_iou_thresh']}")
        print(f"│")
    print(f"└───────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def step_load_images(cfg: dict):
    """
    Step 1: Load content and donor images from disk.

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
    Stage 1b (DDIM inversion) runs on donor_aligned_pil.png — alignment must
    happen here so the KV features captured during inversion correspond to the
    donor as it would appear in the content's coordinate frame.

    Method is read from cfg["alignment"]["method"]:
        "resize"  — simple resize (default, no extra dependencies)
        "tile"    — tile donor to fill content canvas
        "match"   — ORB feature matching + affine RANSAC
        "flow"    — dense optical flow warp (Farneback)
        "affine"  — explicit affine from user-supplied point pairs

    Returns:
        AlignmentResult with aligned_donor (BGR), method_used, fallback, meta
    """
    method     = cfg["alignment"]["method"]
    content_hw = content_bgr.shape[:2]

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

    The mask defines two regions:
        mask=255  — editable region: KV injection + denoising fills this area
        mask=0    — frozen region: blended latent anchoring keeps this
                    pixel-identical to the content image in the output

    mask_type is read from ablation.mask_type:
        "sam"  → SAM segmentation (default)
        "none" → full-image mask (ablation baseline)

    SAM is loaded, used, and immediately deleted to free VRAM before
    Stage 1b loads the diffusion model.

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
                f"  Download from:\n"
                f"  https://github.com/facebookresearch/segment-anything#model-checkpoints\n"
                f"\n"
                f"  To skip SAM: ablation.mask_type: none  or  --mask-type none"
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

    # Free SAM VRAM immediately — Stage 1b loads the diffusion model next
    if predictor is not None:
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[stage1] SAM predictor deleted — VRAM freed for Stage 1b")

    coverage = float(object_mask_uint8.astype(bool).mean() * 100)
    print(
        f"[stage1] Mask ready | "
        f"mask_type={mask_type} | "
        f"coverage={coverage:.1f}%  "
        f"{'✓' if 5.0 < coverage < 95.0 else '⚠ unusual coverage — check mask quality'}"
    )

    return object_mask_uint8


def step_convert(
    cfg:               dict,
    content_bgr:       np.ndarray,
    donor_aligned_bgr: np.ndarray,
    object_mask_uint8: np.ndarray,
):
    """
    Step 4: Convert numpy outputs to PIL images and tensors at target_size.

    Note: prior construction has been removed. Texture transfer is handled
    by PnP KV injection in Stage 1b / Stage 2. This step only converts
    the content image, the aligned donor, and the mask.

    Returns:
        content_pil        : PIL RGB — content image S at target_size
        donor_aligned_pil  : PIL RGB — aligned donor R̃ at target_size
        object_mask_tensor : (1, 1, target_size, target_size) float32 [0, 1]
    """
    target_size = cfg["image"]["target_size"]

    def _to_pil(bgr: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).resize((target_size, target_size), Image.LANCZOS)

    content_pil       = _to_pil(content_bgr)
    donor_aligned_pil = _to_pil(donor_aligned_bgr)
    object_mask_tensor = mask_to_tensor(object_mask_uint8)

    print(
        f"[stage1] Conversion done | "
        f"PIL size={content_pil.size} | "
        f"mask tensor shape={tuple(object_mask_tensor.shape)} "
        f"range=[{float(object_mask_tensor.min()):.2f}, {float(object_mask_tensor.max()):.2f}]"
    )

    return content_pil, donor_aligned_pil, object_mask_tensor


def step_save(
    cfg:               dict,
    content_pil:       "Image.Image",
    donor_aligned_pil: "Image.Image",
    object_mask_tensor: "torch.Tensor",
    meta:              dict,
):
    """
    Step 5: Save all artifacts to the artifacts directory.

    Files written:
        content_pil.png        — content image S (resized, RGB)
        donor_aligned_pil.png  — aligned donor R̃ (resized, RGB)
        object_mask.pt         — (1, 1, S, S) float32 tensor
        meta.json              — all metadata read by Stage 1b and Stage 2

    Stage 1b reads donor_aligned_pil.png as its inversion input.
    Stage 2 reads content_pil.png and object_mask.pt.
    Both stages read meta.json.

    Note: prior_pil.png and masked_input_pil.png are no longer produced.
    Stage 2's latent initialisation uses content_pil.png directly via
    the standard inpainting conditioning path.
    """
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    saves = {
        "content_pil.png"       : ("pil",    content_pil),
        "donor_aligned_pil.png" : ("pil",    donor_aligned_pil),
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
    object_mask_uint8: np.ndarray,
) -> dict:
    """
    Build the meta.json dict that Stage 1b and Stage 2 read at startup.

    Compared to the old meta, prior construction fields (alpha, beta, gamma,
    histogram_match, decomp_method, LF/HF stats) have been removed because
    prior construction no longer exists in this pipeline.

    Stage 1b adds its own fields to meta.json after inversion completes
    (num_inversion_steps, kv_store_dir, layers_stored).
    """
    coverage = float(object_mask_uint8.astype(bool).mean() * 100)

    return {
        # ── Image paths ──────────────────────────────────────────────────
        "content_path"       : os.path.abspath(cfg["paths"]["content_image"]),
        "donor_path"         : os.path.abspath(cfg["paths"]["donor_image"]),

        # ── Image size ────────────────────────────────────────────────────
        "target_size"        : cfg["image"]["target_size"],

        # ── Alignment ────────────────────────────────────────────────────
        "alignment_method"   : alignment_result.method_used,
        "alignment_fallback" : alignment_result.fallback,
        "alignment_meta"     : alignment_result.meta,

        # ── Mask ─────────────────────────────────────────────────────────
        "mask_type"          : cfg["ablation"]["mask_type"],
        "sam_model_type"     : cfg["paths"].get("sam_model_type", "n/a"),
        "prompt_strategy"    : cfg["stage1"]["sam"].get("prompt_strategy", "n/a"),
        "mask_coverage_pct"  : round(coverage, 2),

        # ── Ablation flags (Stage 2 reads these) ─────────────────────────
        # Prior-related flags removed. Remaining flags are all Stage 2 flags.
        "ablation": {
            "mask_type"          : cfg["ablation"]["mask_type"],
            "blended_anchoring"  : cfg["ablation"].get("blended_anchoring",  True),
            "shallow_injection"  : cfg["ablation"].get("shallow_injection",  True),
            "temporal_anneal"    : cfg["ablation"].get("temporal_anneal",    True),
            "compositing"        : cfg["ablation"].get("compositing",        "freq"),
        },

        # ── Provenance ────────────────────────────────────────────────────
        "stage1_script"      : os.path.abspath(__file__),
        "timestamp"          : datetime.datetime.now().isoformat(),

        # ── Stage 1b fields (populated later by stage1b_invert.py) ───────
        # These keys are written here as null so Stage 2 can detect whether
        # Stage 1b has run and raise a clear error if it hasn't.
        "stage1b_complete"   : False,
        "kv_store_dir"       : None,
        "num_inversion_steps": None,
        "layers_stored"      : None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_stage1(cfg: dict):
    """
    Execute all Stage 1 steps in order.

    Steps:
        load → align → mask → convert → meta → save

    Prior construction (decompose + build_prior) has been removed.
    The pipeline produces four artifacts only:
        content_pil.png, donor_aligned_pil.png, object_mask.pt, meta.json

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

    # ── Step 4: Format conversion ─────────────────────────────────────────
    _section("Step 4 — Format conversion (numpy → PIL + tensor)")
    content_pil, donor_aligned_pil, object_mask_tensor = step_convert(
        cfg, content_bgr, donor_aligned_bgr, object_mask_uint8
    )

    # ── Build meta ────────────────────────────────────────────────────────
    meta = build_meta(cfg, alignment_result, object_mask_uint8)

    # ── Step 5: Save artifacts ────────────────────────────────────────────
    _section("Step 5 — Save artifacts")
    step_save(cfg, content_pil, donor_aligned_pil, object_mask_tensor, meta)

    return cfg["paths"]["artifacts_dir"], meta


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _print_summary(artifacts_dir: str, meta: dict):
    coverage = meta["mask_coverage_pct"]

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
    print(f"  target_size      : {meta['target_size']}px")
    print(f"  timestamp        : {meta['timestamp'][:19]}")
    print(f"")
    print(f"  Artifacts saved:")
    print(f"    content_pil.png        — content image S")
    print(f"    donor_aligned_pil.png  — aligned donor R̃  (Stage 1b inversion input)")
    print(f"    object_mask.pt         — mask tensor for KV gating + anchoring")
    print(f"    meta.json              — pipeline metadata")
    print(f"")
    print(f"  ⚠  Prior construction removed — texture transfer via PnP KV injection.")
    print(f"     Run Stage 1b next to capture donor KV features:")
    print(f"       python stage1b_invert.py")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stage 1 — donor alignment and object segmentation. "
            "Prior construction removed; texture transfer is now handled "
            "by PnP KV injection in Stage 1b / Stage 2."
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
        help="Override paths.content_image"
    )
    p.add_argument(
        "--donor", default=None,
        help="Override paths.donor_image"
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

    # ── Method overrides ───────────────────────────────────────────────────
    p.add_argument(
        "--mask-type", default=None,
        choices=["sam", "none"],
        help="Override ablation.mask_type"
    )
    p.add_argument(
        "--align-method", default=None,
        choices=["resize", "tile", "match", "flow", "affine"],
        help="Override alignment.method"
    )

    # ── Image resolution override ──────────────────────────────────────────
    p.add_argument(
        "--target-size", type=int, default=None,
        choices=[512, 768],
        help="Override image.target_size"
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

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"\n[stage1] FATAL — {e}")
        sys.exit(1)

    cfg = apply_cli_overrides(cfg, args)
    print_config(cfg)

    if args.dry_run:
        print("[stage1] --dry-run: exiting without running.\n")
        sys.exit(0)

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

    _print_summary(artifacts_dir, meta)
    sys.exit(0)


if __name__ == "__main__":
    main()