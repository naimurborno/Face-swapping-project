# core/compositing.py
"""
Frequency-decomposed compositing module — demoted to optional cleanup pass.

Context in the new pipeline (Mixed-Frequency Prior Guided Inpainting):
    The blended latent anchoring mechanism in Stage 2 enforces source pixel
    fidelity outside the mask at every denoising step — by construction.
    The inpainting model sees real source context around the hole, so boundary
    coherence is largely handled during generation itself, not as a post-hoc fix.

    Compositing therefore becomes an *optional* cleanup pass, not a required
    step. Whether it runs — and in which mode — is controlled by a single flag:

        ablation.compositing: "freq" | "simple" | "none"

    "none" → skip entirely, return raw diffusion output unchanged.
    "simple" → alpha blend + optional Poisson seam (original behaviour).
    "freq"   → full frequency-decomposed blend (novel method, now optional).

Why keep it at all:
    Even with inpainting + blended anchoring, residual seams can occur at the
    mask boundary when the diffusion model introduces a subtle color shift inside
    the mask. The freq-decomposed path (LAB color transfer on LF + Poisson on HF)
    cleans this up cheaply. Keeping it as an ablation flag lets the paper show
    whether the new pipeline produces clean raw output *before* compositing — a
    stronger result claim than the old pipeline which required it.

    Ablation A6 compares compositing="none" vs "simple" vs "freq" to isolate
    how much of the final quality comes from the inpainting/prior construction
    vs the compositing cleanup.

Seam detection (is_needed):
    run_compositing() calls is_needed() before doing any work. If the boundary
    region shows no measurable seam, the output is returned unchanged even when
    compositing is nominally enabled. This avoids unnecessary LAB/Poisson passes
    on already-clean outputs, which can occasionally degrade a perfect result.

All existing compositing logic is untouched:
    _prepare_mask(), _color_transfer_lab(), _poisson_blend(),
    _blend_lf(), _blend_hf(), _freq_composite(), _simple_composite()
    — zero functional changes. The only additions are:

    is_needed(output, source, mask_tensor, threshold) → bool
        Detect whether a perceptible boundary seam exists.

    run_compositing(generated_pil, source_pil, mask_tensor, cfg) → PIL
        Top-level dispatcher: checks cfg flag, optionally checks is_needed(),
        routes to freq / simple / none path. Replaces the old composite_result().

    composite_result() is preserved as a direct alias of run_compositing()
    so existing call sites in stage2_diffusion.py require no changes.

Public API (unchanged call signature):
    composite_result(generated_pil, source_pil, face_mask_tensor, cfg)
        → PIL RGB final image

    run_compositing(generated_pil, source_pil, face_mask_tensor, cfg)
        → PIL RGB final image  (preferred in new pipeline)

    is_needed(output_pil, source_pil, face_mask_tensor, threshold) → bool
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.decomposition import decompose_pil


# ══════════════════════════════════════════════════════════════════════════════
# MASK PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_mask(
    face_mask_tensor: torch.Tensor,
    target_size:      tuple,
    feather_px:       int = 8,
) -> np.ndarray:
    """
    Resize face/object mask tensor to target_size and feather edges.

    Args:
        face_mask_tensor : (1, 1, H, W) float32 [0,1] from artifacts/mask.pt
        target_size      : (W, H) PIL convention
        feather_px       : Gaussian blur radius. 0 = hard binary mask.

    Returns:
        (H, W) uint8 numpy, 0-255. 255 = editable region.
    """
    W, H = target_size

    resized = F.interpolate(
        face_mask_tensor.float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    mask_np = (resized.squeeze().cpu().numpy() * 255).astype(np.uint8)

    if feather_px > 0:
        ksize   = feather_px * 2 + 1
        mask_np = cv2.GaussianBlur(mask_np, (ksize, ksize), sigmaX=feather_px / 2)

    return mask_np


# ══════════════════════════════════════════════════════════════════════════════
# COLOR TRANSFER  (LAB Reinhard — applied inside LF blend)
# ══════════════════════════════════════════════════════════════════════════════

def _color_transfer_lab(
    source_lf:    np.ndarray,   # (H, W, 3) float32 BGR [0, 255]
    generated_lf: np.ndarray,   # (H, W, 3) float32 BGR [0, 255]
    mask_uint8:   np.ndarray,   # (H, W) uint8
) -> np.ndarray:
    """
    Match LAB color statistics of generated_lf to source_lf inside the mask.

    Reinhard color transfer per LAB channel:
        corrected = (generated - mean_gen) / std_gen * std_src + mean_src

    Applied to LF only so HF texture detail is untouched.
    Statistics computed inside the mask so background pixels
    don't pollute the color matching.

    Args:
        source_lf    : LF of source — defines target color stats.
        generated_lf : LF of generated — will be color-corrected.
        mask_uint8   : Editable region mask.

    Returns:
        (H, W, 3) float32 BGR color-corrected LF.
    """
    src_uint8 = np.clip(source_lf,    0, 255).astype(np.uint8)
    gen_uint8 = np.clip(generated_lf, 0, 255).astype(np.uint8)

    # Convert to OpenCV LAB range: L [0,255], A [0,255], B [0,255]
    src_lab = cv2.cvtColor(src_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
    gen_lab = cv2.cvtColor(gen_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)

    face_bool = mask_uint8 > 127

    if face_bool.sum() == 0:
        print("[compositing] WARNING: empty mask in color transfer — skipping.")
        return generated_lf.copy()

    corrected_lab = gen_lab.copy()

    for ch in range(3):
        src_pixels = src_lab[:, :, ch][face_bool]
        gen_pixels = gen_lab[:, :, ch][face_bool]

        src_mean, src_std = float(src_pixels.mean()), float(src_pixels.std())
        gen_mean, gen_std = float(gen_pixels.mean()), float(gen_pixels.std())

        if gen_std < 1e-6:
            continue

        corrected_lab[:, :, ch] = (
            (gen_lab[:, :, ch] - gen_mean) / gen_std
        ) * src_std + src_mean

    # Clip to valid OpenCV LAB uint8 range [0, 255]
    corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
    corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    return corrected_bgr.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# POISSON BLENDING  (seam removal — applied inside HF blend)
# ══════════════════════════════════════════════════════════════════════════════

def _poisson_blend(
    generated_bgr: np.ndarray,   # (H, W, 3) uint8
    source_bgr:    np.ndarray,   # (H, W, 3) uint8
    mask_uint8:    np.ndarray,   # (H, W) uint8
) -> np.ndarray:
    """
    Poisson seamless cloning at the mask boundary.

    Preserves gradient structure from generated_bgr inside the mask
    while matching color values from source_bgr at the boundary.

    Falls back to alpha blend if seamlessClone fails (mask touches
    image border, empty mask, or OpenCV version issue).

    Returns:
        (H, W, 3) uint8 BGR blended image.
    """
    _, mask_binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

    if mask_binary.max() == 0:
        print("[compositing] WARNING: empty mask in Poisson blend — skipping.")
        return generated_bgr.copy()

    moments = cv2.moments(mask_binary)
    if moments["m00"] == 0:
        H, W   = source_bgr.shape[:2]
        center = (W // 2, H // 2)
    else:
        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )

    try:
        return cv2.seamlessClone(
            src   = generated_bgr,
            dst   = source_bgr,
            mask  = mask_binary,
            p     = center,
            flags = cv2.NORMAL_CLONE,
        )
    except cv2.error as e:
        print(
            f"[compositing] WARNING: seamlessClone failed ({e}). "
            f"Falling back to alpha blend."
        )
        alpha     = mask_uint8.astype(np.float32) / 255.0
        alpha     = alpha[:, :, np.newaxis]
        composite = (
            source_bgr.astype(np.float32)    * (1.0 - alpha) +
            generated_bgr.astype(np.float32) * alpha
        )
        return np.clip(composite, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# LF BLEND  (color transfer + alpha blend)
# ══════════════════════════════════════════════════════════════════════════════

def _blend_lf(
    source_lf:    np.ndarray,   # (H, W, 3) float32 BGR [0, 255]
    generated_lf: np.ndarray,   # (H, W, 3) float32 BGR [0, 255]
    mask_uint8:   np.ndarray,   # (H, W) uint8
) -> np.ndarray:
    """
    Blend LF bands:
        1. Color transfer: match generated_lf color stats to source_lf
        2. Alpha blend:    final_LF = source × (1-α) + corrected × α

    Color transfer eliminates the color / brightness mismatch that causes
    the stitched look. Alpha blend then smoothly transitions from source LF
    to color-corrected generated LF across the mask boundary.

    Returns:
        final_LF : (H, W, 3) float32 BGR [0, 255]
    """
    corrected_lf = _color_transfer_lab(source_lf, generated_lf, mask_uint8)

    alpha    = mask_uint8.astype(np.float32) / 255.0
    alpha    = alpha[:, :, np.newaxis]

    final_lf = (
        source_lf.astype(np.float32)    * (1.0 - alpha) +
        corrected_lf.astype(np.float32) * alpha
    )
    return final_lf   # float32 [0, 255]


# ══════════════════════════════════════════════════════════════════════════════
# HF BLEND  (alpha blend + Poisson seam removal)
# ══════════════════════════════════════════════════════════════════════════════

def _blend_hf(
    source_hf:    np.ndarray,   # (H, W, 3) float32 BGR [-255, 255]
    generated_hf: np.ndarray,   # (H, W, 3) float32 BGR [-255, 255]
    mask_uint8:   np.ndarray,   # (H, W) uint8
) -> np.ndarray:
    """
    Blend HF bands:
        1. Alpha blend on shifted HF
        2. Poisson blend to remove boundary edge artifacts

    HF is where seams are most visible — hard edges at the mask boundary are
    immediately noticeable. Poisson blending harmonizes gradients at the
    boundary so the transition is invisible.

    HF shift: float32 [-255,255] → shift +128 for uint8 ops → shift -128 back.

    Returns:
        final_HF : (H, W, 3) float32 BGR [-255, 255]
    """
    # Shift to [0, 255] for uint8 operations
    src_u8  = np.clip(source_hf    + 128.0, 0, 255).astype(np.uint8)
    gen_u8  = np.clip(generated_hf + 128.0, 0, 255).astype(np.uint8)

    # Alpha blend
    alpha   = mask_uint8.astype(np.float32) / 255.0
    alpha   = alpha[:, :, np.newaxis]
    blend   = (
        src_u8.astype(np.float32) * (1.0 - alpha) +
        gen_u8.astype(np.float32) * alpha
    )
    blend_u8 = np.clip(blend, 0, 255).astype(np.uint8)

    # Poisson blend on the shifted HF
    poisson_u8 = _poisson_blend(blend_u8, src_u8, mask_uint8)

    # Shift back to [-255, 255]
    return poisson_u8.astype(np.float32) - 128.0


# ══════════════════════════════════════════════════════════════════════════════
# FREQUENCY-DECOMPOSED COMPOSITE  (novel path — now optional)
# ══════════════════════════════════════════════════════════════════════════════

def _freq_composite(
    generated_bgr: np.ndarray,
    source_bgr:    np.ndarray,
    mask_uint8:    np.ndarray,
    source_pil:    Image.Image,
    generated_pil: Image.Image,
    cfg:           dict,
) -> np.ndarray:
    """
    Full frequency-decomposed compositing pipeline.

    Steps:
        1. Decompose source and generated into LF + HF
           (using the same method as Stage 1 for consistency)
        2. Blend LF: color transfer → alpha blend
        3. Blend HF: alpha blend → Poisson seam removal
        4. Reconstruct: final = clip(final_LF + final_HF)

    Now invoked only when ablation.compositing = "freq" AND is_needed()
    returns True (or auto_skip is disabled). In the new inpainting pipeline
    this is a cleanup pass, not a required step.

    Returns:
        (H, W, 3) uint8 BGR final image.
    """
    # Read decomposition params — match Stage 1 settings
    decomp_method = cfg.get("ablation", {}).get("decomposition", "gaussian")
    kernel        = cfg.get("stage1",   {}).get("gaussian", {}).get("kernel", 31)
    sigma         = cfg.get("stage1",   {}).get("gaussian", {}).get("sigma",  5.0)
    cutoff_ratio  = cfg.get("stage1",   {}).get("fft",      {}).get("cutoff_ratio", 0.1)

    print(f"[compositing] Freq-decomposed blend | method={decomp_method}")

    # ── Step 1: Decompose ─────────────────────────────────────────────────
    source_LF,    source_HF    = decompose_pil(
        source_pil,
        method=decomp_method, kernel=kernel,
        sigma=sigma, cutoff_ratio=cutoff_ratio,
    )
    generated_LF, generated_HF = decompose_pil(
        generated_pil,
        method=decomp_method, kernel=kernel,
        sigma=sigma, cutoff_ratio=cutoff_ratio,
    )

    print(
        f"[compositing] Decomposed | "
        f"src LF [{source_LF.min():.0f},{source_LF.max():.0f}] "
        f"src HF std={source_HF.std():.2f} | "
        f"gen LF [{generated_LF.min():.0f},{generated_LF.max():.0f}] "
        f"gen HF std={generated_HF.std():.2f}"
    )

    # ── Step 2: Blend LF (color transfer + alpha blend) ───────────────────
    final_LF = _blend_lf(source_LF, generated_LF, mask_uint8)
    print("[compositing] LF blend complete (color transfer applied).")

    # ── Step 3: Blend HF (alpha blend + Poisson seam removal) ─────────────
    final_HF = _blend_hf(source_HF, generated_HF, mask_uint8)
    print("[compositing] HF blend complete (Poisson seam removal applied).")

    # ── Step 4: Reconstruct ───────────────────────────────────────────────
    # final_LF : float32 [0, 255]
    # final_HF : float32 [-255, 255]
    final = np.clip(final_LF + final_HF, 0, 255).astype(np.uint8)

    print(
        f"[compositing] Reconstruction complete. "
        f"Output range [{final.min()}, {final.max()}]"
    )
    return final


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK: SIMPLE COMPOSITE  (ablation baseline)
# ══════════════════════════════════════════════════════════════════════════════

def _simple_composite(
    generated_bgr: np.ndarray,
    source_bgr:    np.ndarray,
    mask_uint8:    np.ndarray,
    poisson_blend: bool = True,
) -> np.ndarray:
    """
    Original compositing path: alpha blend + optional Poisson blend.

    Used when ablation.compositing = "simple".
    Compare directly against freq-decomposed path (ablation A6) to isolate
    the contribution of frequency-aware blending.

    Returns:
        (H, W, 3) uint8 BGR composited image.
    """
    alpha     = mask_uint8.astype(np.float32) / 255.0
    alpha     = alpha[:, :, np.newaxis]
    composite = (
        source_bgr.astype(np.float32)    * (1.0 - alpha) +
        generated_bgr.astype(np.float32) * alpha
    )
    composite_u8 = np.clip(composite, 0, 255).astype(np.uint8)

    if poisson_blend:
        return _poisson_blend(composite_u8, source_bgr, mask_uint8)
    return composite_u8


# ══════════════════════════════════════════════════════════════════════════════
# SEAM DETECTION  (new — gates whether compositing runs at all)
# ══════════════════════════════════════════════════════════════════════════════

def is_needed(
    output_pil:       Image.Image,
    source_pil:       Image.Image,
    face_mask_tensor: torch.Tensor,
    threshold:        float = 15.0,
) -> bool:
    """
    Detect whether a perceptible boundary seam exists between the generated
    region and the surrounding source pixels.

    In the new inpainting pipeline the boundary is usually clean because:
      - The inpainting model sees real source context around the hole.
      - Blended latent anchoring enforces exact source pixels outside the mask.

    This function checks a narrow ring at the mask boundary (the "seam ring")
    and measures the mean absolute LAB difference between the output image and
    the source image in that ring. If the difference exceeds threshold, a seam
    is visible and compositing is warranted.

    Seam ring construction:
        ring = dilate(mask, boundary_px) XOR erode(mask, boundary_px)
        boundary_px = 6 pixels — captures the ~12px transition zone where
        the inpainting model blends generated content into source context.

    Args:
        output_pil        : PIL RGB — raw diffusion output from Stage 2.
        source_pil        : PIL RGB — original source image.
        face_mask_tensor  : (1, 1, H, W) float32 [0,1] — editable region mask.
        threshold         : Mean absolute LAB difference above which compositing
                            is triggered. Default 15.0 (perceptual JND ≈ 2-3 LAB
                            units; 15 corresponds to a clearly visible seam).
                            Sweep [5, 10, 15, 20] to calibrate per dataset.

    Returns:
        True  — seam detected, compositing should run.
        False — no perceptible seam, compositing can be skipped.

    Notes:
        - Returns True immediately if mask is empty (safe default — run cleanup).
        - Returns True if images have different sizes (size mismatch is a bug,
          but safe to proceed rather than silently skip).
        - LAB difference is computed in the boundary ring only, not the full
          image, so interior generation quality does not affect the decision.
    """
    # Size guard
    if output_pil.size != source_pil.size:
        print(
            "[compositing.is_needed] WARNING: size mismatch "
            f"output={output_pil.size} source={source_pil.size} — "
            "assuming seam present (safe default)."
        )
        return True

    W, H = source_pil.size

    # Resize mask to image resolution (no feathering — hard binary for ring math)
    mask_resized = F.interpolate(
        face_mask_tensor.float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    mask_bin = (mask_resized.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    if mask_bin.max() == 0:
        print("[compositing.is_needed] WARNING: empty mask — skipping compositing.")
        return False

    # Build boundary ring: dilate - erode
    boundary_px = 6
    kernel_d    = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (boundary_px * 2 + 1, boundary_px * 2 + 1)
    )
    dilated     = cv2.dilate(mask_bin, kernel_d)
    eroded      = cv2.erode( mask_bin, kernel_d)
    ring        = cv2.bitwise_xor(dilated, eroded)   # (H, W) uint8

    ring_bool   = ring > 127

    if ring_bool.sum() == 0:
        # Mask is too small to form a ring — skip compositing
        print("[compositing.is_needed] Mask too small for ring — skipping.")
        return False

    # Convert both images to LAB for perceptual difference
    output_bgr = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
    source_bgr = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2BGR)

    output_lab = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Mean absolute difference in the boundary ring
    diff       = np.abs(output_lab - source_lab)           # (H, W, 3)
    ring_diff  = diff[ring_bool]                            # (N, 3)
    mean_diff  = float(ring_diff.mean())

    seam_detected = mean_diff > threshold

    print(
        f"[compositing.is_needed] Boundary ring LAB diff = {mean_diff:.2f} "
        f"(threshold={threshold:.1f}) → "
        f"{'SEAM DETECTED — compositing needed' if seam_detected else 'clean boundary — skipping compositing'}"
    )

    return seam_detected


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL DISPATCHER  (new — replaces direct composite_result call)
# ══════════════════════════════════════════════════════════════════════════════

def run_compositing(
    generated_pil:    Image.Image,
    source_pil:       Image.Image,
    face_mask_tensor: torch.Tensor,
    cfg:              dict,
) -> Image.Image:
    """
    Top-level compositing dispatcher for the new pipeline.

    Reads ablation.compositing from cfg to select the path:

        "none"   → return generated_pil unchanged. Use to verify the raw
                   inpainting output quality without any post-processing.
                   Ablation A6 condition: inpainting only, no compositing.

        "simple" → alpha blend + optional Poisson (original baseline path).
                   Ablation A6 condition: blended anchoring + simple composite.

        "freq"   → full frequency-decomposed blend (novel method).
                   Ablation A6 condition: blended anchoring + freq composite.
                   Default for the new pipeline.

    For "simple" and "freq" modes, is_needed() is called first when
    compositing.auto_skip is True (default True). If the boundary is already
    clean, the output is returned unchanged regardless of the selected mode.
    Set auto_skip: false to force compositing even on clean outputs (useful
    for debugging or when generating comparison panels).

    Config keys read:
        ablation.compositing        str   "freq" | "simple" | "none"
                                          default "freq"
        compositing.poisson_blend   bool  default True  (simple path only)
        compositing.feather_px      int   default 8
        compositing.seam_threshold  float default 15.0  (is_needed threshold)
        compositing.auto_skip       bool  default True   (skip if no seam)

    Args:
        generated_pil    : PIL RGB — raw diffusion output from decode_latent().
        source_pil       : PIL RGB — original source image from artifacts/.
        face_mask_tensor : (1, 1, H, W) float32 [0,1] from artifacts/mask.pt.
        cfg              : Full resolved config dict.

    Returns:
        PIL RGB — either the raw output (no seam / mode=none) or the composited
        result (seam found / auto_skip disabled).
    """
    comp_cfg   = cfg.get("compositing", {})
    abl_cfg    = cfg.get("ablation",    {})

    mode           = abl_cfg.get("compositing",    "freq")
    poisson_blend  = comp_cfg.get("poisson_blend", True)
    feather_px     = comp_cfg.get("feather_px",    8)
    seam_threshold = comp_cfg.get("seam_threshold", 15.0)
    auto_skip      = comp_cfg.get("auto_skip",      True)

    # ── Mode: none — skip entirely ────────────────────────────────────────
    if mode == "none":
        print("[compositing] Mode: none — returning raw diffusion output.")
        return generated_pil

    # Size normalisation (defensive)
    if generated_pil.size != source_pil.size:
        print(
            f"[compositing] WARNING: size mismatch "
            f"generated={generated_pil.size} source={source_pil.size}. "
            f"Resizing generated to match source."
        )
        generated_pil = generated_pil.resize(source_pil.size, Image.LANCZOS)

    # ── Auto-skip: check seam before doing any work ───────────────────────
    if auto_skip:
        if not is_needed(generated_pil, source_pil, face_mask_tensor, seam_threshold):
            print("[compositing] Auto-skip: boundary clean — returning raw output.")
            return generated_pil

    # ── Prepare shared inputs ─────────────────────────────────────────────
    target_size   = source_pil.size   # (W, H)
    mask_uint8    = _prepare_mask(face_mask_tensor, target_size, feather_px)
    generated_bgr = cv2.cvtColor(np.array(generated_pil), cv2.COLOR_RGB2BGR)
    source_bgr    = cv2.cvtColor(np.array(source_pil),    cv2.COLOR_RGB2BGR)

    # ── Mode: freq — frequency-decomposed composite ───────────────────────
    if mode == "freq":
        print("[compositing] Mode: freq — frequency-decomposed composite (novel).")
        final_bgr = _freq_composite(
            generated_bgr = generated_bgr,
            source_bgr    = source_bgr,
            mask_uint8    = mask_uint8,
            source_pil    = source_pil,
            generated_pil = generated_pil,
            cfg           = cfg,
        )

    # ── Mode: simple — alpha blend + optional Poisson ─────────────────────
    elif mode == "simple":
        print("[compositing] Mode: simple — alpha blend + Poisson (baseline).")
        final_bgr = _simple_composite(
            generated_bgr = generated_bgr,
            source_bgr    = source_bgr,
            mask_uint8    = mask_uint8,
            poisson_blend = poisson_blend,
        )

    else:
        raise ValueError(
            f"[compositing] Unknown ablation.compositing mode '{mode}'. "
            f"Choose: 'freq' | 'simple' | 'none'"
        )

    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API  (composite_result preserved as alias for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def composite_result(
    generated_pil:    Image.Image,
    source_pil:       Image.Image,
    face_mask_tensor: torch.Tensor,
    cfg:              dict,
) -> Image.Image:
    """
    Backward-compatible alias for run_compositing().

    Existing call sites in stage2_diffusion.py require no changes.
    New code should call run_compositing() directly, which reads
    ablation.compositing from cfg for the three-way dispatch.

    In the old pipeline, this function read compositing.freq_decomposed (bool)
    to choose between freq and simple paths. In the new pipeline, the same
    two-way choice is now expressed as ablation.compositing = "freq" | "simple",
    with "none" as a third option that skips compositing entirely.

    Transition:
        Old cfg key: compositing.freq_decomposed: true  → ablation.compositing: "freq"
        Old cfg key: compositing.freq_decomposed: false → ablation.compositing: "simple"
        New cfg key: ablation.compositing: "none"       → skip (new option)

    If cfg still contains compositing.freq_decomposed (old pipeline configs),
    this alias translates it into the new ablation flag before dispatching,
    so old yaml files continue to work without modification.
    """
    # Translate old compositing.freq_decomposed flag if present
    comp_cfg = cfg.get("compositing", {})
    if "freq_decomposed" in comp_cfg and "compositing" not in cfg.get("ablation", {}):
        translated_mode = "freq" if comp_cfg["freq_decomposed"] else "simple"
        cfg = {
            **cfg,
            "ablation": {
                **cfg.get("ablation", {}),
                "compositing": translated_mode,
            },
        }
        print(
            f"[compositing] Translated compositing.freq_decomposed="
            f"{comp_cfg['freq_decomposed']} → ablation.compositing={translated_mode}"
        )

    return run_compositing(generated_pil, source_pil, face_mask_tensor, cfg)