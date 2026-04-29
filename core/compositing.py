# core/compositing.py
"""
Frequency-decomposed compositing module.

Problem this solves:
    After KV injection, the diffusion output has the right identity but wrong
    color/skin tone relative to the source. A naive hard composite or even
    Poisson blending cannot fix a global color mismatch — it only smooths
    the boundary seam. The face still looks stitched because LF (skin tone,
    color temperature) and HF (texture detail, edges) need to be handled
    separately.

Solution — Frequency-decomposed compositing:
    Instead of blending the full images directly, split both the source and
    the generated image into LF and HF components, blend each frequency band
    independently with the appropriate technique, then reconstruct:

        source_LF,    source_HF    = decompose(source)
        generated_LF, generated_HF = decompose(generated)

        final_LF = blend_lf(source_LF, generated_LF, mask)
                   ← color transfer here (LAB statistics matching)
                   ← ensures skin tone continuity at LF level

        final_HF = blend_hf(source_HF, generated_HF, mask)
                   ← Poisson blending here (seam removal at HF level)
                   ← HF is where hard seams are most perceptually visible

        final = clip(final_LF + final_HF)

    Why this works:
        LF carries global color, brightness, and skin tone.
        Matching LF color statistics (color transfer) eliminates the
        color mismatch that makes faces look stitched.

        HF carries edges, texture, and fine detail.
        Poisson blending on HF removes the boundary seam without
        affecting the color balance fixed in the LF step.

        Separating the two allows each problem to be solved with the
        right tool at the right frequency scale.

Novel contribution for paper:
    Prior work uses frequency decomposition for KV injection guidance
    (Stage 1 of this pipeline) OR for image compositing — never both
    in the same pipeline. This module extends frequency decomposition
    end-to-end: Stage 1 decomposes for injection, Stage 2 decomposes
    for compositing. LF and HF are treated as first-class signals
    throughout the entire face swap pipeline.

Ablation flag (configs/default.yaml):
    compositing:
      freq_decomposed: true    # true = novel freq-decomposed path
                               # false = old hard composite + Poisson only

Public API:
    composite_result(generated_pil, source_pil, face_mask_tensor, cfg)
        → PIL RGB final image
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
    Resize face mask tensor to target_size and feather edges.

    Args:
        face_mask_tensor : (1, 1, H, W) float32 [0,1] from artifacts/face_mask.pt
        target_size      : (W, H) PIL convention
        feather_px       : Gaussian blur radius. 0 = hard binary mask.

    Returns:
        (H, W) uint8 numpy, 0-255. 255 = face region.
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
    Statistics computed inside the face mask so background
    pixels don't pollute the color matching.

    Args:
        source_lf    : LF of source — defines target color stats.
        generated_lf : LF of generated — will be color-corrected.
        mask_uint8   : Face region mask.

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

    Color transfer eliminates the skin tone / brightness mismatch
    that causes the stitched look. Alpha blend then smoothly transitions
    from source LF to color-corrected generated LF across the mask boundary.

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

    HF is where seams are most visible — hard edges at the mask boundary
    are immediately noticeable. Poisson blending harmonizes gradients
    at the boundary so the transition is invisible.

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
# FREQUENCY-DECOMPOSED COMPOSITE  (novel path)
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

    Kept as ablation condition (compositing.freq_decomposed: false).
    Compare directly against freq-decomposed path to isolate the
    contribution of frequency-aware blending.

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
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def composite_result(
    generated_pil:    Image.Image,
    source_pil:       Image.Image,
    face_mask_tensor: torch.Tensor,
    cfg:              dict,
) -> Image.Image:
    """
    Composite the generated face onto the source image.

    Reads compositing.freq_decomposed from cfg to select path:

        freq_decomposed: true  (default — novel method)
            decompose both images into LF + HF
            blend LF with color transfer  → fixes skin tone mismatch
            blend HF with Poisson         → fixes boundary seam
            reconstruct: final = LF + HF

        freq_decomposed: false  (ablation baseline)
            alpha blend + optional Poisson blend (original behavior)

    Config keys:
        compositing.freq_decomposed  bool  default True
        compositing.poisson_blend    bool  default True
        compositing.feather_px       int   default 8

    Args:
        generated_pil    : PIL RGB diffusion output from decode_latent().
        source_pil       : PIL RGB original source from artifacts/.
        face_mask_tensor : (1,1,H,W) float32 [0,1] from artifacts/face_mask.pt.
        cfg              : Full resolved config dict.

    Returns:
        PIL RGB — reference face on source background, no stitching.
    """
    comp_cfg        = cfg.get("compositing", {})
    freq_decomposed = comp_cfg.get("freq_decomposed", True)
    poisson_blend   = comp_cfg.get("poisson_blend",   True)
    feather_px      = comp_cfg.get("feather_px",      8)

    # Size check
    if generated_pil.size != source_pil.size:
        print(
            f"[compositing] WARNING: size mismatch "
            f"generated={generated_pil.size} source={source_pil.size}. "
            f"Resizing generated to match source."
        )
        generated_pil = generated_pil.resize(source_pil.size, Image.LANCZOS)

    target_size = source_pil.size   # (W, H)

    # Prepare mask
    mask_uint8 = _prepare_mask(face_mask_tensor, target_size, feather_px)

    # PIL RGB → BGR numpy
    generated_bgr = cv2.cvtColor(np.array(generated_pil), cv2.COLOR_RGB2BGR)
    source_bgr    = cv2.cvtColor(np.array(source_pil),    cv2.COLOR_RGB2BGR)

    # Select compositing path
    if freq_decomposed:
        print("[compositing] Mode: frequency-decomposed (novel method)")
        final_bgr = _freq_composite(
            generated_bgr = generated_bgr,
            source_bgr    = source_bgr,
            mask_uint8    = mask_uint8,
            source_pil    = source_pil,
            generated_pil = generated_pil,
            cfg           = cfg,
        )
    else:
        print("[compositing] Mode: simple composite (ablation baseline)")
        final_bgr = _simple_composite(
            generated_bgr = generated_bgr,
            source_bgr    = source_bgr,
            mask_uint8    = mask_uint8,
            poisson_blend = poisson_blend,
        )

    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)