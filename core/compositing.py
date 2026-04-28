# core/compositing.py
"""
Post-diffusion compositing module.

Problem this solves:
    The diffusion denoising loop regenerates the ENTIRE image — background,
    hair, neck, clothing all drift slightly from the source even though KV
    injection only targets the face region. The KV mask gates attention
    features but cannot prevent the diffusion process from altering pixels
    outside the face.

Solution:
    After decoding the final latent to a PIL image, composite the result
    back onto the source using the face mask:

        final = source × (1 - mask) + generated × mask

    Everything outside the mask comes directly from source pixels.
    Everything inside the mask comes from the generated image.

    A hard composite at the mask boundary creates a visible seam because
    the generated face has slightly different color/brightness at the edges
    due to lighting normalization in diffusion. Poisson blending
    (cv2.seamlessClone) smooths this transition by solving a gradient-domain
    optimization that matches interior gradients from the generated image
    while matching boundary conditions from the source.

Public API:
    composite_result(generated_pil, source_pil, face_mask_tensor, cfg)
        → PIL RGB final image

    The cfg argument reads one flag:
        cfg["compositing"]["poisson_blend"]  (bool, default True)

    If poisson_blend=False, returns the hard composite without seam removal.
    This is useful for debugging — a visible seam confirms the mask boundary
    is correct before Poisson blending hides it.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# MASK PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_mask(
    face_mask_tensor: torch.Tensor,   # (1, 1, H, W) float32 [0, 1]
    target_size:      tuple,           # (W, H) — PIL convention
    feather_px:       int = 8,
) -> np.ndarray:
    """
    Convert the face mask tensor to a uint8 numpy mask at target_size.

    Steps:
        1. Resize mask tensor to target_size using bilinear interpolation
        2. Convert to uint8 [0, 255]
        3. Feather (Gaussian blur) the edges to avoid hard seams
           even before Poisson blending — feathering softens the
           hard composite fallback and makes Poisson blending more stable.

    Args:
        face_mask_tensor : (1, 1, H, W) float32 mask from Stage 1.
                           1 = face region (inject here), 0 = background.
        target_size      : (W, H) tuple — PIL image size convention.
        feather_px       : Gaussian blur radius in pixels for edge softening.
                           0 = no feathering (hard binary mask).
                           8 = default, softens a ~16px boundary region.

    Returns:
        mask_uint8 : (H, W) uint8 numpy array, values 0–255.
                     255 = face pixel, 0 = background pixel,
                     gradient at boundary if feather_px > 0.
    """
    W, H = target_size

    # Resize mask to match the output image resolution
    resized = F.interpolate(
        face_mask_tensor.float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, H, W)

    # Convert to uint8 numpy
    mask_np = (resized.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # Feather edges — reduces hard seam even before Poisson blending
    if feather_px > 0:
        ksize = feather_px * 2 + 1   # must be odd
        mask_np = cv2.GaussianBlur(mask_np, (ksize, ksize), sigmaX=feather_px / 2)

    return mask_np


# ══════════════════════════════════════════════════════════════════════════════
# HARD COMPOSITE
# ══════════════════════════════════════════════════════════════════════════════

def _hard_composite(
    generated_bgr: np.ndarray,   # (H, W, 3) uint8
    source_bgr:    np.ndarray,   # (H, W, 3) uint8
    mask_uint8:    np.ndarray,   # (H, W)    uint8  [0, 255]
) -> np.ndarray:
    """
    Alpha-blend generated face onto source using the face mask.

        final = source × (1 - α) + generated × α
        where α = mask / 255.0  ∈ [0, 1]

    With a feathered mask, this produces a smooth transition at the boundary.
    With a hard binary mask, the seam is visible — use Poisson blending
    to remove it.

    Args:
        generated_bgr : BGR uint8 output from the diffusion pipeline.
        source_bgr    : BGR uint8 original source image.
        mask_uint8    : (H, W) uint8 mask — face region is 255.

    Returns:
        (H, W, 3) uint8 BGR composite image.
    """
    alpha = mask_uint8.astype(np.float32) / 255.0   # (H, W) float [0, 1]
    alpha = alpha[:, :, np.newaxis]                  # (H, W, 1) for broadcast

    composite = (
        source_bgr.astype(np.float32)    * (1.0 - alpha) +
        generated_bgr.astype(np.float32) * alpha
    )
    return np.clip(composite, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# POISSON BLENDING
# ══════════════════════════════════════════════════════════════════════════════

def _poisson_blend(
    generated_bgr: np.ndarray,   # (H, W, 3) uint8 — source of interior gradients
    source_bgr:    np.ndarray,   # (H, W, 3) uint8 — destination (boundary conditions)
    mask_uint8:    np.ndarray,   # (H, W)    uint8  [0, 255]
) -> np.ndarray:
    """
    Poisson / seamless cloning to remove the color seam at the mask boundary.

    Uses cv2.seamlessClone (NORMAL_CLONE mode) which solves the Poisson
    equation: preserve gradient structure from `generated_bgr` inside the
    mask while matching color values from `source_bgr` at the boundary.

    Result: the swapped face has the reference's texture/identity but
    inherits the source's ambient lighting and color temperature at the edges,
    making the transition invisible.

    Center point strategy:
        cv2.seamlessClone requires a center point for the cloned region.
        We compute the centroid of the mask rather than the image center
        because the face is not always centered (profile shots, cropped images).
        Using the mask centroid places the clone kernel optimally.

    Fallback:
        If seamlessClone fails (e.g., mask is empty, mask touches image border,
        or OpenCV version limitation), returns the hard composite instead.
        This ensures the pipeline never crashes on edge cases — it degrades
        gracefully to the feathered alpha blend.

    Args:
        generated_bgr : BGR uint8 — the diffusion output (face region).
        source_bgr    : BGR uint8 — the original source (background).
        mask_uint8    : (H, W) uint8 mask — face region is non-zero.

    Returns:
        (H, W, 3) uint8 BGR Poisson-blended image.
    """
    # Binarize mask for seamlessClone (it requires 0/255, not gradient values)
    _, mask_binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

    # Check mask is non-empty
    if mask_binary.max() == 0:
        print("[compositing] WARNING: empty mask — returning hard composite.")
        return _hard_composite(generated_bgr, source_bgr, mask_uint8)

    # Compute mask centroid for the clone center point
    moments = cv2.moments(mask_binary)
    if moments["m00"] == 0:
        # Degenerate mask — fall back to image center
        H, W = source_bgr.shape[:2]
        center = (W // 2, H // 2)
    else:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        center = (cx, cy)

    try:
        blended = cv2.seamlessClone(
            src=generated_bgr,    # interior gradients come from here
            dst=source_bgr,       # boundary conditions come from here
            mask=mask_binary,
            p=center,
            flags=cv2.NORMAL_CLONE,
        )
        return blended

    except cv2.error as e:
        # Common cause: mask region touches image border (seamlessClone
        # requires a margin between mask and image edge).
        print(
            f"[compositing] WARNING: seamlessClone failed ({e}). "
            f"Falling back to feathered alpha composite."
        )
        return _hard_composite(generated_bgr, source_bgr, mask_uint8)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def composite_result(
    generated_pil:    Image.Image,     # PIL RGB — diffusion output
    source_pil:       Image.Image,     # PIL RGB — original source image
    face_mask_tensor: torch.Tensor,    # (1, 1, H, W) float32 [0, 1]
    cfg:              dict,
) -> Image.Image:
    """
    Composite the generated face onto the source image using the face mask.

    This is the main entry point called by stage2_diffusion.py after
    decode_latent(). It replaces the final return with a composited image
    that preserves source pixels outside the face region exactly.

    Pipeline:
        1. Prepare mask — resize to output size, feather edges
        2. Convert PIL images to BGR numpy arrays
        3. Hard composite — alpha blend using feathered mask
        4. Poisson blend  — seam removal at mask boundary (if enabled)
        5. Convert back to PIL RGB

    Config reads:
        cfg["compositing"]["poisson_blend"]  (bool)
            true  → apply Poisson seamless cloning after hard composite (default)
            false → return feathered hard composite only (debug / ablation)

        cfg["compositing"]["feather_px"]  (int, default 8)
            Gaussian blur radius for mask edge feathering.
            0 = hard binary mask (useful to verify mask boundary visually).

    Args:
        generated_pil    : PIL RGB output from decode_latent().
                           Same size as source_pil (both at target_size).
        source_pil       : PIL RGB original source image from artifacts/.
        face_mask_tensor : (1, 1, H, W) float32 mask from artifacts/face_mask.pt.
        cfg              : Full resolved config dict.

    Returns:
        PIL RGB image — reference face composited onto source background.
        Size matches generated_pil and source_pil.
    """
    # ── Read compositing config ────────────────────────────────────────────
    comp_cfg      = cfg.get("compositing", {})
    poisson_blend = comp_cfg.get("poisson_blend", True)
    feather_px    = comp_cfg.get("feather_px", 8)

    # ── Validate sizes match ───────────────────────────────────────────────
    if generated_pil.size != source_pil.size:
        print(
            f"[compositing] WARNING: size mismatch — "
            f"generated={generated_pil.size} source={source_pil.size}. "
            f"Resizing generated to match source."
        )
        generated_pil = generated_pil.resize(source_pil.size, Image.LANCZOS)

    target_size = source_pil.size   # (W, H)

    # ── 1. Prepare mask ────────────────────────────────────────────────────
    mask_uint8 = _prepare_mask(face_mask_tensor, target_size, feather_px)

    # ── 2. PIL RGB → BGR numpy ─────────────────────────────────────────────
    generated_bgr = cv2.cvtColor(np.array(generated_pil), cv2.COLOR_RGB2BGR)
    source_bgr    = cv2.cvtColor(np.array(source_pil),    cv2.COLOR_RGB2BGR)

    # ── 3. Hard composite (alpha blend with feathered mask) ────────────────
    composite_bgr = _hard_composite(generated_bgr, source_bgr, mask_uint8)

    # ── 4. Poisson blend (seam removal at boundary) ────────────────────────
    if poisson_blend:
        # Pass the hard composite as the source so Poisson blending
        # smooths the boundary of an already-reasonable composite.
        # Passing raw generated_bgr directly can over-correct if the
        # generated image has large global color shift.
        final_bgr = _poisson_blend(composite_bgr, source_bgr, mask_uint8)
    else:
        final_bgr = composite_bgr
        print("[compositing] Poisson blending disabled — returning hard composite.")

    # ── 5. BGR numpy → PIL RGB ─────────────────────────────────────────────
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)