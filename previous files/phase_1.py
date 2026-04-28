# phase1_bridge.py
"""
Bridge between Phase 1 (face alignment + Gaussian decomposition)
and Phase 3 (frequency-decomposed KV injection pipeline).

Phase 1 outputs numpy BGR arrays (OpenCV convention).
Phase 3 pipeline expects PIL RGB images and torch tensors.

This module handles all the conversions so neither Phase 1 nor Phase 3
needs to know about the other's internals.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass


@dataclass
class PipelineInputs:
    """
    Everything the pipeline needs, ready to use.

    aligned_pil   : PIL RGB — reference face warped to source pose.
                    Used as the reference for display and optionally
                    as the "whole reference" in phase2 mode.

    lf_pil        : PIL RGB — low-frequency component [0,255].
                    Encodes head shape, skin tone, proportions.

    hf_pil        : PIL RGB — high-frequency component, shifted to [0,255].
                    Encodes iris detail, pores, fine texture.
                    (Original HF range is [-255,255]; shifted by +128 for VAE.)

    face_mask     : torch.Tensor (1,1,H,W) float32 [0,1].
                    Spatial mask for KV injection — 1 inside face, 0 outside.

    yaw_diff_deg  : float — rough yaw difference between source and reference.
                    If > 35°, warn the user; results will degrade.

    source_pil    : PIL RGB — source image (for display only).
    """
    aligned_pil  : Image.Image
    lf_pil       : Image.Image
    hf_pil       : Image.Image
    face_mask    : torch.Tensor     # (1,1,H,W) float32
    yaw_diff_deg : float
    source_pil   : Image.Image


def phase1_to_pipeline(
    source_bgr   : np.ndarray,     # Phase 1 source input (uint8 BGR)
    phase1_result,                  # DecomposeResult from align_and_decompose()
    target_size  : int = 512,
) -> PipelineInputs:
    """
    Convert Phase 1 DecomposeResult into pipeline-ready inputs.

    Args:
        source_bgr   : Original source image in BGR uint8 (for display).
        phase1_result: Output of align_and_decompose() from Phase 1.
        target_size  : Resize all images to this square size for SD.
                       Use 512 for SD2.1-base, 768 for SD2.1.

    Returns:
        PipelineInputs with everything the pipeline needs.

    HF shift convention:
        Phase 1 HF is in [-255, 255]. We add 128 to shift to [0, 255]
        before converting to PIL and passing to the VAE encoder. This is
        lossless in terms of spatial structure — the VAE KV features still
        capture all texture patterns. The shift is NOT reversed during
        injection; it simply means HF latents encode a "shifted texture map"
        rather than a pure residual. This is intentional.
    """
    S = target_size

    # ── 1. Aligned reference face ─────────────────────────────────────────────
    aligned_rgb = cv2.cvtColor(phase1_result.aligned_face, cv2.COLOR_BGR2RGB)
    aligned_pil = Image.fromarray(aligned_rgb).resize((S, S), Image.LANCZOS)

    # ── 2. LF component ───────────────────────────────────────────────────────
    # LF is float32 [0, 255] BGR — just clip, cast, convert
    lf_uint8 = np.clip(phase1_result.LF, 0, 255).astype(np.uint8)
    lf_rgb   = cv2.cvtColor(lf_uint8, cv2.COLOR_BGR2RGB)
    lf_pil   = Image.fromarray(lf_rgb).resize((S, S), Image.LANCZOS)

    # ── 3. HF component (shift from [-255,255] → [0,255]) ────────────────────
    hf_shifted = np.clip(phase1_result.HF + 128.0, 0, 255).astype(np.uint8)
    hf_rgb     = cv2.cvtColor(hf_shifted, cv2.COLOR_BGR2RGB)
    hf_pil     = Image.fromarray(hf_rgb).resize((S, S), Image.LANCZOS)

    # ── 4. Face mask → (1,1,H,W) float32 tensor ─────────────────────────────
    # Phase 1 mask is uint8 (H,W) with values 0/255.
    # Resize to target_size, normalize to [0,1].
    mask_resized = cv2.resize(
        phase1_result.face_mask, (S, S), interpolation=cv2.INTER_LINEAR
    )
    mask_tensor = torch.from_numpy(mask_resized.astype(np.float32) / 255.0)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)   # (1,1,S,S)

    # ── 5. Source for display ─────────────────────────────────────────────────
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    source_pil = Image.fromarray(source_rgb).resize((S, S), Image.LANCZOS)

    # ── 6. Warn on large yaw ─────────────────────────────────────────────────
    if phase1_result.yaw_diff > 35.0:
        print(
            f"[phase1_bridge] WARNING: yaw_diff={phase1_result.yaw_diff:.1f}° > 35°. "
            f"Affine warp is unreliable at this pose difference. "
            f"Consider using a reference image with a closer head angle to the source."
        )

    return PipelineInputs(
        aligned_pil  = aligned_pil,
        lf_pil       = lf_pil,
        hf_pil       = hf_pil,
        face_mask    = mask_tensor,
        yaw_diff_deg = phase1_result.yaw_diff,
        source_pil   = source_pil,
    )


def encode_pil_to_latent(vae, pil_image: Image.Image, image_processor) -> torch.Tensor:
    """
    Encode a single PIL image to a VAE latent tensor.

    Args:
        vae           : pipe.vae
        pil_image     : PIL RGB image, already at the target resolution
        image_processor: pipe.image_processor (handles [-1,1] normalization)

    Returns:
        latent: (1, 4, H/8, W/8) float16 on the same device as vae
    """
    device    = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    pixel = image_processor.preprocess(pil_image).to(device, dtype=vae_dtype)
    with torch.no_grad():
        latent = vae.encode(pixel).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    return latent