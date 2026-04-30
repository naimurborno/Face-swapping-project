# core/decomposition.py
"""
Frequency decomposition and mixed-frequency prior construction for the
Mixed-Frequency Prior Guided Inpainting pipeline.

This module has two responsibilities:

1. FREQUENCY DECOMPOSITION (unchanged from original)
   Split any image into a low-frequency (LF) component and a high-frequency
   (HF) residual using Gaussian blur, FFT rectangular mask, or no split.
   The ablation.decomposition flag in configs/default.yaml selects the method.

2. MIXED-FREQUENCY PRIOR CONSTRUCTION (new)
   Given a content image S and an aligned donor image R̃, synthesize a
   photometrically plausible prior image P inside the editable mask region:

       P = α·S_LF + β·R̃_LF + γ·R̃_HF

   where:
       S_LF  — source low frequency: spatial placement, coarse geometry,
               illumination field. Keeps the output spatially compatible
               with the surrounding source context.
       R̃_LF  — donor low frequency: donor coarse semantic appearance,
               color palette, material tone.
       R̃_HF  — donor high frequency: fine surface detail, texture,
               microstructure, material identity.

   The prior is then embedded into the source image inside the mask:

       X₀ = (1 − M) ⊙ S + M ⊙ P

   X₀ becomes the inpainting model input. Outside the mask: exact source
   pixels. Inside: the mixed prior that tells the diffusion model what
   should emerge.

   CRITICAL — PHOTOMETRIC PLAUSIBILITY:
   P = α·S_LF + β·R̃_LF + γ·R̃_HF is not naturally a photorealistic image.
   If it is too unnatural, the inpainting model treats it as corruption
   and invents a plausible image from scratch, ignoring the prior entirely.
   To prevent this, histogram_match_to_source() normalizes P's tone and
   color statistics to match the surrounding source region before embedding.

Decomposition method selection (ablation.decomposition in default.yaml):
    "gaussian" — Gaussian blur split (default). Smooth rolloff, no ringing.
    "fft"      — FFT rectangular mask split (ablation A1). Introduces Gibbs
                 ringing — used only to quantify degradation vs Gaussian.
    "none"     — No split; image returned as both LF and HF (ablation A1).
                 Tests whether the frequency split itself contributes.

Public API:
    DecomposeResult          — NamedTuple: LF, HF, method, lf_energy, hf_std
    PriorResult              — NamedTuple: P, X0, S_LF, S_HF, R_LF, R_HF, α, β, γ

    decompose()              — BGR numpy → DecomposeResult
    decompose_pil()          — PIL RGB → (LF, HF) BGR numpy (used by compositing)
    decompose_gaussian()     — Gaussian split primitive
    decompose_fft()          — FFT split primitive

    build_prior()            — (S, R̃, M, α, β, γ) → PriorResult   [NEW]
    build_masked_input()     — embed P into S under mask → X₀       [NEW]
    histogram_match_to_source() — normalize P tone to match S        [NEW]

    to_pil_inputs()          — DecomposeResult → PIL images for VAE encoding
    mask_to_tensor()         — uint8 mask → (1,1,H,W) float32 tensor
    build_chimera()          — source_LF + aligned_HF (kept for ablation A1)

Dependencies:
    pip install opencv-python-headless numpy torch pillow
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Tuple
from PIL import Image


# ── Output containers ─────────────────────────────────────────────────────────

class DecomposeResult(NamedTuple):
    """
    Output of decompose(). All arrays are float32 in BGR channel order.

    LF        : (H, W, 3) float32, range [0, 255]
                Low-frequency component: coarse structure, color, geometry.
                For content image S  → spatial placement, illumination.
                For donor image R̃   → coarse semantic appearance, material tone.

    HF        : (H, W, 3) float32, range [-255, 255]  (Gaussian / FFT)
                or (H, W, 3) float32, range [0, 255]   (method="none")
                High-frequency residual: fine detail, texture, microstructure.
                For donor image R̃   → the primary attribute transfer signal.
                For method="none"   → HF == LF == original image (no split).

    method    : str — "gaussian" | "fft" | "none"
    lf_energy : float — mean squared energy of LF (sanity check)
    hf_std    : float — std of HF (expect > 10 for textured images;
                very low std means the image is smooth / low texture)
    """
    LF        : np.ndarray
    HF        : np.ndarray
    method    : str
    lf_energy : float
    hf_std    : float


class PriorResult(NamedTuple):
    """
    Output of build_prior(). All images are uint8 BGR unless noted.

    P         : (H, W, 3) uint8 BGR — mixed-frequency prior image.
                Photometrically normalized. Intended for embedding inside mask.
                P = clip(histogram_match(α·S_LF + β·R̃_LF + γ·R̃_HF))

    X0        : (H, W, 3) uint8 BGR — masked input image for inpainting.
                X₀ = (1 − M) ⊙ S + M ⊙ P
                Outside mask: exact source pixels.
                Inside mask: mixed-frequency prior.

    S_LF      : (H, W, 3) float32 — source low frequency
    S_HF      : (H, W, 3) float32 — source high frequency (not used in prior,
                retained for compositing and ablation logging)
    R_LF      : (H, W, 3) float32 — donor low frequency
    R_HF      : (H, W, 3) float32 — donor high frequency
    alpha     : float — S_LF weight used
    beta      : float — R̃_LF weight used
    gamma     : float — R̃_HF weight used
    """
    P     : np.ndarray
    X0    : np.ndarray
    S_LF  : np.ndarray
    S_HF  : np.ndarray
    R_LF  : np.ndarray
    R_HF  : np.ndarray
    alpha : float
    beta  : float
    gamma : float


# ── Gaussian decomposition ────────────────────────────────────────────────────

def decompose_gaussian(
    image_bgr: np.ndarray,   # (H, W, 3) uint8 or float32
    kernel:    int   = 31,
    sigma:     float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image into LF and HF via Gaussian blur.

        LF = GaussianBlur(image)
        HF = image − LF

    Args:
        image_bgr : BGR image, uint8 or float32.
        kernel    : Gaussian kernel size. Must be odd. Default 31.
                    Larger → smoother LF, more detail pushed into HF.
                    Recommended range: 21–51 for 512px images.
        sigma     : Gaussian sigma. Default 5.0.
                    Larger → more LF bleed into mid-frequency range.

    Returns:
        LF : float32 [0, 255]      — structure / coarse appearance
        HF : float32 [-255, 255]   — texture / detail residual

    Reconstruction: LF + HF == original (exact up to float32 precision).
    """
    assert kernel % 2 == 1, (
        f"[decomposition] Gaussian kernel must be odd, got {kernel}. "
        f"Use 31 (default), 21, 41, or 51."
    )
    img_f = image_bgr.astype(np.float32)
    LF    = cv2.GaussianBlur(img_f, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    HF    = img_f - LF
    return LF, HF


# ── FFT decomposition ─────────────────────────────────────────────────────────

def decompose_fft(
    image_bgr:    np.ndarray,    # (H, W, 3) uint8 or float32
    cutoff_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image into LF and HF via FFT rectangular mask.

        LF = IFFT(FFT(image) * rectangular_mask_centred_at_DC)
        HF = image − LF

    Args:
        image_bgr    : BGR image, uint8 or float32.
        cutoff_ratio : Fraction of half-spatial-frequency radius treated as LF.
                       0.1 = inner 10% of the spectrum (good default at 512px).

    Returns:
        LF : float32 [0, 255]
        HF : float32 [-255, 255]

    WARNING — Ablation use only:
        FFT with a rectangular mask causes Gibbs ringing on natural images.
        Use Gaussian for all non-ablation runs.
    """
    img_f    = image_bgr.astype(np.float32)
    H, W, C  = img_f.shape
    LF       = np.zeros_like(img_f)

    cy, cx = H // 2, W // 2
    rh = max(1, int(cutoff_ratio * H / 2))
    rw = max(1, int(cutoff_ratio * W / 2))

    mask = np.zeros((H, W), dtype=np.float32)
    mask[cy - rh: cy + rh, cx - rw: cx + rw] = 1.0

    for c in range(C):
        channel = img_f[:, :, c]
        f       = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f)
        f_lf    = f_shift * mask
        f_lf_us = np.fft.ifftshift(f_lf)
        lf_c    = np.fft.ifft2(f_lf_us).real.astype(np.float32)
        LF[:, :, c] = lf_c

    HF = img_f - LF
    return LF, HF


# ── Main decompose entry point ────────────────────────────────────────────────

def decompose(
    image_bgr:    np.ndarray,
    method:       str   = "gaussian",
    kernel:       int   = 31,
    sigma:        float = 5.0,
    cutoff_ratio: float = 0.1,
) -> DecomposeResult:
    """
    Decompose any BGR image into LF and HF components.

    Used for both content image S and aligned donor image R̃ in Stage 1.
    Also used by compositing.py for frequency-decomposed blending.

    Args:
        image_bgr    : BGR uint8 image — content or donor image.
        method       : "gaussian" | "fft" | "none"
        kernel       : Gaussian kernel size (method="gaussian" only).
        sigma        : Gaussian sigma (method="gaussian" only).
        cutoff_ratio : FFT cutoff fraction (method="fft" only).

    Returns:
        DecomposeResult(LF, HF, method, lf_energy, hf_std)
    """
    if method not in ("gaussian", "fft", "none"):
        raise ValueError(
            f"[decomposition] Unknown method '{method}'. "
            f"Choose: 'gaussian' | 'fft' | 'none'"
        )

    if method == "none":
        img_f = image_bgr.astype(np.float32)
        LF    = img_f.copy()
        HF    = img_f.copy()
        print("[decomposition] method='none' — LF == HF == original (no split).")

    elif method == "gaussian":
        LF, HF = decompose_gaussian(image_bgr, kernel=kernel, sigma=sigma)
        print(
            f"[decomposition] method='gaussian' | "
            f"kernel={kernel} sigma={sigma}"
        )

    else:   # "fft"
        LF, HF = decompose_fft(image_bgr, cutoff_ratio=cutoff_ratio)
        print(
            f"[decomposition] method='fft' | "
            f"cutoff_ratio={cutoff_ratio}"
        )

    lf_energy = float(np.mean(LF ** 2))
    hf_std    = float(np.std(HF))

    print(
        f"[decomposition] lf_energy={lf_energy:.1f} | "
        f"hf_std={hf_std:.3f}"
    )

    return DecomposeResult(
        LF        = LF,
        HF        = HF,
        method    = method,
        lf_energy = lf_energy,
        hf_std    = hf_std,
    )


# ── PIL decomposition helper ──────────────────────────────────────────────────

def decompose_pil(
    pil_image:    Image.Image,
    method:       str   = "gaussian",
    kernel:       int   = 31,
    sigma:        float = 5.0,
    cutoff_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: PIL RGB image → (LF, HF) as BGR float32 numpy arrays.

    Called by compositing.py which works in BGR numpy throughout so it can
    call cv2 operations (seamlessClone, GaussianBlur) without extra conversions.
    Also used internally by build_prior() when PIL inputs are provided.

    Args:
        pil_image    : PIL RGB image at target resolution (512 or 768px).
        method       : "gaussian" | "fft" | "none"
        kernel       : Gaussian kernel size (method="gaussian" only).
        sigma        : Gaussian sigma (method="gaussian" only).
        cutoff_ratio : FFT cutoff (method="fft" only).

    Returns:
        LF : (H, W, 3) float32 BGR, range [0, 255]
        HF : (H, W, 3) float32 BGR, range [-255, 255]
             (or [0, 255] if method="none")

    Example usage in compositing.py:
        source_LF,    source_HF    = decompose_pil(source_pil,    method)
        generated_LF, generated_HF = decompose_pil(generated_pil, method)
        final = final_LF + final_HF
    """
    rgb_array = np.array(pil_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    result    = decompose(
        bgr_array,
        method       = method,
        kernel       = kernel,
        sigma        = sigma,
        cutoff_ratio = cutoff_ratio,
    )
    return result.LF, result.HF


# ── Histogram matching ────────────────────────────────────────────────────────

def histogram_match_to_source(
    prior_bgr:   np.ndarray,   # (H, W, 3) float32 — raw prior P before clipping
    source_bgr:  np.ndarray,   # (H, W, 3) uint8   — full source image S
    mask:        np.ndarray,   # (H, W)    uint8   — binary mask, 255 = editable
) -> np.ndarray:
    """
    Normalize the prior image's tone and color statistics to match the
    surrounding source region, making P photometrically compatible with S.

    WHY THIS IS CRITICAL:
    P = α·S_LF + β·R̃_LF + γ·R̃_HF is not a photorealistic image.
    Its mean brightness and color balance may differ significantly from
    the source. If the inpainting model sees a masked region whose color
    statistics are wildly different from the surrounding source pixels, it
    treats P as corruption and ignores it, generating a plausible but
    donor-unrelated completion.

    Histogram matching constrains P to the same perceptual register as the
    source — same overall brightness, same color temperature — so the model
    interprets P as a meaningful structural hint rather than noise.

    METHOD:
    Per-channel linear normalization in LAB color space:
        1. Convert P and source_outside_mask to LAB
        2. Match P's L, A, B channel statistics (mean + std) to source
        3. Convert back to BGR

    This is a Reinhard-style color transfer, applied selectively to P.

    Args:
        prior_bgr  : (H, W, 3) float32 — raw prior before clip.
                     Output of α·S_LF + β·R̃_LF + γ·R̃_HF.
        source_bgr : (H, W, 3) uint8   — full source image S.
        mask       : (H, W) uint8 binary mask, 255 inside editable region.
                     Used to extract source statistics from OUTSIDE the mask
                     (the boundary context the model will see as ground truth).

    Returns:
        (H, W, 3) float32 — histogram-matched prior, ready for clipping.
    """
    # Clip prior to valid range before LAB conversion
    P_uint8 = np.clip(prior_bgr, 0, 255).astype(np.uint8)

    # Extract source pixels OUTSIDE mask — these are the ground truth context
    outside_mask = (mask == 0)
    if outside_mask.sum() < 100:
        # Mask covers almost the entire image — fall back to full source stats
        outside_mask = np.ones_like(mask, dtype=bool)

    # Convert both to LAB
    P_lab  = cv2.cvtColor(P_uint8,                   cv2.COLOR_BGR2LAB).astype(np.float32)
    S_lab  = cv2.cvtColor(source_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute source statistics from the outside-mask region only
    P_out  = np.zeros_like(P_lab)

    for c in range(3):
        p_chan    = P_lab[:, :, c]
        s_outside = S_lab[:, :, c][outside_mask]

        p_mean, p_std = p_chan.mean(), p_chan.std() + 1e-6
        s_mean, s_std = s_outside.mean(), s_outside.std() + 1e-6

        # Linear rescale: shift P's statistics to match source
        p_matched = (p_chan - p_mean) * (s_std / p_std) + s_mean
        P_out[:, :, c] = p_matched

    # Convert matched LAB back to BGR float32
    P_out_uint8  = np.clip(P_out, 0, 255).astype(np.uint8)
    P_matched_bgr = cv2.cvtColor(P_out_uint8, cv2.COLOR_LAB2BGR).astype(np.float32)

    return P_matched_bgr


# ── Masked input construction ─────────────────────────────────────────────────

def build_masked_input(
    source_bgr: np.ndarray,   # (H, W, 3) uint8 — content image S
    prior_bgr:  np.ndarray,   # (H, W, 3) uint8 — mixed prior P
    mask:       np.ndarray,   # (H, W)    uint8 — binary, 255 = editable
) -> np.ndarray:
    """
    Embed the mixed-frequency prior inside the mask to form the inpainting input.

        X₀ = (1 − M) ⊙ S + M ⊙ P

    Outside mask: exact source pixels — the inpainting model sees real context.
    Inside mask:  mixed-frequency prior — the structural hint for completion.

    Args:
        source_bgr : (H, W, 3) uint8 — full source image S.
        prior_bgr  : (H, W, 3) uint8 — histogram-matched prior P.
        mask       : (H, W) uint8, values 0 or 255.

    Returns:
        (H, W, 3) uint8 — masked input image X₀.
    """
    M_3ch = (mask[:, :, np.newaxis] / 255.0).astype(np.float32)   # (H, W, 1) [0,1]
    S_f   = source_bgr.astype(np.float32)
    P_f   = prior_bgr.astype(np.float32)

    X0_f  = (1.0 - M_3ch) * S_f + M_3ch * P_f
    X0    = np.clip(X0_f, 0, 255).astype(np.uint8)

    print(
        f"[decomposition] Masked input X₀ built | "
        f"mask coverage={(mask > 0).mean() * 100:.1f}%"
    )
    return X0


# ── Mixed-frequency prior construction ───────────────────────────────────────

def build_prior(
    source_bgr:       np.ndarray,          # (H, W, 3) uint8 — content image S
    donor_aligned_bgr: np.ndarray,         # (H, W, 3) uint8 — donor R̃ aligned to S
    mask:             np.ndarray,          # (H, W)    uint8 — binary, 255 = editable
    alpha:            float = 0.6,         # S_LF weight
    beta:             float = 0.5,         # R̃_LF weight
    gamma:            float = 0.8,         # R̃_HF weight
    method:           str   = "gaussian",  # decomposition method
    kernel:           int   = 31,
    sigma:            float = 5.0,
    cutoff_ratio:     float = 0.1,
    histogram_match:  bool  = True,
) -> PriorResult:
    """
    Build the mixed-frequency prior P and the masked input image X₀.

    This is the core of Stage 1 in the new pipeline. The prior encodes:
        — where the content is  (from S_LF, weighted by α)
        — what the donor looks like coarsely  (from R̃_LF, weighted by β)
        — what fine detail the donor has  (from R̃_HF, weighted by γ)

    Mathematical formulation:
        S_LF, S_HF = decompose(S)
        R̃_LF, R̃_HF = decompose(R̃)
        P_raw = α·S_LF + β·R̃_LF + γ·R̃_HF
        P     = histogram_match(clip(P_raw), S, mask)   if histogram_match
              = clip(P_raw)                              otherwise
        X₀    = (1 − M) ⊙ S + M ⊙ P

    COEFFICIENT GUIDANCE:
        α = 0.6  (default)
            Controls source structural compatibility inside the mask.
            High α → output geometry stays close to S (retexturing tasks).
            Low  α → donor geometry can override S (identity transfer tasks).

        β = 0.5  (default)
            Controls donor coarse appearance transfer.
            High β → donor color palette and tone dominate the masked region.
            Low  β → prior color stays closer to source.
            Note: β interacts with α — if α + β >> 1, P saturates.
            Safe range: α + β ≤ 1.2 before histogram matching corrects it.

        γ = 0.8  (default)
            Controls donor fine detail strength in the prior.
            High γ → strong texture hint visible in P, diffusion reinforces it.
            Low  γ → diffusion invents detail (less faithful to donor).
            γ should generally be the highest weight — R̃_HF is the primary
            attribute transfer signal and the inpainting model needs a clear hint.

    Args:
        source_bgr        : (H, W, 3) uint8 — content image S.
        donor_aligned_bgr : (H, W, 3) uint8 — donor image R̃ already aligned to S.
        mask              : (H, W) uint8 binary mask, 255 = editable region.
        alpha             : S_LF weight. Default 0.6.
        beta              : R̃_LF weight. Default 0.5.
        gamma             : R̃_HF weight. Default 0.8.
        method            : Decomposition method. Default "gaussian".
        kernel            : Gaussian kernel size. Default 31.
        sigma             : Gaussian sigma. Default 5.0.
        cutoff_ratio      : FFT cutoff ratio. Default 0.1.
        histogram_match   : If True, normalize P's statistics to match source
                            context before embedding. Strongly recommended.
                            Set False only for ablation A4 comparison.

    Returns:
        PriorResult(P, X0, S_LF, S_HF, R_LF, R_HF, alpha, beta, gamma)

    Raises:
        ValueError if source and donor have different spatial sizes.
        ValueError if mask spatial size does not match source.
    """
    # ── Input validation ──────────────────────────────────────────────────
    if source_bgr.shape[:2] != donor_aligned_bgr.shape[:2]:
        raise ValueError(
            f"[decomposition] Source shape {source_bgr.shape[:2]} != "
            f"donor shape {donor_aligned_bgr.shape[:2]}. "
            f"Run alignment.run_donor_alignment() before build_prior()."
        )
    if source_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"[decomposition] Source shape {source_bgr.shape[:2]} != "
            f"mask shape {mask.shape[:2]}. "
            f"Resize mask to match source before calling build_prior()."
        )

    # ── Decompose source S ────────────────────────────────────────────────
    print("[decomposition] Decomposing source image S...")
    src_result = decompose(
        source_bgr,
        method       = method,
        kernel       = kernel,
        sigma        = sigma,
        cutoff_ratio = cutoff_ratio,
    )
    S_LF = src_result.LF    # float32 [0, 255]  — source geometry anchor
    S_HF = src_result.HF    # float32 [-255,255] — retained for compositing/logging

    # ── Decompose aligned donor R̃ ─────────────────────────────────────────
    print("[decomposition] Decomposing aligned donor image R̃...")
    donor_result = decompose(
        donor_aligned_bgr,
        method       = method,
        kernel       = kernel,
        sigma        = sigma,
        cutoff_ratio = cutoff_ratio,
    )
    R_LF = donor_result.LF    # float32 [0, 255]  — donor coarse appearance
    R_HF = donor_result.HF    # float32 [-255,255] — donor fine detail (primary signal)

    # ── Build raw prior ───────────────────────────────────────────────────
    # P_raw = α·S_LF + β·R̃_LF + γ·R̃_HF
    # All three terms are float32. R̃_HF is in [-255, 255] so the sum may
    # exceed [0, 255] — clipping and histogram matching correct this.
    P_raw = alpha * S_LF + beta * R_LF + gamma * R_HF   # float32, unconstrained

    print(
        f"[decomposition] Prior built | "
        f"α={alpha} β={beta} γ={gamma} | "
        f"P_raw range=[{P_raw.min():.1f}, {P_raw.max():.1f}]"
    )

    # ── Histogram matching ────────────────────────────────────────────────
    if histogram_match:
        P_matched = histogram_match_to_source(P_raw, source_bgr, mask)
        print("[decomposition] Histogram matching applied.")
    else:
        P_matched = P_raw
        print("[decomposition] Histogram matching skipped (ablation mode).")

    # ── Clip to valid uint8 range ─────────────────────────────────────────
    P = np.clip(P_matched, 0, 255).astype(np.uint8)

    # ── Build masked input X₀ ─────────────────────────────────────────────
    X0 = build_masked_input(source_bgr, P, mask)

    print(
        f"[decomposition] PriorResult ready | "
        f"P mean={P.mean():.1f} std={P.std():.2f} | "
        f"X₀ mean={X0.mean():.1f}"
    )

    return PriorResult(
        P     = P,
        X0    = X0,
        S_LF  = S_LF,
        S_HF  = S_HF,
        R_LF  = R_LF,
        R_HF  = R_HF,
        alpha = alpha,
        beta  = beta,
        gamma = gamma,
    )


# ── PIL conversion helper ─────────────────────────────────────────────────────

def to_pil_inputs(
    result:      DecomposeResult,
    image_bgr:   np.ndarray,
    target_size: int = 512,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Convert a DecomposeResult into PIL RGB images ready for VAE encoding.

    Called by stage1_segment.py after decompose() returns, and for logging
    intermediate LF/HF visualizations to artifacts/.

    HF shift convention:
        Gaussian and FFT HF are in [-255, 255]. Adding 128 shifts them to
        [0, 255] for PIL / uint8 compatibility. Stage 2 does NOT reverse
        this shift — the shifted map is used for visualization only.
        Exception: method="none" — HF is already [0, 255], no shift needed.

    Args:
        result      : DecomposeResult from decompose().
        image_bgr   : uint8 BGR image that was decomposed.
        target_size : Resize all outputs to this square size.

    Returns:
        image_pil : PIL RGB — original image (content or donor)
        lf_pil    : PIL RGB — LF component [0, 255]
        hf_pil    : PIL RGB — HF component shifted to [0, 255] for visualization
    """
    S = target_size

    img_rgb   = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb).resize((S, S), Image.LANCZOS)

    lf_uint8 = np.clip(result.LF, 0.0, 255.0).astype(np.uint8)
    lf_rgb   = cv2.cvtColor(lf_uint8, cv2.COLOR_BGR2RGB)
    lf_pil   = Image.fromarray(lf_rgb).resize((S, S), Image.LANCZOS)

    if result.method == "none":
        hf_shifted = np.clip(result.HF, 0.0, 255.0).astype(np.uint8)
    else:
        hf_shifted = np.clip(result.HF + 128.0, 0.0, 255.0).astype(np.uint8)

    hf_rgb = cv2.cvtColor(hf_shifted, cv2.COLOR_BGR2RGB)
    hf_pil = Image.fromarray(hf_rgb).resize((S, S), Image.LANCZOS)

    return image_pil, lf_pil, hf_pil


def prior_result_to_pil(
    prior_result: PriorResult,
    target_size:  int = 512,
) -> Tuple[Image.Image, Image.Image]:
    """
    Convert a PriorResult into PIL RGB images for saving to artifacts/.

    Args:
        prior_result : PriorResult from build_prior().
        target_size  : Resize to this square size.

    Returns:
        prior_pil  : PIL RGB — the mixed prior P
        x0_pil     : PIL RGB — the masked input X₀
    """
    S = target_size

    P_rgb     = cv2.cvtColor(prior_result.P,  cv2.COLOR_BGR2RGB)
    prior_pil = Image.fromarray(P_rgb).resize((S, S), Image.LANCZOS)

    X0_rgb = cv2.cvtColor(prior_result.X0, cv2.COLOR_BGR2RGB)
    x0_pil = Image.fromarray(X0_rgb).resize((S, S), Image.LANCZOS)

    return prior_pil, x0_pil


# ── Torch tensor helper ───────────────────────────────────────────────────────

def mask_to_tensor(
    mask_uint8:  np.ndarray,
    target_size: int = 512,
) -> torch.Tensor:
    """
    Convert a segmentation mask to a (1, 1, H, W) float32 tensor in [0, 1].

    Called by stage1_segment.py when building the artifacts bundle.
    Also used by stage2_diffusion.py to produce Mz (latent-space mask)
    via F.interpolate to 64×64 for blended latent anchoring.

    Args:
        mask_uint8   : Raw uint8 mask from get_object_mask(). Values 0 or 255.
        target_size  : Resize to match pipeline spatial resolution (512px).

    Returns:
        (1, 1, target_size, target_size) float32 tensor, values in [0, 1].
    """
    mask_resized = cv2.resize(
        mask_uint8,
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = torch.from_numpy(mask_resized.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0).unsqueeze(0)   # (1, 1, S, S)


# ── Chimera builder (retained for ablation A1) ────────────────────────────────

def build_chimera(
    source_bgr:   np.ndarray,   # (H, W, 3) uint8 — content/source image
    aligned_bgr:  np.ndarray,   # (H, W, 3) uint8 — donor warped to source pose
    method:       str   = "gaussian",
    kernel:       int   = 31,
    sigma:        float = 5.0,
    cutoff_ratio: float = 0.1,
) -> np.ndarray:
    """
    Build chimera image: source_LF + aligned_HF.

    RETAINED FOR ABLATION A1 ONLY.
    In the new pipeline, build_prior() replaces build_chimera() as the
    primary prior construction method. build_chimera() is the old Phase 3
    approach (source_LF + donor_HF only, no donor_LF term, no histogram
    matching) and serves as the ablation A1 baseline:

        ablation.prior_construction = "chimera"  → uses this function
        ablation.prior_construction = "full"     → uses build_prior()
        ablation.prior_construction = "none"     → no prior, pure inpainting

    Args:
        source_bgr  : (H, W, 3) uint8 — content image S.
        aligned_bgr : (H, W, 3) uint8 — donor image R̃ aligned to S.
        method      : Decomposition method.
        kernel      : Gaussian kernel size.
        sigma       : Gaussian sigma.
        cutoff_ratio: FFT cutoff ratio.

    Returns:
        chimera_bgr : (H, W, 3) uint8 — source_LF + aligned_HF, clipped.
    """
    src_result = decompose(
        source_bgr,
        method       = method,
        kernel       = kernel,
        sigma        = sigma,
        cutoff_ratio = cutoff_ratio,
    )
    source_LF = src_result.LF    # float32 [0, 255]

    ref_result = decompose(
        aligned_bgr,
        method       = method,
        kernel       = kernel,
        sigma        = sigma,
        cutoff_ratio = cutoff_ratio,
    )
    aligned_HF = ref_result.HF   # float32 [-255, 255]

    chimera = source_LF + aligned_HF
    chimera = np.clip(chimera, 0, 255).astype(np.uint8)

    print(
        f"[decomposition] Chimera built (ablation A1 baseline) | "
        f"source_LF mean={source_LF.mean():.1f} | "
        f"aligned_HF std={aligned_HF.std():.2f}"
    )
    return chimera


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    """
    Verify all decomposition methods, prior construction, and helpers.
    Runs on CPU, no GPU or real images needed.
    """
    print("[decomposition] Running smoke test...\n")

    rng           = np.random.default_rng(42)
    dummy_source  = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    dummy_donor   = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    dummy_mask    = np.zeros((512, 512), dtype=np.uint8)
    dummy_mask[128:384, 128:384] = 255   # centre square

    dummy_pil = Image.fromarray(cv2.cvtColor(dummy_source, cv2.COLOR_BGR2RGB))

    # ── Test decompose() ──────────────────────────────────────────────────
    for method in ("gaussian", "fft", "none"):
        result = decompose(dummy_source, method=method)

        assert result.LF.dtype == np.float32,   f"[{method}] LF dtype wrong"
        assert result.HF.dtype == np.float32,   f"[{method}] HF dtype wrong"
        assert result.LF.shape == (512, 512, 3), f"[{method}] LF shape wrong"

        if method != "none":
            recon = result.LF.astype(np.float64) + result.HF.astype(np.float64)
            orig  = dummy_source.astype(np.float64)
            err   = np.abs(recon - orig).max()
            assert err < 0.5, f"[{method}] Reconstruction error: {err:.4f}"
            print(f"  decompose()     method={method} | recon_err={err:.2e} ✓")
        else:
            print(f"  decompose()     method={method} | LF==HF==original ✓")

    # ── Test decompose_pil() ──────────────────────────────────────────────
    LF_pil, HF_pil = decompose_pil(dummy_pil, method="gaussian")
    assert LF_pil.shape == (512, 512, 3)
    assert HF_pil.shape == (512, 512, 3)
    print(f"  decompose_pil() method=gaussian | shape={LF_pil.shape} ✓")

    # ── Test histogram_match_to_source() ─────────────────────────────────
    dummy_prior_f = dummy_source.astype(np.float32) * 1.5 + 30.0
    matched       = histogram_match_to_source(dummy_prior_f, dummy_source, dummy_mask)
    assert matched.shape == (512, 512, 3)
    assert matched.dtype == np.float32
    print(f"  histogram_match_to_source() | shape={matched.shape} ✓")

    # ── Test build_masked_input() ─────────────────────────────────────────
    P_uint8 = np.clip(matched, 0, 255).astype(np.uint8)
    X0      = build_masked_input(dummy_source, P_uint8, dummy_mask)
    assert X0.shape == (512, 512, 3)
    assert X0.dtype == np.uint8
    # Outside mask must equal source exactly
    outside = dummy_mask == 0
    assert np.all(X0[outside] == dummy_source[outside]), \
        "build_masked_input: outside-mask pixels must equal source"
    print(f"  build_masked_input() | shape={X0.shape} | outside==source ✓")

    # ── Test build_prior() ────────────────────────────────────────────────
    for hm in (True, False):
        pr = build_prior(
            dummy_source, dummy_donor, dummy_mask,
            alpha=0.6, beta=0.5, gamma=0.8,
            histogram_match=hm,
        )
        assert pr.P.dtype  == np.uint8
        assert pr.X0.dtype == np.uint8
        assert pr.P.shape  == (512, 512, 3)
        assert pr.X0.shape == (512, 512, 3)
        assert np.all(pr.X0[dummy_mask == 0] == dummy_source[dummy_mask == 0]), \
            "build_prior: X0 outside mask must equal source"
        print(f"  build_prior() histogram_match={hm} | P={pr.P.shape} X0={pr.X0.shape} ✓")

    # ── Test prior_result_to_pil() ────────────────────────────────────────
    pr         = build_prior(dummy_source, dummy_donor, dummy_mask)
    prior_pil, x0_pil = prior_result_to_pil(pr, target_size=512)
    assert prior_pil.size == (512, 512)
    assert x0_pil.size    == (512, 512)
    print(f"  prior_result_to_pil() | prior={prior_pil.size} x0={x0_pil.size} ✓")

    # ── Test to_pil_inputs() ─────────────────────────────────────────────
    result = decompose(dummy_source, method="gaussian")
    img_pil, lf_pil, hf_pil = to_pil_inputs(result, dummy_source, target_size=512)
    assert lf_pil.size == (512, 512)
    assert hf_pil.size == (512, 512)
    print(f"  to_pil_inputs() | lf={lf_pil.size} hf={hf_pil.size} ✓")

    # ── Test mask_to_tensor() ─────────────────────────────────────────────
    t = mask_to_tensor(dummy_mask, target_size=512)
    assert t.shape == (1, 1, 512, 512)
    assert t.dtype == torch.float32
    assert t.min() >= 0.0 and t.max() <= 1.0
    print(f"  mask_to_tensor() | shape={tuple(t.shape)} ✓")

    # ── Test build_chimera() (ablation baseline) ──────────────────────────
    chimera = build_chimera(dummy_source, dummy_donor, method="gaussian")
    assert chimera.shape == (512, 512, 3)
    assert chimera.dtype == np.uint8
    print(f"  build_chimera() | shape={chimera.shape} ✓")

    print("\n[decomposition] All smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()