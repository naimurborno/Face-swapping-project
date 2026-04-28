# core/decomposition.py
"""
Frequency decomposition for the KV-injection face swap pipeline.

Single responsibility: split an aligned reference face image into a
low-frequency (LF) component and a high-frequency (HF) component.

The ablation.decomposition flag in configs/default.yaml selects the method:
    "gaussian" — Gaussian blur split (default)
                 Smooth rolloff, no ringing artifacts. Best for faces.
    "fft"      — FFT rectangular mask split (ablation A1)
                 Sharp frequency cutoff. Introduces Gibbs ringing at boundaries
                 which degrades face texture — used only to quantify this effect.
    "none"     — No split; aligned image returned as both LF and HF (ablation A1)
                 Equivalent to Phase 2 mode. Tests whether the freq split itself
                 contributes anything vs. uniform injection of the whole image.

Why Gaussian is the default:
    The Gaussian kernel has a smooth frequency rolloff (no hard cutoff), so the
    LF component contains face shape and skin tone without edge ringing, and the
    HF component contains iris, pore, and texture detail without ghost edges.
    FFT with a rectangular mask causes Gibbs ringing which corrupts the iris
    and skin texture that shallow U-Net layers are supposed to capture.

Public API:
    DecomposeResult   — NamedTuple holding LF, HF images and metadata
    decompose()       — main entry point: aligned image → DecomposeResult
    decompose_gaussian()  — Gaussian method (used directly by stage1)
    decompose_fft()       — FFT method (used directly by stage1)
    to_pil_inputs()   — convert DecomposeResult → PIL images ready for VAE

Dependencies:
    pip install opencv-python-headless numpy torch pillow
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import NamedTuple, Tuple
from PIL import Image


# ── Output container ──────────────────────────────────────────────────────────

class DecomposeResult(NamedTuple):
    """
    Output of decompose(). All arrays are float32 in BGR channel order.

    LF            : (H, W, 3) float32, range [0, 255]
                    Low-frequency component: head shape, skin tone, proportions.
                    Goes into deep U-Net layers (8×8 / 16×16 spatial resolution).

    HF            : (H, W, 3) float32, range [-255, 255]  (Gaussian / FFT)
                    or (H, W, 3) float32, range [0, 255]   (method="none")
                    High-frequency residual: iris, pores, skin texture.
                    Goes into shallow U-Net layers (64×64 spatial resolution).
                    For "none" method, HF == LF == aligned image (no split).

    method        : str — which method produced this result
    lf_energy     : float — mean squared energy of LF (sanity check)
    hf_std        : float — std of HF (expect > 10 for textured faces;
                    very low std means the image is smooth / low texture)
    """
    LF        : np.ndarray
    HF        : np.ndarray
    method    : str
    lf_energy : float
    hf_std    : float


# ── Gaussian decomposition ────────────────────────────────────────────────────

def decompose_gaussian(
    aligned_bgr: np.ndarray,   # (H, W, 3) uint8 or float32
    kernel:      int   = 31,
    sigma:       float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image into LF and HF via Gaussian blur.

        LF = GaussianBlur(image)
        HF = image - LF

    Args:
        aligned_bgr : BGR image, uint8 or float32. If uint8, converted internally.
        kernel      : Gaussian kernel size. Must be odd. Default 31.
                      Larger → smoother LF, more detail pushed into HF.
                      Recommended range: 21–51 for 512px face images.
        sigma       : Gaussian sigma. Default 5.0.
                      Larger → more LF bleed (LF retains more mid-frequency).

    Returns:
        LF : float32 [0, 255]      — structure / skin tone
        HF : float32 [-255, 255]   — texture / detail residual

    Reconstruction: LF + HF == original (exact up to float32 precision).

    Why this is preferred over FFT:
        Gaussian has infinite support in frequency space (smooth rolloff).
        There is no hard cutoff → no Gibbs ringing → no ghost edges at the
        nose, lips, or iris boundary that would appear in shallow-layer KV.
    """
    assert kernel % 2 == 1, (
        f"[decomposition] Gaussian kernel must be odd, got {kernel}. "
        f"Use 31 (default), 21, 41, or 51."
    )

    img_f = aligned_bgr.astype(np.float32)
    LF    = cv2.GaussianBlur(img_f, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    HF    = img_f - LF

    return LF, HF


# ── FFT decomposition ─────────────────────────────────────────────────────────

def decompose_fft(
    aligned_bgr:  np.ndarray,    # (H, W, 3) uint8 or float32
    cutoff_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image into LF and HF via FFT rectangular mask.

        LF = IFFT(FFT(image) * rectangular_mask_centred_at_DC)
        HF = image - LF

    Args:
        aligned_bgr  : BGR image, uint8 or float32.
        cutoff_ratio : Fraction of the half-spatial-frequency radius treated as LF.
                       0.1 = inner 10% of the spectrum (good default at 512px).
                       Raise (e.g. 0.15) if LF still contains visible edges.
                       Lower (e.g. 0.05) if HF loses fine detail.

    Returns:
        LF : float32 [0, 255]      — low-frequency reconstruction
        HF : float32 [-255, 255]   — high-frequency residual

    Reconstruction: LF + HF == original (exact to ~1e-3 float32 precision).

    WARNING — Ablation use only:
        FFT with a rectangular mask causes Gibbs ringing (oscillation near
        sharp edges). On face images this appears as halo artefacts around the
        nose, lips, and iris boundary. These artefacts propagate into the HF
        KV cache and corrupt texture injection in shallow U-Net layers.
        This method is intentionally included so the ablation can quantify
        the degradation relative to Gaussian. Do NOT use it as the default.

    Float16 safety:
        torch.fft.fft2 does not support float16 (ComplexHalf is experimental
        and produces NaN on most ops). This function casts to float32 internally
        and casts back to the input dtype before returning.
    """
    orig_dtype = np.float32   # always work in float32

    # Convert to float32 for FFT
    img_f = aligned_bgr.astype(np.float32)   # (H, W, 3)

    # Process each channel independently (FFT is linear, channels are independent)
    H, W, C = img_f.shape
    LF = np.zeros_like(img_f)

    cy, cx = H // 2, W // 2
    rh = max(1, int(cutoff_ratio * H / 2))
    rw = max(1, int(cutoff_ratio * W / 2))

    # Build rectangular LF mask centred at DC (after fftshift)
    mask = np.zeros((H, W), dtype=np.float32)
    mask[cy - rh: cy + rh, cx - rw: cx + rw] = 1.0

    for c in range(C):
        channel = img_f[:, :, c]

        # FFT → shift DC to centre → apply mask → shift back → IFFT
        f       = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f)
        f_lf    = f_shift * mask
        f_lf_us = np.fft.ifftshift(f_lf)
        lf_c    = np.fft.ifft2(f_lf_us).real.astype(np.float32)

        LF[:, :, c] = lf_c

    HF = img_f - LF

    return LF, HF


# ── Main entry point ──────────────────────────────────────────────────────────

def decompose(
    aligned_bgr: np.ndarray,   # (H, W, 3) uint8 BGR — output of alignment.warp_reference()
    method:      str   = "gaussian",
    # Gaussian parameters (used when method="gaussian")
    kernel:      int   = 31,
    sigma:       float = 5.0,
    # FFT parameters (used when method="fft")
    cutoff_ratio: float = 0.1,
) -> DecomposeResult:
    """
    Decompose an aligned reference face into LF and HF components.

    This is the single function called by stage1_segment.py. The method
    argument maps directly to the ablation.decomposition flag in default.yaml,
    so switching the ablation requires zero code changes here.

    Args:
        aligned_bgr  : BGR uint8 image — the affine-warped reference face.
                       Must be the output of alignment.warp_reference().
        method       : "gaussian" | "fft" | "none"
                       Reads from ablation.decomposition in default.yaml.
        kernel       : Gaussian kernel size (method="gaussian" only).
                       Reads from stage1.gaussian.kernel in default.yaml.
        sigma        : Gaussian sigma (method="gaussian" only).
                       Reads from stage1.gaussian.sigma in default.yaml.
        cutoff_ratio : FFT cutoff fraction (method="fft" only).
                       Reads from stage1.fft.cutoff_ratio in default.yaml.

    Returns:
        DecomposeResult(LF, HF, method, lf_energy, hf_std)

        LF  : float32 BGR, [0, 255]     — always safe to clip and cast to uint8
        HF  : float32 BGR, [-255, 255]  — add 128 before passing to PIL/VAE
              Exception: method="none" → HF == LF (same image, range [0, 255])

    Raises:
        ValueError for unknown method strings.
        AssertionError if Gaussian kernel is even.

    Post-decomposition sanity checks (printed to stdout):
        hf_std > 10  — expected for a textured face image.
                        Very low std (< 5) means the image is unusually smooth
                        (over-blurred input, or extreme low-texture face).
        lf_energy    — informational; no hard threshold.
    """
    if method not in ("gaussian", "fft", "none"):
        raise ValueError(
            f"[decomposition] Unknown method '{method}'. "
            f"Choose: 'gaussian' | 'fft' | 'none'"
        )

    # ── "none": no decomposition — pass whole image as both LF and HF ────
    if method == "none":
        img_f   = aligned_bgr.astype(np.float32)
        LF      = img_f
        HF      = img_f   # identical — no frequency split
        print("[decomposition] method='none' — LF == HF == aligned image (no split).")

    # ── "gaussian": Gaussian blur split ───────────────────────────────────
    elif method == "gaussian":
        LF, HF = decompose_gaussian(aligned_bgr, kernel=kernel, sigma=sigma)
        print(
            f"[decomposition] method='gaussian' | "
            f"kernel={kernel} sigma={sigma}"
        )

    # ── "fft": FFT rectangular mask split ─────────────────────────────────
    else:  # method == "fft"
        LF, HF = decompose_fft(aligned_bgr, cutoff_ratio=cutoff_ratio)
        print(
            f"[decomposition] method='fft' | "
            f"cutoff_ratio={cutoff_ratio}"
        )

    # ── Sanity metrics ─────────────────────────────────────────────────────
    lf_energy = float(LF.astype(np.float64).mean() ** 2)
    hf_std    = float(HF.std())

    print(
        f"[decomposition] LF energy={lf_energy:.2f} | "
        f"HF std={hf_std:.2f} "
        f"{'✓ ok' if hf_std > 10 else '⚠ low — check input image texture'}"
    )

    return DecomposeResult(
        LF        = LF,
        HF        = HF,
        method    = method,
        lf_energy = lf_energy,
        hf_std    = hf_std,
    )


# ── PIL conversion helper ─────────────────────────────────────────────────────

def to_pil_inputs(
    result:      DecomposeResult,
    aligned_bgr: np.ndarray,        # original aligned image (for aligned_pil)
    target_size: int = 512,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Convert a DecomposeResult into PIL RGB images ready for VAE encoding.

    This is called by stage1_segment.py after decompose() returns. It handles
    all the dtype casts, channel flips (BGR→RGB), and HF shifting so that
    Stage 2 / the VAE encoder never needs to know what method was used.

    HF shift convention:
        Gaussian and FFT HF are in [-255, 255]. Adding 128 shifts them to
        [0, 255] for PIL / uint8 compatibility. The VAE encodes structural
        patterns, not absolute pixel values, so this shift does not affect
        what the KV features capture. Stage 2 does NOT reverse this shift —
        the "shifted texture map" latent is what gets injected.
        Exception: method="none" — HF is already in [0, 255], no shift needed.

    Args:
        result      : DecomposeResult from decompose().
        aligned_bgr : The uint8 BGR aligned reference image (for aligned_pil).
        target_size : Resize all outputs to this square size (512 or 768).

    Returns:
        aligned_pil : PIL RGB — full aligned reference (for display / phase2 mode)
        lf_pil      : PIL RGB — LF component, [0, 255]
        hf_pil      : PIL RGB — HF component, shifted to [0, 255]
    """
    S = target_size

    # ── Aligned reference (display + phase2 baseline) ─────────────────────
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    aligned_pil = Image.fromarray(aligned_rgb).resize((S, S), Image.LANCZOS)

    # ── LF: clip to [0, 255], cast to uint8, convert BGR→RGB ──────────────
    lf_uint8 = np.clip(result.LF, 0.0, 255.0).astype(np.uint8)
    lf_rgb   = cv2.cvtColor(lf_uint8, cv2.COLOR_BGR2RGB)
    lf_pil   = Image.fromarray(lf_rgb).resize((S, S), Image.LANCZOS)

    # ── HF: shift [-255,255] → [0,255], cast to uint8, convert BGR→RGB ────
    # Exception: method="none" → HF is already in [0,255] (no shift needed)
    if result.method == "none":
        hf_shifted = np.clip(result.HF, 0.0, 255.0).astype(np.uint8)
    else:
        hf_shifted = np.clip(result.HF + 128.0, 0.0, 255.0).astype(np.uint8)

    hf_rgb = cv2.cvtColor(hf_shifted, cv2.COLOR_BGR2RGB)
    hf_pil = Image.fromarray(hf_rgb).resize((S, S), Image.LANCZOS)

    return aligned_pil, lf_pil, hf_pil


# ── Torch tensor helper (for direct VAE encode in stage2) ─────────────────────

def mask_to_tensor(
    mask_uint8:  np.ndarray,   # (H, W) uint8 0/255 — output of segmentation.get_face_mask()
    target_size: int = 512,
) -> torch.Tensor:
    """
    Convert a segmentation mask to a (1, 1, H, W) float32 tensor in [0, 1].

    This is the format KVCache.face_mask expects. Called by stage1_segment.py
    when building the artifacts bundle.

    Args:
        mask_uint8   : Raw uint8 mask from get_face_mask().
        target_size  : Resize to match the pipeline's spatial resolution.

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


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    """
    Verify all three decomposition methods on a synthetic image.
    Runs on CPU, no GPU or real images needed.
    """
    print("[decomposition] Running smoke test...")

    dummy_bgr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    for method in ("gaussian", "fft", "none"):
        result = decompose(dummy_bgr, method=method)

        assert result.LF.dtype  == np.float32, f"LF dtype wrong: {result.LF.dtype}"
        assert result.HF.dtype  == np.float32, f"HF dtype wrong: {result.HF.dtype}"
        assert result.LF.shape  == dummy_bgr.shape
        assert result.HF.shape  == dummy_bgr.shape
        assert result.method    == method

        # Reconstruction check (not applicable to "none" since LF == HF == original)
        if method != "none":
            recon_err = np.abs(
                result.LF.astype(np.float64) +
                result.HF.astype(np.float64) -
                dummy_bgr.astype(np.float64)
            ).max()
            assert recon_err < 0.5, (
                f"[{method}] Reconstruction error too large: {recon_err:.4f}"
            )
            print(f"  method={method} | recon_err={recon_err:.4e} | ✓")
        else:
            print(f"  method={method} | LF==HF==original | ✓")

        # PIL conversion check
        aligned_pil, lf_pil, hf_pil = to_pil_inputs(result, dummy_bgr, target_size=512)
        assert lf_pil.size == (512, 512)
        assert hf_pil.size == (512, 512)

    print("[decomposition] All smoke tests passed.\n")


if __name__ == "__main__":
    _smoke_test()