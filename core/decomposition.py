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
    DecomposeResult       — NamedTuple holding LF, HF images and metadata
    decompose()           — main entry point: aligned BGR numpy → DecomposeResult
    decompose_pil()       — NEW: PIL RGB image → (lf_pil, hf_pil) directly
                            Used by core/compositing.py for freq-aware blending.
                            This is the only addition to the original file.
    decompose_gaussian()  — Gaussian method (used directly by stage1)
    decompose_fft()       — FFT method (used directly by stage1)
    to_pil_inputs()       — convert DecomposeResult → PIL images ready for VAE
    mask_to_tensor()      — segmentation mask → (1,1,H,W) float32 tensor

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

    Returns:
        LF : float32 [0, 255]      — low-frequency reconstruction
        HF : float32 [-255, 255]   — high-frequency residual

    WARNING — Ablation use only:
        FFT with a rectangular mask causes Gibbs ringing on face images.
        Use Gaussian for all non-ablation runs.
    """
    img_f  = aligned_bgr.astype(np.float32)
    H, W, C = img_f.shape
    LF     = np.zeros_like(img_f)

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


# ── Main entry point ──────────────────────────────────────────────────────────

def decompose(
    aligned_bgr:  np.ndarray,
    method:       str   = "gaussian",
    kernel:       int   = 31,
    sigma:        float = 5.0,
    cutoff_ratio: float = 0.1,
) -> DecomposeResult:
    """
    Decompose an aligned reference face into LF and HF components.

    Called by stage1_segment.py and by decompose_pil() below.

    Args:
        aligned_bgr  : BGR uint8 image — the affine-warped reference face.
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
        img_f = aligned_bgr.astype(np.float32)
        LF    = img_f
        HF    = img_f
        print("[decomposition] method='none' — LF == HF == aligned image (no split).")

    elif method == "gaussian":
        LF, HF = decompose_gaussian(aligned_bgr, kernel=kernel, sigma=sigma)
        print(
            f"[decomposition] method='gaussian' | "
            f"kernel={kernel} sigma={sigma}"
        )

    else:
        LF, HF = decompose_fft(aligned_bgr, cutoff_ratio=cutoff_ratio)
        print(
            f"[decomposition] method='fft' | "
            f"cutoff_ratio={cutoff_ratio}"
        )

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


# ── NEW: PIL entry point for compositing ──────────────────────────────────────

def decompose_pil(
    pil_image:    Image.Image,
    method:       str   = "gaussian",
    kernel:       int   = 31,
    sigma:        float = 5.0,
    cutoff_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a PIL RGB image into LF and HF numpy arrays.

    This is the entry point used by core/compositing.py for
    frequency-decomposed blending. It handles the PIL→BGR→PIL
    conversion internally so compositing.py never needs to know
    about channel ordering or dtype conventions.

    Unlike decompose() which works on BGR numpy arrays (Stage 1 convention),
    this function accepts and returns values in a compositing-friendly format:

        Input : PIL RGB image (any size)
        Output: LF float32 [0, 255]      BGR numpy (H, W, 3)
                HF float32 [-255, 255]   BGR numpy (H, W, 3)
                   Exception: method="none" → HF in [0, 255]

    The caller (compositing.py) works in BGR numpy throughout so it can
    call cv2 operations (seamlessClone, GaussianBlur) without extra conversions.

    Args:
        pil_image    : PIL RGB image — either source_pil or generated_pil.
                       Must already be at the target resolution (512 or 768px).
        method       : "gaussian" | "fft" | "none"
                       Should match the ablation.decomposition flag used in
                       Stage 1 so LF/HF splits are consistent across the pipeline.
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
        # blend LF and HF separately, then reconstruct
        final = final_LF + final_HF
    """
    # PIL RGB → BGR numpy (OpenCV convention used throughout compositing)
    rgb_array = np.array(pil_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    result = decompose(
        bgr_array,
        method       = method,
        kernel       = kernel,
        sigma        = sigma,
        cutoff_ratio = cutoff_ratio,
    )

    return result.LF, result.HF


# ── PIL conversion helper ─────────────────────────────────────────────────────

def to_pil_inputs(
    result:      DecomposeResult,
    aligned_bgr: np.ndarray,
    target_size: int = 512,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Convert a DecomposeResult into PIL RGB images ready for VAE encoding.

    Called by stage1_segment.py after decompose() returns.

    HF shift convention:
        Gaussian and FFT HF are in [-255, 255]. Adding 128 shifts them to
        [0, 255] for PIL / uint8 compatibility. Stage 2 does NOT reverse
        this shift — the shifted texture map latent is what gets injected.
        Exception: method="none" — HF is already in [0, 255], no shift needed.

    Args:
        result      : DecomposeResult from decompose().
        aligned_bgr : uint8 BGR aligned reference image.
        target_size : Resize all outputs to this square size.

    Returns:
        aligned_pil : PIL RGB — full aligned reference
        lf_pil      : PIL RGB — LF component [0, 255]
        hf_pil      : PIL RGB — HF component shifted to [0, 255]
    """
    S = target_size

    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    aligned_pil = Image.fromarray(aligned_rgb).resize((S, S), Image.LANCZOS)

    lf_uint8 = np.clip(result.LF, 0.0, 255.0).astype(np.uint8)
    lf_rgb   = cv2.cvtColor(lf_uint8, cv2.COLOR_BGR2RGB)
    lf_pil   = Image.fromarray(lf_rgb).resize((S, S), Image.LANCZOS)

    if result.method == "none":
        hf_shifted = np.clip(result.HF, 0.0, 255.0).astype(np.uint8)
    else:
        hf_shifted = np.clip(result.HF + 128.0, 0.0, 255.0).astype(np.uint8)

    hf_rgb = cv2.cvtColor(hf_shifted, cv2.COLOR_BGR2RGB)
    hf_pil = Image.fromarray(hf_rgb).resize((S, S), Image.LANCZOS)

    return aligned_pil, lf_pil, hf_pil


# ── Torch tensor helper ───────────────────────────────────────────────────────

def mask_to_tensor(
    mask_uint8:  np.ndarray,
    target_size: int = 512,
) -> torch.Tensor:
    """
    Convert a segmentation mask to a (1, 1, H, W) float32 tensor in [0, 1].

    Called by stage1_segment.py when building the artifacts bundle.

    Args:
        mask_uint8   : Raw uint8 mask from get_face_mask().
        target_size  : Resize to match the pipeline spatial resolution.

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

#---------Build Chimera function----------------------

def build_chimera(
    source_bgr:    np.ndarray,   # (H, W, 3) uint8 — source face
    aligned_bgr:   np.ndarray,   # (H, W, 3) uint8 — reference warped to source pose
    method:        str   = "gaussian",
    kernel:        int   = 31,
    sigma:         float = 5.0,
    cutoff_ratio:  float = 0.1,
) -> np.ndarray:
    """
    Build chimera image: source_LF + aligned_HF

    Takes the low-frequency structure from the SOURCE (Person A's geometry,
    pose, proportions) and the high-frequency texture from the ALIGNED
    REFERENCE (Person B's iris, pores, skin detail).

    Because aligned_bgr is already warped into source pose, the HF spatial
    positions correspond correctly to source_LF spatial positions — no
    misalignment.

    Returns:
        chimera_bgr : (H, W, 3) uint8 — ready to save as PIL / encode to VAE
    """
    # Decompose source → take only LF
    src_result = decompose(source_bgr, method=method,
                           kernel=kernel, sigma=sigma,
                           cutoff_ratio=cutoff_ratio)
    source_LF = src_result.LF   # float32 [0, 255]

    # Decompose aligned reference → take only HF
    ref_result = decompose(aligned_bgr, method=method,
                           kernel=kernel, sigma=sigma,
                           cutoff_ratio=cutoff_ratio)
    aligned_HF = ref_result.HF  # float32 [-255, 255]

    # Merge: source shape + reference texture
    chimera = source_LF + aligned_HF          # float32 [-255, 255]
    chimera = np.clip(chimera, 0, 255).astype(np.uint8)

    print(
        f"[decomposition] Chimera built | "
        f"source_LF mean={source_LF.mean():.1f} | "
        f"aligned_HF std={aligned_HF.std():.2f}"
    )
    return chimera

# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    """
    Verify all three decomposition methods on a synthetic image.
    Also verifies decompose_pil() round-trip.
    Runs on CPU, no GPU or real images needed.
    """
    print("[decomposition] Running smoke test...")

    dummy_bgr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    dummy_pil = Image.fromarray(cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB))

    for method in ("gaussian", "fft", "none"):
        # Test decompose() — BGR numpy path
        result = decompose(dummy_bgr, method=method)

        assert result.LF.dtype == np.float32
        assert result.HF.dtype == np.float32
        assert result.LF.shape == dummy_bgr.shape
        assert result.HF.shape == dummy_bgr.shape
        assert result.method   == method

        if method != "none":
            recon_err = np.abs(
                result.LF.astype(np.float64) +
                result.HF.astype(np.float64) -
                dummy_bgr.astype(np.float64)
            ).max()
            assert recon_err < 0.5, f"[{method}] Recon error: {recon_err:.4f}"
            print(f"  decompose()     method={method} | recon_err={recon_err:.4e} | ✓")
        else:
            print(f"  decompose()     method={method} | LF==HF==original | ✓")

        # Test decompose_pil() — PIL RGB path (used by compositing)
        LF_pil, HF_pil = decompose_pil(dummy_pil, method=method)
        assert LF_pil.shape == dummy_bgr.shape
        assert HF_pil.shape == dummy_bgr.shape
        print(f"  decompose_pil() method={method} | LF={LF_pil.shape} HF={HF_pil.shape} | ✓")

        # Test to_pil_inputs()
        aligned_pil, lf_pil, hf_pil = to_pil_inputs(result, dummy_bgr, target_size=512)
        assert lf_pil.size == (512, 512)
        assert hf_pil.size == (512, 512)
        print(f"  to_pil_inputs() method={method} | ✓")

    print("[decomposition] All smoke tests passed.\n")


if __name__ == "__main__":
    _smoke_test()