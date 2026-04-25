# freq_utils.py
"""
Frequency decomposition utilities for Phase 3.

fft_decompose(): Splits an image tensor into LF and HF components via FFT.
encode_freq_latents(): VAE-encodes LF and HF images to latent space.

Both are self-contained and have no dependency on the diffusers pipeline.
"""
import torch
import torch.nn.functional as F
from typing import Tuple


def fft_decompose(
    image: torch.Tensor,
    cutoff_ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split image into Low-Frequency (LF) and High-Frequency (HF) components.

    Args:
        image:        (B, C, H, W) float tensor, any value range.
        cutoff_ratio: Fraction of the spatial frequency radius to treat as LF.
                      0.1 = inner 10% of the spectrum (good default for
                      face structure vs texture split at 512px).
                      Raise if LF still contains edges; lower if HF loses detail.

    Returns:
        lf: (B, C, H, W) — low-frequency reconstruction (head shape, proportions)
        hf: (B, C, H, W) — high-frequency residual (iris, pores, skin texture)

    Reconstruction: lf + hf == image (exact up to float precision ~1e-6).
    """
    # FFT over spatial dims
    f = torch.fft.fft2(image)
    f_shift = torch.fft.fftshift(f, dim=(-2, -1))

    B, C, H, W = image.shape
    cy, cx = H // 2, W // 2
    rh = max(1, int(cutoff_ratio * H / 2))
    rw = max(1, int(cutoff_ratio * W / 2))

    # Circular (or rectangular) LF mask centred at DC
    mask = torch.zeros(H, W, device=image.device, dtype=image.dtype)
    mask[cy - rh: cy + rh, cx - rw: cx + rw] = 1.0
    mask = mask.unsqueeze(0).unsqueeze(0)       # (1, 1, H, W)

    f_lf = f_shift * mask
    f_lf = torch.fft.ifftshift(f_lf, dim=(-2, -1))
    lf = torch.fft.ifft2(f_lf).real

    hf = image - lf
    return lf, hf


@torch.no_grad()
def encode_freq_latents(
    vae,
    image: torch.Tensor,         # (1, 3, H, W), float16, [-1, 1]
    cutoff_ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FFT-decompose an image and VAE-encode both components to latent space.

    Runs two encoder passes back-to-back (not batched) to avoid doubling
    VRAM peak. On Colab T4 this is safer than stacking them.

    Args:
        vae:          pipe.vae (already on device, frozen)
        image:        (1, 3, H, W) float16 tensor, values in [-1, 1]
        cutoff_ratio: passed through to fft_decompose

    Returns:
        lf_latent: (1, 4, H/8, W/8) — latent of LF image
        hf_latent: (1, 4, H/8, W/8) — latent of HF image
    """
    lf_img, hf_img = fft_decompose(image, cutoff_ratio)

    # Clamp back to valid pixel range to avoid VAE instability on HF extremes
    lf_img = lf_img.clamp(-1.0, 1.0)
    hf_img = hf_img.clamp(-1.0, 1.0)

    sf = vae.config.scaling_factor

    lf_latent = vae.encode(lf_img.to(dtype=vae.dtype)).latent_dist.sample() * sf
    hf_latent = vae.encode(hf_img.to(dtype=vae.dtype)).latent_dist.sample() * sf

    return lf_latent, hf_latent


def smoke_test():
    """Run without GPU to verify FFT correctness."""
    dummy = torch.randn(1, 3, 512, 512)
    lf, hf = fft_decompose(dummy, cutoff_ratio=0.1)

    assert lf.shape == dummy.shape, f"LF shape: {lf.shape}"
    assert hf.shape == dummy.shape, f"HF shape: {hf.shape}"

    recon_err = (lf + hf - dummy).abs().max().item()
    assert recon_err < 1e-4, f"Reconstruction error too large: {recon_err:.2e}"

    print(f"[freq_utils] smoke_test passed | recon_err={recon_err:.2e}")
    print(f"  LF energy: {lf.pow(2).mean().item():.4f}")
    print(f"  HF energy: {hf.pow(2).mean().item():.4f}")


if __name__ == "__main__":
    smoke_test()