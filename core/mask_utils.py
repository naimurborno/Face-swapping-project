# core/mask_utils.py
import torch
import torch.nn.functional as F


def mask_to_token_mask(
    face_mask:    "torch.Tensor",   # (1, 1, H, W), float, values in [0, 1]
    spatial_size: int,              # e.g. 64 for SD2.1-base 512px (512/8=64)
    threshold:    float = 0.5,
) -> "torch.Tensor":
    """
    Resize the pixel-space face mask to match an attention layer's spatial
    resolution, then flatten to a 1D boolean token mask.

    Called by kv_attention._resize_mask_to_tokens() during inject passes.

    Args:
        face_mask    : (1, 1, H, W) float tensor, values in [0, 1].
                       Produced by segmentation.get_face_mask() and stored
                       on KVCache.face_mask.
        spatial_size : Square root of the attention sequence length.
                       SD2.1-base (512px) layer sizes:
                           down_blocks.0 / up_blocks.3 → 64  (64×64 = 4096 tokens)
                           down_blocks.2 / up_blocks.1 → 16  (16×16 = 256  tokens)
                           mid_block                   →  8  ( 8×8  = 64   tokens)
                       SD2.1 (768px) layer sizes are 96, 24, 12 respectively.
        threshold    : Pixel values above this become True (face token).
                       Default 0.5 — anything more than half-covered is face.

    Returns:
        (spatial_size * spatial_size,) bool tensor.
        True  = face token  → injection applied here.
        False = background  → injection suppressed here.
    """
    resized = F.interpolate(
        face_mask,
        size=(spatial_size, spatial_size),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, S, S)
    flat = resized.squeeze().flatten()   # (S*S,)
    return flat > threshold