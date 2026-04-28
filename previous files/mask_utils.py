# mask_utils.py
import torch
import torch.nn.functional as F

def mask_to_token_mask(
    face_mask: "torch.Tensor",   # (1, 1, H, W), float, values in [0,1]
    spatial_size: int,            # e.g. 96 for 768px SD2 (768/8=96)
    threshold: float = 0.5,
) -> "torch.Tensor":
    """
    Resize the pixel-space face mask to match the attention layer's
    spatial resolution, then flatten to a 1D boolean token mask.

    Returns: (spatial_size * spatial_size,) bool tensor
    """
    resized = F.interpolate(
        face_mask,
        size=(spatial_size, spatial_size),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, S, S)
    flat = resized.squeeze().flatten()  # (S*S,)
    return flat > threshold