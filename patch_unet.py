# patch_unet.py
from diffusers.models.attention_processor import Attention
from kv_attention import KVInjectionAttention


# ── Depth classification ──────────────────────────────────────────────────────
#
# SD 2.1-base (512px) U-Net spatial resolutions:
#   down_blocks.0  → 64×64   shallow  (HF injection)
#   down_blocks.1  → 32×32   —        (skipped; mid-freq ambiguous)
#   down_blocks.2  → 16×16   deep     (LF injection)
#   mid_block      → 8×8     deep     (LF injection)
#   up_blocks.1    → 16×16   deep     (LF injection)
#   up_blocks.2    → 32×32   —        (skipped)
#   up_blocks.3    → 64×64   shallow  (HF injection)
#
# Adding 32×32 to either bucket is valid but muddies the ablation signal.
# Keep it skipped until you have clean LF/HF results to build on.
#
# To adapt to SD 1.5 or other resolutions, update the strings below.

_DEEP_PREFIXES = (
    "mid_block",
    "down_blocks.2",
    "up_blocks.1",
)

_SHALLOW_PREFIXES = (
    "down_blocks.0",
    "up_blocks.3",
)


def _classify_depth(layer_name: str):
    """
    Returns "deep", "shallow", or None (skip).

    layer_name is the full dotted path to the attn1 module,
    e.g. "down_blocks.0.attentions.0.transformer_blocks.0.attn1"
    """
    for prefix in _DEEP_PREFIXES:
        if layer_name.startswith(prefix):
            return "deep"
    for prefix in _SHALLOW_PREFIXES:
        if layer_name.startswith(prefix):
            return "shallow"
    return None


# ── Patching ──────────────────────────────────────────────────────────────────

def patch_unet_attention(unet):
    """
    Replace every attn1 (self-attention) module with KVInjectionAttention,
    preserving all weights and passing:
      - layer_name: stable string key for the KV cache
      - depth_category: "shallow" | "deep" | None

    Layers classified as None are still replaced (so they can handle the
    kv_cache kwarg without error) but will passthrough without store/inject.

    Returns: unet (mutated in-place), depth_map (dict: layer_name → category)
    """
    patched = 0
    depth_map = {}

    for name, module in unet.named_modules():
        if not (name.endswith(".attn1") and isinstance(module, Attention)):
            continue

        depth = _classify_depth(name)
        depth_map[name] = depth

        # Navigate to parent module
        parts = name.split(".")
        parent = unet
        for p in parts[:-1]:
            parent = getattr(parent, p)

        device = next(module.parameters()).device
        dtype  = next(module.parameters()).dtype

        new_attn = KVInjectionAttention(
            query_dim=module.to_q.in_features,
            cross_attention_dim=None,           # self-attn
            heads=module.heads,
            dim_head=module.to_q.out_features // module.heads,
            dropout=0.0,
            bias=module.to_q.bias is not None,
            upcast_attention=module.upcast_attention,
            out_bias=module.to_out[0].bias is not None,
            # Phase 3 additions:
            layer_name=name,
            depth_category=depth,
        ).to(device, dtype=dtype)

        new_attn.load_state_dict(module.state_dict())
        new_attn.eval()

        setattr(parent, parts[-1], new_attn)
        patched += 1

    # Summary
    n_deep    = sum(1 for v in depth_map.values() if v == "deep")
    n_shallow = sum(1 for v in depth_map.values() if v == "shallow")
    n_skip    = sum(1 for v in depth_map.values() if v is None)
    print(
        f"[patch_unet] Replaced {patched} attn1 layers | "
        f"deep={n_deep}  shallow={n_shallow}  skip={n_skip}"
    )
    return unet, depth_map