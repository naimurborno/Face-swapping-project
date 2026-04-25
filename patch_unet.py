# patch_unet.py
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
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
    for prefix in _DEEP_PREFIXES:
        if layer_name.startswith(prefix):
            return "deep"
    for prefix in _SHALLOW_PREFIXES:
        if layer_name.startswith(prefix):
            return "shallow"
    return None


# ── Kwargs-safe passthrough processor ────────────────────────────────────────
# Applied to attn2 (cross-attention) layers to silence the diffusers warning
# "cross_attention_kwargs ['kv_cache'] are not expected by AttnProcessor2_0".
# These layers never inject anything — the processor just absorbs the extra kwarg.

class _KwargSafeProcessor(AttnProcessor2_0):
    """AttnProcessor2_0 that silently accepts and ignores unknown kwargs."""
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        # Strip kv_cache (and anything else unexpected) before calling super
        kwargs.pop("kv_cache", None)
        return super().__call__(
            attn, hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )


# ── Patching ──────────────────────────────────────────────────────────────────

def patch_unet_attention(unet):
    """
    1. Replace every attn1 (self-attention) Attention module with
       KVInjectionAttention, preserving all weights.
    2. Register _KwargSafeProcessor on every attn2 (cross-attention) module
       so that passing kv_cache in cross_attention_kwargs produces no warnings.

    Returns: unet (mutated in-place), depth_map (dict: layer_name → category)
    """
    patched = 0
    depth_map = {}

    for name, module in unet.named_modules():
        if not isinstance(module, Attention):
            continue

        is_self_attn  = name.endswith(".attn1")
        is_cross_attn = name.endswith(".attn2")

        if is_self_attn:
            depth = _classify_depth(name)
            depth_map[name] = depth

            parts  = name.split(".")
            parent = unet
            for p in parts[:-1]:
                parent = getattr(parent, p)

            device = next(module.parameters()).device
            dtype  = next(module.parameters()).dtype

            new_attn = KVInjectionAttention(
                query_dim=module.to_q.in_features,
                cross_attention_dim=None,
                heads=module.heads,
                dim_head=module.to_q.out_features // module.heads,
                dropout=0.0,
                bias=module.to_q.bias is not None,
                upcast_attention=module.upcast_attention,
                out_bias=module.to_out[0].bias is not None,
                layer_name=name,
                depth_category=depth,
            ).to(device, dtype=dtype)

            new_attn.load_state_dict(module.state_dict())
            new_attn.eval()
            setattr(parent, parts[-1], new_attn)
            patched += 1

        elif is_cross_attn:
            # Swap processor only — weights and module stay unchanged
            module.set_processor(_KwargSafeProcessor())

    n_deep    = sum(1 for v in depth_map.values() if v == "deep")
    n_shallow = sum(1 for v in depth_map.values() if v == "shallow")
    n_skip    = sum(1 for v in depth_map.values() if v is None)
    print(
        f"[patch_unet] Replaced {patched} attn1 layers | "
        f"deep={n_deep}  shallow={n_shallow}  skip={n_skip}"
    )
    return unet, depth_map