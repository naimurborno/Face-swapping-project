# patch_unet.py
from diffusers.models.attention_processor import Attention
from kv_attention import KVInjectionAttention


def patch_unet_attention(unet):
    """
    Replace every self-attention (attn1) module in the UNet with
    KVInjectionAttention, preserving all weights.
    """
    patched = 0
    for name, module in unet.named_modules():
        # attn1 = self-attention, attn2 = cross-attention
        if name.endswith(".attn1") and isinstance(module, Attention):
            # Get parent module
            parts = name.split(".")
            parent = unet
            for p in parts[:-1]:
                parent = getattr(parent, p)

            # Create new module with same config
            new_attn = KVInjectionAttention(
                query_dim=module.to_q.in_features,
                cross_attention_dim=None,  # self-attn
                heads=module.heads,
                dim_head=module.to_q.out_features // module.heads,
                dropout=0.0,
                bias=module.to_q.bias is not None,
                upcast_attention=module.upcast_attention,
                out_bias=module.to_out[0].bias is not None,
            ).to(next(module.parameters()).device,
                 dtype=next(module.parameters()).dtype)

            # Copy all weights directly
            new_attn.load_state_dict(module.state_dict())
            new_attn.eval()

            # Replace
            setattr(parent, parts[-1], new_attn)
            patched += 1

    print(f"[INFO] Patched {patched} self-attention layers with KVInjectionAttention")
    return unet