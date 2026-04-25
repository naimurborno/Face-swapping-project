# kv_attention.py
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


def _resize_mask_to_tokens(face_mask, seq_len, device):
    """Resize pixel-space mask (1,1,H,W) to (seq_len,) bool token mask."""
    spatial = int(seq_len ** 0.5)
    if spatial * spatial != seq_len:
        return None  # non-square sequence, skip
    m = F.interpolate(
        face_mask.to(device),
        size=(spatial, spatial),
        mode="bilinear",
        align_corners=False,
    )
    return (m.squeeze().flatten() > 0.5)  # (seq_len,) bool


class KVInjectionAttention(Attention):
    """
    Drop-in replacement for diffusers Attention.
    Reads kv_cache from cross_attention_kwargs if present.
    Only modifies self-attention (encoder_hidden_states is None).
    """

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **cross_attention_kwargs,
    ):
        # Pull our cache out — don't pass it to super
        kv_cache = cross_attention_kwargs.pop("kv_cache", None)

        # Only inject on self-attention
        is_self_attn = encoder_hidden_states is None

        if kv_cache is None or not is_self_attn or kv_cache.mode == "bypass":
            return super().forward(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

        # ── Project Q, K, V ──────────────────────────────────────────────────
        B, seq_len, _ = hidden_states.shape

        # layernorm if present
        norm_hs = self.norm_cross(hidden_states) \
                  if hasattr(self, "norm_cross") and self.norm_cross is not None \
                  else hidden_states

        q = self.to_q(norm_hs)
        k = self.to_k(norm_hs)
        v = self.to_v(norm_hs)

        # Reshape: (B, seq, heads*hd) → (B, heads, seq, hd)
        hd = q.shape[-1] // self.heads
        q = q.view(B, seq_len, self.heads, hd).transpose(1, 2)
        k = k.view(B, seq_len, self.heads, hd).transpose(1, 2)
        v = v.view(B, seq_len, self.heads, hd).transpose(1, 2)

        # ── KV Cache ─────────────────────────────────────────────────────────
        layer_name = str(id(self))

        if kv_cache.mode == "store":
            kv_cache.store(layer_name, k, v)

        elif kv_cache.mode == "inject":
            k_ref, v_ref = kv_cache.get(layer_name)

            if k_ref is not None:
                # Match batch size (CFG doubles it)
                if k_ref.shape[0] != B:
                    k_ref = k_ref.expand(B, -1, -1, -1)
                    v_ref = v_ref.expand(B, -1, -1, -1)

                if kv_cache.face_mask is not None:
                    token_mask = _resize_mask_to_tokens(
                        kv_cache.face_mask, seq_len, k.device
                    )
                    if token_mask is not None:
                        # (1, 1, seq, 1) → broadcast over B, heads, seq, hd
                        m = token_mask.view(1, 1, seq_len, 1).expand(B, self.heads, seq_len, hd)
                        k = torch.where(m, k_ref, k)
                        v = torch.where(m, v_ref, v)
                else:
                    k, v = k_ref, v_ref  # global injection (debug)

        # ── Scaled dot-product attention ──────────────────────────────────────
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )  # (B, heads, seq, hd)

        # Merge heads: (B, heads, seq, hd) → (B, seq, heads*hd)
        oB, oh, os, ohd = out.shape
        out = out.transpose(1, 2).contiguous().reshape(oB, os, oh * ohd)
        out = out.to(q.dtype)

        # Output projection
        out = self.to_out[0](out)
        out = self.to_out[1](out)

        return out