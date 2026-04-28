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
    Phase 3 drop-in replacement for diffusers Attention.

    New vs Phase 2:
      - Accepts `layer_name` (str) as a stable cache key instead of str(id(self)).
      - Accepts `depth_category` ("shallow" | "deep" | None).
          shallow → injects HF features, weighted by lambda_hf
          deep    → injects LF features, weighted by lambda_lf
          None    → skipped entirely (no store, no inject)
      - Reads from the correct sub-cache (lf/hf) during inject.
      - Writes to the active freq sub-cache during store.

    Interface:
      Pass kv_cache via cross_attention_kwargs={"kv_cache": cache}.
      Only modifies self-attention (encoder_hidden_states is None).
    """

    def __init__(self, *args, layer_name: str = "", depth_category=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_name = layer_name
        # depth_category: "shallow" | "deep" | None
        self.depth_category = depth_category

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **cross_attention_kwargs,
    ):
        # Pull our cache out — don't forward it to super()
        kv_cache = cross_attention_kwargs.pop("kv_cache", None)

        # Only inject on self-attention; cross-attention is untouched
        is_self_attn = encoder_hidden_states is None

        # Skip conditions: no cache, cross-attn, bypass mode, or unclassified layer
        if (kv_cache is None
                or not is_self_attn
                or kv_cache.mode == "bypass"
                or self.depth_category is None):
            return super().forward(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

        # ── Project Q, K, V ──────────────────────────────────────────────────
        B, seq_len, _ = hidden_states.shape

        norm_hs = (self.norm_cross(hidden_states)
                   if hasattr(self, "norm_cross") and self.norm_cross is not None
                   else hidden_states)

        q = self.to_q(norm_hs)
        k = self.to_k(norm_hs)
        v = self.to_v(norm_hs)

        hd = q.shape[-1] // self.heads
        q = q.view(B, seq_len, self.heads, hd).transpose(1, 2)   # (B, h, S, d)
        k = k.view(B, seq_len, self.heads, hd).transpose(1, 2)
        v = v.view(B, seq_len, self.heads, hd).transpose(1, 2)

        # ── Store mode: write current K,V into the active sub-cache ──────────
        if kv_cache.mode == "store":
            kv_cache.store(self.layer_name, k, v)
            # After storing, run standard attention (no injection during store)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            return self._out_proj(out, q, B, seq_len)

        # ── Inject mode: fetch the correct freq cache and blend ───────────────
        if kv_cache.mode == "inject":
            # Depth → freq band + lambda
            if self.depth_category == "deep":
                freq = "lf"
                weight = kv_cache.lambda_lf
            else:  # "shallow"
                freq = "hf"
                weight = kv_cache.lambda_hf

            k_ref, v_ref = kv_cache.get(self.layer_name, freq)

            if k_ref is not None and weight > 0.0:
                # CFG doubles the batch — tile reference to match
                if k_ref.shape[0] != B:
                    k_ref = k_ref.expand(B, -1, -1, -1)
                    v_ref = v_ref.expand(B, -1, -1, -1)

                if kv_cache.face_mask is not None:
                    token_mask = _resize_mask_to_tokens(
                        kv_cache.face_mask, seq_len, k.device
                    )
                    if token_mask is not None:
                        # (1,1,seq,1) broadcast over (B, heads, seq, hd)
                        m = token_mask.view(1, 1, seq_len, 1).expand(
                            B, self.heads, seq_len, hd
                        )
                        # Additive injection scaled by lambda — avoids clobbering
                        # identity signal from the source latent's own K,V
                        k = k + weight * torch.where(m, k_ref - k, torch.zeros_like(k))
                        v = v + weight * torch.where(m, v_ref - v, torch.zeros_like(v))
                    # else: non-square seq (skip mask, use k/v unchanged)
                else:
                    # Global injection (debug / no-mask mode)
                    k = k + weight * (k_ref - k)
                    v = v + weight * (v_ref - v)

        # ── Attention ─────────────────────────────────────────────────────────
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return self._out_proj(out, q, B, seq_len)

    # ── Helper ────────────────────────────────────────────────────────────────

    def _out_proj(self, out, q, B, seq_len):
        """Merge heads → output projection."""
        inner_dim = q.shape[1] * q.shape[-1]   # heads * head_dim
        out = out.transpose(1, 2).contiguous().reshape(B, seq_len, inner_dim)
        out = out.to(q.dtype)
        out = self.to_out[0](out)   # linear
        out = self.to_out[1](out)   # dropout (identity at eval)
        return out