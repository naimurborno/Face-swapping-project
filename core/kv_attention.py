# core/kv_attention.py
"""
KV-injection self-attention module for the face swap pipeline.

Drop-in replacement for diffusers Attention. Handles all three modes of the
KV cache (store / inject / bypass) and all three depth_routing ablation
conditions without any changes needed in stage2_diffusion.py.

depth_routing ablation (configs/default.yaml → ablation.depth_routing):
    "correct"  — deep layers inject LF, shallow layers inject HF  (proposed)
    "swapped"  — deep layers inject HF, shallow layers inject LF  (ablation A2)
    "uniform"  — all classified layers inject from the LF cache,
                 weighted by lambda_lf                             (ablation A2)

The depth_routing value is read from kv_cache.depth_routing, which
stage2_diffusion.py sets once before the denoising loop starts.
KVInjectionAttention itself never imports the config — it just reads the
attribute off the cache object it receives at forward time.

Interface:
    Pass kv_cache via cross_attention_kwargs={"kv_cache": cache}.
    Only self-attention layers are modified (encoder_hidden_states is None).
    Cross-attention layers fall through to super().forward() unchanged.
"""

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


# ── Mask resize helper ────────────────────────────────────────────────────────

def _resize_mask_to_tokens(
    face_mask: torch.Tensor,   # (1, 1, H, W) float [0, 1]
    seq_len:   int,
    device:    torch.device,
) -> "torch.Tensor | None":
    """
    Resize a pixel-space face mask to a (seq_len,) boolean token mask.

    The attention sequence is assumed to be a flattened spatial grid.
    For SD2.1-base at 512px, spatial sizes per layer are:
        down_blocks.0  → 64×64 = 4096 tokens
        down_blocks.2  → 16×16 = 256  tokens
        mid_block      → 8×8   = 64   tokens
        up_blocks.1    → 16×16 = 256  tokens
        up_blocks.3    → 64×64 = 4096 tokens

    For non-square sequences (seq_len ≠ spatial² for any integer spatial)
    the mask is skipped and None is returned — the caller then runs injection
    globally rather than raising an error.

    Args:
        face_mask : (1, 1, H, W) float tensor, values in [0, 1].
        seq_len   : Number of attention tokens (H_spatial × W_spatial).
        device    : Target device for the resized mask.

    Returns:
        (seq_len,) bool tensor — True = face token (inject here).
        None if seq_len is not a perfect square.
    """
    spatial = int(seq_len ** 0.5)
    if spatial * spatial != seq_len:
        return None   # non-square sequence, skip spatial gating

    m = F.interpolate(
        face_mask.to(device),
        size=(spatial, spatial),
        mode="bilinear",
        align_corners=False,
    )
    return (m.squeeze().flatten() > 0.5)   # (seq_len,) bool


# ── Depth routing resolver ────────────────────────────────────────────────────

def _resolve_freq_and_weight(
    depth_category: str,        # "deep" | "shallow"
    depth_routing:  str,        # "correct" | "swapped" | "uniform"
    kv_cache,
):
    """
    Map (depth_category, depth_routing) → (freq_band, injection_weight).

    This is the single function that implements the A2 ablation switch.
    All three routing modes are handled here so the rest of forward() is
    routing-agnostic.

    depth_routing="correct" (proposed method):
        deep    → "lf", lambda_lf   (deep layers capture global structure → LF)
        shallow → "hf", lambda_hf   (shallow layers capture local texture → HF)

    depth_routing="swapped" (ablation A2 — inverted routing):
        deep    → "hf", lambda_hf   (intentionally wrong assignment)
        shallow → "lf", lambda_lf   (intentionally wrong assignment)
        Expected to degrade quality, validating the frequency-depth hypothesis.

    depth_routing="uniform" (ablation A2 — no depth distinction):
        deep    → "lf", lambda_lf   (both depth categories get the same cache)
        shallow → "lf", lambda_lf   (shallow gets LF instead of HF)
        Tests whether depth-aware routing adds value over uniform injection.

    Args:
        depth_category : The layer's depth classification set by patch_unet.py.
        depth_routing  : The ablation flag from kv_cache.depth_routing.
        kv_cache       : KVCache instance — source of lambda values.

    Returns:
        freq   : "lf" or "hf" — which sub-cache to read from.
        weight : float — injection strength for this layer and step.

    Raises:
        ValueError for unknown depth_routing strings (fail-fast, not silent).
    """
    if depth_routing == "correct":
        if depth_category == "deep":
            return "lf", kv_cache.lambda_lf
        else:   # "shallow"
            return "hf", kv_cache.lambda_hf

    elif depth_routing == "swapped":
        # Inverted: deep gets HF, shallow gets LF
        if depth_category == "deep":
            return "hf", kv_cache.lambda_hf
        else:   # "shallow"
            return "lf", kv_cache.lambda_lf

    elif depth_routing == "uniform":
        # All classified layers read from LF cache, weighted by lambda_lf
        # Both depth categories are treated identically — no routing distinction
        return "lf", kv_cache.lambda_lf

    else:
        raise ValueError(
            f"[kv_attention] Unknown depth_routing '{depth_routing}'. "
            f"Choose: 'correct' | 'swapped' | 'uniform'"
        )


# ── Main module ───────────────────────────────────────────────────────────────

class KVInjectionAttention(Attention):
    """
    Drop-in replacement for diffusers Attention for self-attention layers.

    Constructed by patch_unet.py via load_state_dict() — all original weights
    are preserved. The only behavioural difference is the KV cache logic in
    forward().

    Args:
        layer_name     : Stable string identifier for this layer
                         (e.g. "down_blocks.0.attentions.0.attn1").
                         Used as the key in KVCache.store() / get().
        depth_category : "deep" | "shallow" | None
                         Set by patch_unet._classify_depth() based on the
                         layer's U-Net position and spatial resolution.
                         None = skip entirely (no store, no inject).
        All other args/kwargs passed through to diffusers Attention.__init__.
    """

    def __init__(
        self,
        *args,
        layer_name:     str = "",
        depth_category       = None,   # "deep" | "shallow" | None
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.layer_name     = layer_name
        self.depth_category = depth_category

    def forward(
        self,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask        = None,
        **cross_attention_kwargs,
    ):
        # ── Extract cache — never forward it to super() ───────────────────
        kv_cache = cross_attention_kwargs.pop("kv_cache", None)

        # Only self-attention is modified; cross-attention passes through
        is_self_attn = (encoder_hidden_states is None)

        # Skip conditions — fall through to standard diffusers attention:
        #   • no cache attached
        #   • cross-attention layer (encoder_hidden_states is not None)
        #   • cache is in bypass mode
        #   • layer is unclassified (depth_category is None — mid-freq ambiguous)
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

        # ── Project Q, K, V ───────────────────────────────────────────────
        B, seq_len, _ = hidden_states.shape

        # Apply pre-norm if present (SD2 uses group norm inside attn)
        norm_hs = (
            self.norm_cross(hidden_states)
            if hasattr(self, "norm_cross") and self.norm_cross is not None
            else hidden_states
        )

        q = self.to_q(norm_hs)
        k = self.to_k(norm_hs)
        v = self.to_v(norm_hs)

        hd = q.shape[-1] // self.heads   # head_dim
        q  = q.view(B, seq_len, self.heads, hd).transpose(1, 2)   # (B, h, S, d)
        k  = k.view(B, seq_len, self.heads, hd).transpose(1, 2)
        v  = v.view(B, seq_len, self.heads, hd).transpose(1, 2)

        # ── Store mode ────────────────────────────────────────────────────
        # Write current K,V into the active sub-cache (lf or hf).
        # Then run standard attention — no injection during store passes.
        # freq_mode on the cache tells store() which sub-cache to write to.
        if kv_cache.mode == "store":
            kv_cache.store(self.layer_name, k, v)
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )
            return self._out_proj(out, q, B, seq_len)

        # ── Inject mode ───────────────────────────────────────────────────
        if kv_cache.mode == "inject":

            # Resolve which freq sub-cache and weight to use for this layer.
            # This is where the depth_routing ablation flag takes effect.
            depth_routing = getattr(kv_cache, "depth_routing", "correct")
            freq, weight  = _resolve_freq_and_weight(
                self.depth_category, depth_routing, kv_cache
            )

            k_ref, v_ref = kv_cache.get(self.layer_name, freq)

            if k_ref is not None and weight > 0.0:

                # CFG doubles batch size — tile the stored reference to match
                if k_ref.shape[0] != B:
                    k_ref = k_ref.expand(B, -1, -1, -1)
                    v_ref = v_ref.expand(B, -1, -1, -1)

                if kv_cache.face_mask is not None:
                    # Spatial gating: inject only inside the face region
                    token_mask = _resize_mask_to_tokens(
                        kv_cache.face_mask, seq_len, k.device
                    )
                    if token_mask is not None:
                        # Expand to (B, heads, seq_len, head_dim) for broadcast
                        m = token_mask.view(1, 1, seq_len, 1).expand(
                            B, self.heads, seq_len, hd
                        )
                        # Additive blend: source K,V nudged toward reference
                        # inside the face mask, unchanged outside.
                        # weight=0 → no change; weight=1 → full replacement.
                        k = k + weight * torch.where(m, k_ref - k, torch.zeros_like(k))
                        v = v + weight * torch.where(m, v_ref - v, torch.zeros_like(v))
                    # else: non-square seq — skip mask, k/v unchanged this layer

                else:
                    # No spatial mask — global injection across all tokens
                    # Equivalent to ablation.mask_type = "none"
                    k = k + weight * (k_ref - k)
                    v = v + weight * (v_ref - v)

        # ── Scaled dot-product attention ──────────────────────────────────
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        return self._out_proj(out, q, B, seq_len)

    # ── Output projection helper ──────────────────────────────────────────────

    def _out_proj(
        self,
        out:     torch.Tensor,   # (B, heads, seq_len, head_dim)
        q:       torch.Tensor,   # used only for dtype and shape reference
        B:       int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Merge attention heads and apply output projection.

        (B, heads, seq_len, head_dim) → (B, seq_len, inner_dim) → to_out
        """
        inner_dim = q.shape[1] * q.shape[-1]   # heads × head_dim
        out = out.transpose(1, 2).contiguous().reshape(B, seq_len, inner_dim)
        out = out.to(q.dtype)
        out = self.to_out[0](out)   # linear projection
        out = self.to_out[1](out)   # dropout (identity at eval time)
        return out