# attention_processor.py
import torch
import torch.nn.functional as F
from mask_utils import mask_to_token_mask


class KVInjectionProcessor:
    """
    Matches the contract of diffusers AttnProcessor2_0.
    The Attention module handles residual + layernorm externally.
    This processor ONLY handles: project → (optionally inject K,V) → attend → out_proj.
    """

    def __init__(self, layer_name: str, kv_cache):
        self.layer_name = layer_name
        self.kv_cache = kv_cache

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        B, seq_len, _ = hidden_states.shape

        # SD2 self-attention has a spatial norm; handle if present
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)
            seq_len = H * W

        # Layernorm before projection (if present inside attn)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Self-attention: kv_input = hidden_states
        kv_input = hidden_states if encoder_hidden_states is None \
                   else encoder_hidden_states

        # ── Project Q, K, V ──
        q = attn.to_q(hidden_states)
        k = attn.to_k(kv_input)
        v = attn.to_v(kv_input)

        # ── Reshape to (B, heads, seq, head_dim) ──
        inner_dim = k.shape[-1]
        head_dim  = inner_dim // attn.heads

        q = q.view(B, seq_len, attn.heads, head_dim).transpose(1, 2)  # (B, h, S, d)
        k = k.view(B, seq_len, attn.heads, head_dim).transpose(1, 2)
        v = v.view(B, seq_len, attn.heads, head_dim).transpose(1, 2)

        # ── KV Cache logic ──
        mode = self.kv_cache.mode

        if mode == "store":
            self.kv_cache.store(self.layer_name, k, v)

        elif mode == "inject":
            k_ref, v_ref = self.kv_cache.get(self.layer_name)

            if k_ref is not None:
                if self.kv_cache.face_mask is not None:
                    spatial_size = int(seq_len ** 0.5)

                    if spatial_size * spatial_size == seq_len:
                        token_mask = mask_to_token_mask(
                            self.kv_cache.face_mask.to(k.device),
                            spatial_size=spatial_size,
                        )  # (seq_len,) bool

                        # Expand to (B, heads, seq_len, 1)
                        mask_exp = token_mask.view(1, 1, seq_len, 1).expand(
                            B, attn.heads, seq_len, head_dim
                        )

                        # k_ref/v_ref shape: (B, heads, seq_len, head_dim)
                        # If batch sizes differ (CFG doubles batch), handle it
                        if k_ref.shape[0] != B:
                            # CFG: ref was run with B, source is 2B — tile ref
                            k_ref = k_ref.expand(B, -1, -1, -1)
                            v_ref = v_ref.expand(B, -1, -1, -1)

                        k = torch.where(mask_exp, k_ref, k)
                        v = torch.where(mask_exp, v_ref, v)
                else:
                    # Global injection (debug mode — no mask)
                    if k_ref.shape[0] != B:
                        k_ref = k_ref.expand(B, -1, -1, -1)
                        v_ref = v_ref.expand(B, -1, -1, -1)
                    k = k_ref
                    v = v_ref

        # ── Scaled dot-product attention (flash-attn path) ──
        # Use PyTorch 2.0 built-in if available (faster, less VRAM)
        if hasattr(F, "scaled_dot_product_attention"):
            hidden_states = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )  # (B, heads, seq, head_dim)
        else:
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            hidden_states = torch.matmul(attn_weights, v)

        # ── Merge heads: (B, heads, seq, head_dim) → (B, seq, inner_dim) ──
        hidden_states = hidden_states.transpose(1, 2).reshape(B, seq_len, inner_dim)
        hidden_states = hidden_states.to(q.dtype)

        # ── Output projection ──
        hidden_states = attn.to_out[0](hidden_states)   # linear
        hidden_states = attn.to_out[1](hidden_states)   # dropout (identity at eval)

        # Reshape back if input was 4D
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).view(B, C, H, W)

        return hidden_states