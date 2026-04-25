# attention_processor.py
import torch
import torch.nn.functional as F
from mask_utils import mask_to_token_mask


class KVInjectionProcessor:
    """
    Drop-in replacement for diffusers AttnProcessor2_0.
    Wired to a shared KVCache instance.

    On "store" mode: runs normal attention, saves K and V to cache.
    On "inject" mode: replaces K,V for masked tokens with cached ref K,V.
    On "bypass" mode: pure normal attention, no side effects.
    """

    def __init__(self, layer_name: str, kv_cache: "KVCache"):
        self.layer_name = layer_name
        self.kv_cache = kv_cache

    def __call__(
        self,
        attn,           # Attention module (has .to_q, .to_k, .to_v, etc.)
        hidden_states,  # (B, seq_len, dim)
        encoder_hidden_states=None,  # None for self-attention
        attention_mask=None,
        **kwargs,
    ):
        # Self-attention: query source is hidden_states itself
        residual = hidden_states
        B, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = attn.to_q(hidden_states)

        # For self-attention, key/value source = hidden_states
        kv_input = hidden_states if encoder_hidden_states is None \
                   else encoder_hidden_states
        k = attn.to_k(kv_input)
        v = attn.to_v(kv_input)

        # Reshape to multi-head format
        # attn.heads = number of attention heads
        head_dim = q.shape[-1] // attn.heads
        def split_heads(x):
            # (B, seq, dim) -> (B*heads, seq, head_dim)
            x = x.reshape(B, seq_len, attn.heads, head_dim)
            x = x.permute(0, 2, 1, 3)
            return x.reshape(B * attn.heads, seq_len, head_dim)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # ---- KV Cache logic ----
        mode = self.kv_cache.mode

        if mode == "store":
            # Save this layer's K,V from reference pass
            self.kv_cache.store(self.layer_name, k, v)

        elif mode == "inject" and self.layer_name in self.kv_cache.cache:
            k_ref, v_ref = self.kv_cache.get(self.layer_name)

            if self.kv_cache.face_mask is not None:
                # Compute spatial side length from seq_len
                spatial_size = int(seq_len ** 0.5)

                if spatial_size * spatial_size == seq_len:
                    # Build token mask for this resolution
                    token_mask = mask_to_token_mask(
                        self.kv_cache.face_mask.to(k.device),
                        spatial_size=spatial_size,
                    )  # (seq_len,) bool

                    # Expand mask for (B*heads, seq_len, head_dim)
                    mask_expanded = token_mask.unsqueeze(0).unsqueeze(-1)
                    # (1, seq_len, 1) -> broadcasts over batch*heads and head_dim

                    k = torch.where(mask_expanded, k_ref, k)
                    v = torch.where(mask_expanded, v_ref, v)
                else:
                    # Non-square sequence (e.g. cross-attn tokens) — skip injection
                    pass
            else:
                # No mask: global replacement (useful for debugging)
                k = k_ref
                v = v_ref

        # ---- Standard scaled dot-product attention ----
        scale = head_dim ** -0.5
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        hidden_states = torch.bmm(attn_weights, v)

        # Merge heads back
        hidden_states = hidden_states.reshape(B, attn.heads, seq_len, head_dim)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(B, seq_len, attn.heads * head_dim)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # dropout (identity at eval)

        return hidden_states