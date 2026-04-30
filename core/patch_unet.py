# core/patch_unet.py
"""
U-Net patching for the KV-injection face swap pipeline.

Two things happen here:
    1. Every attn1 (self-attention) Attention module is replaced with a
       KVInjectionAttention instance with identical weights. This is the
       module that implements the store / inject / bypass logic.

    2. Every attn2 (cross-attention) Attention module gets a standalone
       _KwargSafeProcessor swapped onto it. This processor reimplements the
       standard flash-attention path directly — it does NOT inherit from
       AttnProcessor2_0. This is the fix for the repeated warning:
           "cross_attention_kwargs ['kv_cache'] are not expected by
            AttnProcessor2_0 and will be ignored."
       The warning fires because newer diffusers versions do strict keyword
       inspection on the parent's __call__ signature. A standalone processor
       with its own __call__ receives kv_cache cleanly, pops it, and runs
       standard attention with no warning at all.

Depth classification (SD 2.1-base, 512px):
    Spatial resolution per U-Net block determines depth category:

    Block             Resolution   Category   Injection
    ──────────────────────────────────────────────────
    down_blocks.0     64 × 64      shallow    HF cache
    down_blocks.1     32 × 32      (skipped)  —
    down_blocks.2     16 × 16      deep       LF cache
    mid_block          8 × 8       deep       LF cache
    up_blocks.1       16 × 16      deep       LF cache
    up_blocks.2       32 × 32      (skipped)  —
    up_blocks.3       64 × 64      shallow    HF cache

    Skipped blocks (32×32) are mid-frequency ambiguous — assigning them to
    either LF or HF would introduce noise into the ablation. They are left
    with depth_category=None and fall through to standard attention.

    The deep/shallow prefix lists are also exposed in default.yaml under
    injection.depth so they can be inspected, but patch_unet.py uses its own
    hardcoded tuples (which match the yaml) rather than reading the config.
    This keeps the patching function self-contained and importable without
    loading the config.

Public API:
    patch_unet_attention(unet) → (unet, depth_map)
        Full patch: every classified attn1 layer gets KVInjectionAttention.
        Used by the Phase 3 pipeline (default) and ablation A3 "full" condition.
        depth_map : dict[layer_name → "deep" | "shallow" | None]

    patch_unet_shallow_only(unet) → (unet, depth_map)
        Shallow-only patch: only down_blocks.0 / up_blocks.3 (64×64) layers get
        KVInjectionAttention. Deep layers (mid_block, down_blocks.2, up_blocks.1)
        receive _KwargSafeProcessor only — standard attention, no injection.
        Used by the Mixed-Frequency Prior Guided Inpainting pipeline (Phase 4),
        where source geometry is enforced by per-step blended latent anchoring
        rather than by deep-layer KV injection, making deep injection redundant
        and potentially harmful.
        depth_map : dict[layer_name → "deep" | "shallow" | None]
                    Same schema as patch_unet_attention() — deep layers are still
                    classified and logged, just not patched with KVInjectionAttention.
"""

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from core.kv_attention import KVInjectionAttention


# ── Depth classification ──────────────────────────────────────────────────────

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
    Return the depth category for a self-attention layer by name prefix.

    Returns:
        "deep"    — 8×8 or 16×16 spatial resolution (LF injection)
        "shallow" — 64×64 spatial resolution (HF injection)
        None      — 32×32 spatial resolution (skipped; mid-freq ambiguous)
    """
    for prefix in _DEEP_PREFIXES:
        if layer_name.startswith(prefix):
            return "deep"
    for prefix in _SHALLOW_PREFIXES:
        if layer_name.startswith(prefix):
            return "shallow"
    return None


# ── Standalone cross-attention processor ─────────────────────────────────────
#
# WHY standalone and not a subclass of AttnProcessor2_0:
#
# diffusers >= 0.26 introduced strict keyword argument validation in the
# Attention.forward() dispatch path. When a processor is called, diffusers
# inspects the processor's __call__ signature and emits a warning for every
# kwarg that appears in cross_attention_kwargs but not in the signature.
#
# The old fix (inherit AttnProcessor2_0, pop kv_cache before super()) does
# not work because diffusers inspects the *parent's* __call__ signature, not
# the overriding child's. The parent does not list kv_cache → warning fires.
#
# The correct fix is a fully standalone processor. Its __call__ signature
# accepts **kwargs explicitly (or lists kv_cache directly), so diffusers'
# inspection finds the kwarg and emits no warning. The processor pops
# kv_cache and implements the flash-attention path itself — no super() call,
# no inherited signature, no warning.

class _KwargSafeProcessor:
    """
    Standalone cross-attention processor that silently absorbs kv_cache.

    Applied to every attn2 (cross-attention) layer by patch_unet_attention().
    These layers play no role in KV injection — only self-attention (attn1)
    layers participate. This processor exists purely to prevent the repeated
    diffusers warning when kv_cache is passed in cross_attention_kwargs.

    Also applied to deep attn1 layers by patch_unet_shallow_only(). In that
    context it serves a dual purpose: suppress the kv_cache warning AND leave
    the deep self-attention computation completely unmodified, so the blended
    latent anchoring mechanism (not KV injection) drives structural fidelity.

    Implements the same scaled dot-product attention path as AttnProcessor2_0
    (PyTorch 2.0 F.scaled_dot_product_attention) with no inheritance.
    """

    def __call__(
        self,
        attn,
        hidden_states:          torch.Tensor,
        encoder_hidden_states:  torch.Tensor = None,
        attention_mask:         torch.Tensor = None,
        temb:                   torch.Tensor = None,
        # kv_cache is listed explicitly so diffusers' kwarg inspector finds it
        kv_cache                             = None,
        **kwargs,                                      # absorb any future extras
    ) -> torch.Tensor:
        """
        Standard cross-attention forward pass.

        kv_cache is accepted and immediately discarded — cross-attention layers
        are never modified by the injection pipeline.
        """
        # kv_cache received and discarded — nothing to do with it here
        _ = kv_cache

        residual = hidden_states

        # ── Spatial norm (SD2 cross-attention has this; SD1 does not) ────
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        B, seq_len, _ = hidden_states.shape

        # ── Group norm (if present inside attn) ──────────────────────────
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        # ── Project Q, K, V ───────────────────────────────────────────────
        kv_input = hidden_states if encoder_hidden_states is None \
                   else encoder_hidden_states

        q = attn.to_q(hidden_states)
        k = attn.to_k(kv_input)
        v = attn.to_v(kv_input)

        # ── Reshape to multi-head format ──────────────────────────────────
        inner_dim = k.shape[-1]
        head_dim  = inner_dim // attn.heads

        # kv seq_len may differ from q seq_len in cross-attention
        kv_seq = k.shape[1] if k.ndim == 3 else k.shape[2]

        q = q.view(B, seq_len, attn.heads, head_dim).transpose(1, 2)
        k = k.view(B, kv_seq,  attn.heads, head_dim).transpose(1, 2)
        v = v.view(B, kv_seq,  attn.heads, head_dim).transpose(1, 2)

        # ── Scaled dot-product attention ──────────────────────────────────
        hidden_states = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )   # (B, heads, seq, head_dim)

        # ── Merge heads ───────────────────────────────────────────────────
        hidden_states = (
            hidden_states
            .transpose(1, 2)
            .reshape(B, seq_len, inner_dim)
            .to(q.dtype)
        )

        # ── Output projection ─────────────────────────────────────────────
        hidden_states = attn.to_out[0](hidden_states)   # linear
        hidden_states = attn.to_out[1](hidden_states)   # dropout (identity at eval)

        # ── Reshape back if input was 4D ──────────────────────────────────
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).view(B, C, H, W)

        # ── Residual (Attention module handles this externally in diffusers,
        #    but some processor implementations add it here for safety) ────
        # diffusers Attention.forward() adds the residual after calling the
        # processor, so we do NOT add it here to avoid double-adding.

        return hidden_states


# ── Main patching function ────────────────────────────────────────────────────

def patch_unet_attention(unet):
    """
    Patch all self-attention and cross-attention layers in the U-Net.

    Self-attention (attn1):
        Each Attention module is replaced with a KVInjectionAttention instance.
        Weights are copied via load_state_dict() so the model's behaviour is
        identical to the original until a KVCache is passed.

    Cross-attention (attn2):
        A _KwargSafeProcessor is swapped onto each cross-attention module via
        set_processor(). The module itself (weights, structure) is unchanged.
        The processor absorbs kv_cache silently without warnings.

    This function is idempotent when called on an already-patched U-Net:
        KVInjectionAttention modules will not match isinstance(module, Attention)
        ... actually they DO since KVInjectionAttention inherits from Attention.
        To prevent double-patching, patch_unet_attention() checks for existing
        KVInjectionAttention instances and skips them.

    Used by:
        Phase 3 pipeline (default) — full frequency-decomposed KV injection.
        Ablation A3 "full" condition — compare full vs shallow-only injection.

    Args:
        unet : The diffusers UNet2DConditionModel. Mutated in-place.

    Returns:
        unet      : The same unet, mutated.
        depth_map : dict[str → "deep" | "shallow" | None]
                    Maps every patched attn1 layer_name to its depth category.
                    Passed to stage2_diffusion.py for logging / verification.
    """
    patched   = 0
    skipped   = 0
    depth_map = {}

    for name, module in unet.named_modules():

        # ── Skip already-patched self-attention layers ────────────────────
        # KVInjectionAttention IS an Attention subclass, so isinstance passes.
        # We check the specific type to avoid double-patching.
        if type(module) is KVInjectionAttention:
            skipped += 1
            continue

        if not isinstance(module, Attention):
            continue

        is_self_attn  = name.endswith(".attn1")
        is_cross_attn = name.endswith(".attn2")

        # ── Self-attention: replace module ────────────────────────────────
        if is_self_attn:
            depth = _classify_depth(name)
            depth_map[name] = depth

            # Navigate to parent module to call setattr
            parts  = name.split(".")
            parent = unet
            for p in parts[:-1]:
                parent = getattr(parent, p)

            device = next(module.parameters()).device
            dtype  = next(module.parameters()).dtype

            new_attn = KVInjectionAttention(
                query_dim           = module.to_q.in_features,
                cross_attention_dim = None,
                heads               = module.heads,
                dim_head            = module.to_q.out_features // module.heads,
                dropout             = 0.0,
                bias                = module.to_q.bias is not None,
                upcast_attention    = module.upcast_attention,
                out_bias            = module.to_out[0].bias is not None,
                layer_name          = name,
                depth_category      = depth,
            ).to(device, dtype=dtype)

            # Copy all weights — projection matrices, norms, output proj
            new_attn.load_state_dict(module.state_dict())
            new_attn.eval()

            setattr(parent, parts[-1], new_attn)
            patched += 1

        # ── Cross-attention: swap processor only ──────────────────────────
        elif is_cross_attn:
            module.set_processor(_KwargSafeProcessor())

    # ── Summary ───────────────────────────────────────────────────────────
    n_deep    = sum(1 for v in depth_map.values() if v == "deep")
    n_shallow = sum(1 for v in depth_map.values() if v == "shallow")
    n_skip    = sum(1 for v in depth_map.values() if v is None)

    if skipped > 0:
        print(
            f"[patch_unet] Already patched — skipped {skipped} "
            f"KVInjectionAttention layers."
        )
    else:
        print(
            f"[patch_unet] Patched {patched} attn1 layers | "
            f"deep={n_deep}  shallow={n_shallow}  skipped(mid-freq)={n_skip}"
        )

    return unet, depth_map


# ── Shallow-only patching function ────────────────────────────────────────────

def patch_unet_shallow_only(unet):
    """
    Patch only the shallow self-attention layers (64×64 spatial resolution).

    Used by the Mixed-Frequency Prior Guided Inpainting pipeline (Phase 4).

    Rationale
    ─────────
    In Phase 4, source geometry preservation is guaranteed by blended latent
    anchoring — at every denoising step the non-mask region of the latent is
    overwritten with the noised source latent z_S. This makes deep-layer KV
    injection redundant: the structural signal that deep layers would have
    carried is already enforced at the latent level, step-by-step.

    Patching deep layers with KVInjectionAttention in this setting would be
    actively harmful: it would push the U-Net's structural decisions toward the
    chimera/prior reference rather than letting the blended anchoring do its
    job, potentially fighting the blended latent and producing incoherent output.

    Shallow layers (down_blocks.0, up_blocks.3 — 64×64 spatial) are still
    patched because they carry donor fine-detail (T_HF) — the texture, pores,
    veining — which blended latent anchoring does not deliver. These layers are
    only active during the inject pass; the donor store pass runs once before
    the denoising loop and populates kv_cache._hf_cache at shallow keys only.

    Deep attn1 layers (mid_block, down_blocks.2, up_blocks.1) receive
    _KwargSafeProcessor instead of KVInjectionAttention. This gives them:
        • The correct standard-attention behaviour (no injection side-effects)
        • Silent absorption of the kv_cache kwarg (no diffusers warnings)
    Their weights are never touched — module.set_processor() replaces the
    processor only, leaving all projection matrices intact.

    Mid-frequency (32×32) attn1 layers classified as None fall through to
    a _KwargSafeProcessor via the cross-attention path — same as in the full
    patch. These are unclassified and receive no KV injection in either
    pipeline variant.

    Idempotency
    ───────────
    Same guard as patch_unet_attention(): already-patched KVInjectionAttention
    instances at shallow layers are detected by type check and skipped. Calling
    this function twice on the same unet is safe.

    Ablation use
    ────────────
    To isolate the contribution of shallow KV injection in Phase 4, set
    ablation.shallow_injection: false in default.yaml — stage2_diffusion.py
    will call patch_unet_attention() with injection_scale=0 or skip patching
    entirely, giving a pure blended-latent-inpainting baseline.

    Args:
        unet : The diffusers UNet2DConditionModel (inpainting variant).
               Mutated in-place.

    Returns:
        unet      : The same unet, mutated.
        depth_map : dict[str → "deep" | "shallow" | None]
                    Same schema as patch_unet_attention(). Deep layers appear
                    in the map with their correct classification, but they are
                    NOT backed by KVInjectionAttention — the map entry indicates
                    classification only, not injection status. Callers should
                    use the "patched_shallow" key in the summary print to
                    distinguish from the full patch.
    """
    patched_shallow = 0   # attn1 layers replaced with KVInjectionAttention
    bypassed_deep   = 0   # attn1 deep layers given _KwargSafeProcessor only
    skipped         = 0   # already-patched KVInjectionAttention layers
    depth_map       = {}

    for name, module in unet.named_modules():

        # ── Skip already-patched shallow layers ───────────────────────────
        if type(module) is KVInjectionAttention:
            skipped += 1
            # Still record in depth_map so callers get a complete picture
            depth_map[name] = module.depth_category
            continue

        if not isinstance(module, Attention):
            continue

        is_self_attn  = name.endswith(".attn1")
        is_cross_attn = name.endswith(".attn2")

        if is_self_attn:
            depth = _classify_depth(name)
            depth_map[name] = depth

            if depth == "shallow":
                # ── Replace with KVInjectionAttention ─────────────────────
                # Identical construction path as patch_unet_attention().
                # depth_category="shallow" → inject pass reads _hf_cache,
                # weighted by lambda_hf (ramps 0→injection_scale over steps).
                parts  = name.split(".")
                parent = unet
                for p in parts[:-1]:
                    parent = getattr(parent, p)

                device = next(module.parameters()).device
                dtype  = next(module.parameters()).dtype

                new_attn = KVInjectionAttention(
                    query_dim           = module.to_q.in_features,
                    cross_attention_dim = None,
                    heads               = module.heads,
                    dim_head            = module.to_q.out_features // module.heads,
                    dropout             = 0.0,
                    bias                = module.to_q.bias is not None,
                    upcast_attention    = module.upcast_attention,
                    out_bias            = module.to_out[0].bias is not None,
                    layer_name          = name,
                    depth_category      = "shallow",
                ).to(device, dtype=dtype)

                new_attn.load_state_dict(module.state_dict())
                new_attn.eval()

                setattr(parent, parts[-1], new_attn)
                patched_shallow += 1

            else:
                # depth == "deep" or None (mid-freq 32×32)
                # ── Processor-only swap — no weight changes, no injection ──
                # _KwargSafeProcessor absorbs kv_cache silently and runs
                # standard scaled-dot-product attention. The module's learned
                # projection weights (to_q, to_k, to_v, to_out) are untouched.
                # Blended latent anchoring in the denoising loop drives
                # structural fidelity at these layers instead.
                module.set_processor(_KwargSafeProcessor())
                if depth == "deep":
                    bypassed_deep += 1
                # None (mid-freq) layers: processor swap still suppresses the
                # kv_cache warning but we don't count them separately.

        elif is_cross_attn:
            # Cross-attention: same treatment as in patch_unet_attention().
            module.set_processor(_KwargSafeProcessor())

    # ── Summary ───────────────────────────────────────────────────────────
    n_deep    = sum(1 for v in depth_map.values() if v == "deep")
    n_shallow = sum(1 for v in depth_map.values() if v == "shallow")
    n_none    = sum(1 for v in depth_map.values() if v is None)

    if skipped > 0:
        print(
            f"[patch_unet_shallow_only] Already patched — skipped {skipped} "
            f"KVInjectionAttention layers."
        )
    else:
        print(
            f"[patch_unet_shallow_only] "
            f"KVInjectionAttention: shallow={patched_shallow} | "
            f"_KwargSafeProcessor (bypassed): deep={bypassed_deep} | "
            f"classified total: deep={n_deep}  shallow={n_shallow}  "
            f"skipped(mid-freq)={n_none}"
        )

    return unet, depth_map