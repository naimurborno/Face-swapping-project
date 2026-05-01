# core/kv_cache.py
"""
Dual-store KV cache for PnP per-step injection.

Supports two operating paths:

  STORE PATH (used by stage1b_invert.py during DDIM inversion):
      set_freq_mode("hf") → set_mode("store")
      KVInjectionAttention.forward() calls cache.store() at each layer.
      Caller then calls cache.save_step() to flush _hf_cache to disk
      as artifacts/kv_store/step{idx:03d}_{layer_name}.pt.

  LOAD PATH (used by stage2_diffusion.py during denoising):
      load_kv_for_timestep() in stage2 calls cache._hf_cache directly,
      writing (k, v) tensors it loaded from disk.
      set_mode("inject") → KVInjectionAttention reads _hf_cache per layer.

The old per-run store-then-inject pattern (store once before the loop,
inject every step) is gone. stage2 never calls set_mode("store") —
the _hf_cache is populated by the load path instead.

The _lf_cache and LF-related lambda (lambda_lf) are retained for
interface compatibility with KVInjectionAttention and for the depth_routing
ablation. In the current pipeline lambda_lf is always 0.0 and _lf_cache
is always empty during denoising — only _hf_cache carries live data.

Modes:
    "bypass"  — passthrough, no cache I/O (default at construction)
    "store"   — capture K,V into the active freq sub-cache (stage1b only)
    "inject"  — inject cached K,V into the denoising pass (stage2)

freq_mode: "lf" | "hf"
    Routing target during mode="store". Always "hf" in the current pipeline
    because only shallow (HF) layers are patched. Kept for ablation parity
    with the depth_routing logic in kv_attention.py.

lambda_lf / lambda_hf:
    Per-step injection weights read by KVInjectionAttention.
    lambda_lf is always 0.0 in stage2 (no deep layer injection).
    lambda_hf = injection_scale (flat, principled default) or ramped if
    temporal_anneal=True (ablation A4).

face_mask: (1, 1, H, W) float32 [0, 1]
    Spatial injection mask. 1 = inject, 0 = suppress.
    Set from object_mask.pt produced by stage1_segment.py.
    If None, injection is global (ablation.mask_type = "none").

depth_routing: "correct" | "swapped" | "uniform"
    Read by KVInjectionAttention._resolve_freq_and_weight().
    Always "correct" in stage2 (shallow → HF). Kept for ablation A2.
"""

import os
import torch


class KVCache:

    def __init__(self):
        self._lf_cache: dict  = {}     # layer_name → (k, v)  — unused in stage2
        self._hf_cache: dict  = {}     # layer_name → (k, v)  — populated per step
        self.mode:      str   = "bypass"
        self.freq_mode: str   = "hf"
        self.lambda_lf: float = 0.0    # always 0 in stage2 (no deep injection)
        self.lambda_hf: float = 1.0
        self.face_mask        = None   # (1, 1, H, W) float32 [0,1] or None
        self.depth_routing: str = "correct"

    # ── Mode control ──────────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        """
        Switch the cache operating mode.

        Args:
            mode : "store"  — next UNet forward captures K,V (stage1b)
                   "inject" — next UNet forward blends cached K,V (stage2)
                   "bypass" — next UNet forward runs normally (no cache I/O)
        """
        assert mode in ("store", "inject", "bypass"), \
            f"[kv_cache] Invalid mode '{mode}'. Choose: store | inject | bypass"
        self.mode = mode

    def set_freq_mode(self, freq: str):
        """
        Set which sub-cache receives K,V during the next store pass.

        Only relevant during mode="store" (stage1b). Has no effect during
        inject or bypass passes.

        Args:
            freq : "lf" — low-frequency sub-cache
                   "hf" — high-frequency sub-cache (always used in stage1b)
        """
        assert freq in ("lf", "hf"), \
            f"[kv_cache] Invalid freq_mode '{freq}'. Choose: lf | hf"
        self.freq_mode = freq

    def set_lambdas(
        self,
        step_idx:        int,
        total_steps:     int,
        temporal_anneal: bool  = False,
        injection_scale: float = 1.0,
    ):
        """
        Update lambda_lf and lambda_hf for the current denoising step.

        Called once per step in stage2_diffusion.py before the inject pass.

        Default behaviour (temporal_anneal=False) — flat schedule:
            lambda_lf = 0.0              (deep injection disabled)
            lambda_hf = injection_scale  (constant throughout all steps)

        Flat is the principled default in the PnP pipeline because KV tensors
        are already timestep-matched from DDIM inversion. No heuristic ramp
        is needed to compensate for stale features.

        Temporal annealing (temporal_anneal=True, ablation A4):
            progress    = step_idx / (total_steps - 1)   [0 → 1]
            lambda_hf   = progress × injection_scale      [ramps up]
        Kept as an ablation to compare against flat. Expected to perform
        slightly worse than flat because it suppresses injection early when
        coarse structure is being decided — the opposite of what helped in
        the old static-cache pipeline.

        Args:
            step_idx        : Current denoising step index (0-based).
            total_steps     : Total number of active denoising steps.
            temporal_anneal : False = flat (default). True = ablation A4.
            injection_scale : Global λ multiplier. Ablation A6 sweeps
                              [0.3, 0.5, 0.8, 1.0].
        """
        self.lambda_lf = 0.0   # deep injection always disabled in stage2

        if temporal_anneal:
            progress       = step_idx / max(total_steps - 1, 1)
            self.lambda_hf = progress * injection_scale
        else:
            self.lambda_hf = injection_scale

    # ── Cache operations ──────────────────────────────────────────────────────

    def store(self, key: str, k: torch.Tensor, v: torch.Tensor):
        """
        Write K,V tensors to the active sub-cache (lf or hf).

        Called by KVInjectionAttention during mode="store" passes in stage1b.
        Tensors are detached and cloned so the computation graph is not held.

        Args:
            key : Layer name string used as lookup key.
            k   : (B, heads, seq_len, head_dim) key tensor.
            v   : (B, heads, seq_len, head_dim) value tensor.
        """
        target = self._lf_cache if self.freq_mode == "lf" else self._hf_cache
        target[key] = (k.detach().clone(), v.detach().clone())

    def get(self, key: str, freq: str):
        """
        Retrieve cached K,V tensors for a layer from one sub-cache.

        Called by KVInjectionAttention during mode="inject" passes.

        In stage2, freq is always "hf" because only shallow layers are
        patched and depth_routing="correct" maps shallow → hf.

        Args:
            key  : Layer name string — must match what was passed to store()
                   or what load_kv_for_timestep() wrote into _hf_cache.
            freq : "lf" or "hf" — which sub-cache to read from.

        Returns:
            (k, v) tensor tuple if found.
            (None, None) if not found — caller skips injection gracefully.
        """
        source = self._lf_cache if freq == "lf" else self._hf_cache
        return source.get(key, (None, None))

    def save_step(self, step_idx: int, kv_store_dir: str):
        """
        Flush the current _hf_cache to disk for one inversion step.

        Called by stage1b_invert.py after each DDIM inversion step to
        persist the captured K,V tensors. Files are named:

            kv_store/step{step_idx:03d}_{safe_layer_name}.pt

        where safe_layer_name replaces '.' with '_' to avoid path issues.
        Each file contains a dict {"k": tensor, "v": tensor} saved at the
        dtype of the stored tensors (float16 recommended for disk efficiency).

        The _hf_cache is NOT cleared after saving — stage1b calls clear()
        explicitly at the start of each inversion step.

        Args:
            step_idx    : Denoising step index (0-based). Must match the
                          step_idx used in stage2's load_kv_for_timestep().
            kv_store_dir: Destination directory (artifacts/kv_store/).
                          Created if it does not exist.
        """
        os.makedirs(kv_store_dir, exist_ok=True)

        for layer_name, (k, v) in self._hf_cache.items():
            safe_name = layer_name.replace(".", "_")
            fname     = f"step{step_idx:03d}_{safe_name}.pt"
            fpath     = os.path.join(kv_store_dir, fname)
            torch.save({"k": k.cpu(), "v": v.cpu()}, fpath)

    def clear(self):
        """
        Clear both sub-caches.

        Called by stage1b_invert.py at the start of each inversion step
        before the UNet forward so stale KV from the previous step is not
        accidentally captured or injected.

        stage2_diffusion.py does not call this — load_kv_for_timestep()
        calls _hf_cache.clear() directly before populating it, so the
        cache is always fresh at injection time without needing an explicit
        clear() call in the loop.
        """
        self._lf_cache.clear()
        self._hf_cache.clear()

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line status string for verbose logging."""
        return (
            f"KVCache | mode={self.mode} freq_mode={self.freq_mode} | "
            f"λ_LF={self.lambda_lf:.3f} λ_HF={self.lambda_hf:.3f} | "
            f"LF keys={len(self._lf_cache)} HF keys={len(self._hf_cache)}"
        )

    def hf_keys(self) -> list:
        """Return list of layer names currently in _hf_cache."""
        return list(self._hf_cache.keys())

    def lf_keys(self) -> list:
        """Return list of layer names currently in _lf_cache."""
        return list(self._lf_cache.keys())