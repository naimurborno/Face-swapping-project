# core/kv_cache.py
"""
Dual-store KV cache with frequency separation and temporal lambda annealing.

Almost identical to the original kv_cache.py. The one addition is that
set_lambdas() now accepts temporal_anneal and injection_scale arguments so
the A4 ablation flag (configs/default.yaml → ablation.temporal_anneal) is
respected without any logic changes in stage2_diffusion.py.

Modes:
    "bypass"  — passthrough, no-op (default at construction)
    "store"   — capture K,V into the active freq sub-cache (lf or hf)
    "inject"  — inject cached K,V into the denoising pass

freq_mode: "lf" | "hf"
    Active during mode="store". Routes K,V to the correct sub-cache.
    Set via set_freq_mode() before each store pass.

lambda_lf / lambda_hf:
    Per-step injection weights. Updated each denoising step via set_lambdas().
    Deep layers  (8×8 / 16×16) read lambda_lf.
    Shallow layers (64×64)     read lambda_hf.

face_mask: (1, 1, H, W) float32 [0, 1]
    Spatial injection mask. 1 = inject, 0 = suppress.
    Set from the SAM / convex hull mask produced by stage1_segment.py.
    If None, injection is global (equivalent to ablation.mask_type = "none").
"""

import torch


class KVCache:

    def __init__(self):
        self._lf_cache: dict = {}    # layer_name → (k, v)
        self._hf_cache: dict = {}    # layer_name → (k, v)
        self.mode:      str  = "bypass"
        self.freq_mode: str  = "lf"  # active store target during "store" mode
        self.lambda_lf: float = 1.0
        self.lambda_hf: float = 0.0
        self.face_mask        = None  # (1, 1, H, W) float32 [0,1] or None

    # ── Mode control ──────────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        """
        Switch the cache operating mode.

        Args:
            mode : "store"  — next U-Net forward captures K,V into sub-cache
                   "inject" — next U-Net forward blends cached K,V into current
                   "bypass" — next U-Net forward runs normally (no cache I/O)
        """
        assert mode in ("store", "inject", "bypass"), \
            f"[kv_cache] Invalid mode '{mode}'. Choose: store | inject | bypass"
        self.mode = mode

    def set_freq_mode(self, freq: str):
        """
        Set which sub-cache receives K,V during the next store pass.

        Must be called before set_mode("store"). Has no effect during inject
        or bypass passes.

        Args:
            freq : "lf" — low-frequency store pass (LF reference forward)
                   "hf" — high-frequency store pass (HF reference forward)
        """
        assert freq in ("lf", "hf"), \
            f"[kv_cache] Invalid freq_mode '{freq}'. Choose: lf | hf"
        self.freq_mode = freq

    def set_lambdas(
        self,
        step_idx:        int,
        total_steps:     int,
        temporal_anneal: bool  = True,
        injection_scale: float = 1.0,
    ):
        """
        Update lambda_lf and lambda_hf for the current denoising step.

        Called once per step in stage2_diffusion.py before the store passes.
        The temporal_anneal and injection_scale arguments map directly to
        configs/default.yaml:
            ablation.temporal_anneal  → temporal_anneal
            injection.injection_scale → injection_scale

        Temporal annealing (temporal_anneal=True, ablation A4 default):
            Progress t = step_idx / (total_steps - 1), ranging 0→1
            as the denoiser moves from high noise (t=0) to low noise (t=1).

                λ_LF(t) = (1 - t) × injection_scale
                λ_HF(t) = t       × injection_scale

            Rationale: at high noise, coarse structure (LF) dominates the
            denoising signal — injecting LF features strongly here guides
            identity shape. At low noise, fine detail (HF) matters most —
            injecting HF features late captures iris, pores, and texture.

        Flat lambdas (temporal_anneal=False, ablation A4 comparison):
            λ_LF = λ_HF = injection_scale throughout all steps.
            This is the Phase 2 baseline schedule — no frequency-aware timing.
            Ablation A4 compares annealed vs flat to validate the annealing.

        Args:
            step_idx        : Current denoising step index (0-based).
            total_steps     : Total number of denoising steps.
            temporal_anneal : True = annealed schedule (default).
                              False = flat lambdas (ablation A4 comparison).
            injection_scale : Global lambda multiplier applied to both lambdas.
                              Reads from injection.injection_scale in default.yaml.
                              Ablation A6 sweeps this over [0.3, 0.5, 0.8, 1.0].
        """
        if temporal_anneal:
            progress       = step_idx / max(total_steps - 1, 1)
            self.lambda_lf = (1.0 - progress) * injection_scale
            self.lambda_hf = progress         * injection_scale
        else:
            # Flat schedule — both lambdas constant throughout all steps
            self.lambda_lf = injection_scale
            self.lambda_hf = injection_scale

    # ── Cache operations ──────────────────────────────────────────────────────

    def store(self, key: str, k: torch.Tensor, v: torch.Tensor):
        """
        Write K,V tensors to the active sub-cache (lf or hf).

        Called by KVInjectionAttention during mode="store" passes.
        Tensors are detached and cloned so the computation graph is not held.

        Args:
            key : Layer name string (e.g. "down_blocks.0.attentions.0.attn1").
                  Used as the lookup key during inject passes.
            k   : (B, heads, seq_len, head_dim) key tensor.
            v   : (B, heads, seq_len, head_dim) value tensor.
        """
        target = self._lf_cache if self.freq_mode == "lf" else self._hf_cache
        target[key] = (k.detach().clone(), v.detach().clone())

    def get(self, key: str, freq: str):
        """
        Retrieve cached K,V tensors for a layer from one sub-cache.

        Called by KVInjectionAttention during mode="inject" passes.

        Args:
            key  : Layer name string — must match what was passed to store().
            freq : "lf" or "hf" — which sub-cache to read from.
                   Deep layers   pass "lf" (lambda_lf weighted).
                   Shallow layers pass "hf" (lambda_hf weighted).

        Returns:
            (k, v) tensor tuple if the key exists in the requested sub-cache.
            (None, None) if not found — caller skips injection gracefully.
        """
        source = self._lf_cache if freq == "lf" else self._hf_cache
        return source.get(key, (None, None))

    def clear(self):
        """
        Clear both sub-caches.

        Must be called at the start of each denoising step before the store
        passes so stale KV from the previous step is not accidentally injected.
        stage2_diffusion.py calls this at the top of the step loop.
        """
        self._lf_cache.clear()
        self._hf_cache.clear()

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line status string for verbose logging in stage2_diffusion.py."""
        return (
            f"KVCache | mode={self.mode} freq_mode={self.freq_mode} | "
            f"λ_LF={self.lambda_lf:.3f} λ_HF={self.lambda_hf:.3f} | "
            f"LF keys={len(self._lf_cache)} HF keys={len(self._hf_cache)}"
        )