# kv_cache.py
import torch


class KVCache:
    """
    Phase 3 upgrade: dual-store KV cache with frequency separation.

    Modes:
        "bypass"  — passthrough, no-op (default)
        "store"   — capture K,V into the active freq store (lf or hf)
        "inject"  — inject cached K,V into the denoising pass

    freq_mode: "lf" | "hf"
        Active during mode="store". Determines which sub-cache receives the K,V.

    lambda_lf / lambda_hf:
        Injection weights updated each denoising step via set_lambdas().
        Deep layers use lambda_lf, shallow layers use lambda_hf.

    face_mask: (1, 1, H, W) float [0,1] — spatial injection mask.
    """

    def __init__(self):
        self._lf_cache: dict = {}   # layer_name → (k, v)
        self._hf_cache: dict = {}   # layer_name → (k, v)
        self.mode: str = "bypass"
        self.freq_mode: str = "lf"  # active store target during "store" mode
        self.lambda_lf: float = 1.0
        self.lambda_hf: float = 0.0
        self.face_mask = None       # (1, 1, H, W) float [0,1]

    # ── Mode control ──────────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        assert mode in ("store", "inject", "bypass"), \
            f"Invalid mode '{mode}'. Choose: store | inject | bypass"
        self.mode = mode

    def set_freq_mode(self, freq: str):
        """Call before a store pass to direct K,V into the right sub-cache."""
        assert freq in ("lf", "hf"), \
            f"Invalid freq_mode '{freq}'. Choose: lf | hf"
        self.freq_mode = freq

    def set_lambdas(self, step_idx: int, total_steps: int):
        """
        Temporal annealing — call once per denoising step before store passes.

        Progress 0→1 as step_idx goes 0→total_steps-1 (high noise → low noise).

            λ_LF = 1 - progress  (dominant early: coarse structure at high noise)
            λ_HF = progress      (dominant late: fine detail at low noise)
        """
        progress = step_idx / max(total_steps - 1, 1)
        self.lambda_lf = 1.0 - progress
        self.lambda_hf = progress

    # ── Cache operations ──────────────────────────────────────────────────────

    def store(self, key: str, k: torch.Tensor, v: torch.Tensor):
        """Write K,V to whichever sub-cache is active (lf or hf)."""
        target = self._lf_cache if self.freq_mode == "lf" else self._hf_cache
        target[key] = (k.detach().clone(), v.detach().clone())

    def get(self, key: str, freq: str):
        """
        Retrieve K,V for a given layer from the requested sub-cache.

        Args:
            key:  layer identifier (layer name string)
            freq: "lf" or "hf" — which sub-cache to read from

        Returns:
            (k, v) tensors or (None, None) if not cached.
        """
        source = self._lf_cache if freq == "lf" else self._hf_cache
        return source.get(key, (None, None))

    def clear(self):
        """Clear both sub-caches. Call before each new store pass pair."""
        self._lf_cache.clear()
        self._hf_cache.clear()

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        return (
            f"KVCache | mode={self.mode} freq_mode={self.freq_mode} | "
            f"λ_LF={self.lambda_lf:.3f} λ_HF={self.lambda_hf:.3f} | "
            f"LF keys={len(self._lf_cache)} HF keys={len(self._hf_cache)}"
        )