# kv_cache.py
class KVCache:
    """
    Stores K_ref, V_ref tensors keyed by attention layer name.
    Mode switches between 'store' (reference pass) and 'inject' (source pass).
    """
    def __init__(self):
        self.cache = {}          # {layer_name: (K_ref, V_ref)}
        self.mode = "bypass"     # "store" | "inject" | "bypass"
        self.face_mask = None    # (1, 1, H, W) float tensor, pixel space

    def set_mode(self, mode: str):
        assert mode in ("store", "inject", "bypass")
        self.mode = mode

    def store(self, layer_name: str, k: "Tensor", v: "Tensor"):
        self.cache[layer_name] = (k.detach().clone(), v.detach().clone())

    def get(self, layer_name: str):
        return self.cache.get(layer_name, (None, None))

    def clear(self):
        self.cache.clear()