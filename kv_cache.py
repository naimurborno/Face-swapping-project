# kv_cache.py
import torch

class KVCache:
    def __init__(self):
        self.cache = {}
        self.mode = "bypass"   # "store" | "inject" | "bypass"
        self.face_mask = None  # (1, 1, H, W) float [0,1]

    def set_mode(self, mode):
        assert mode in ("store", "inject", "bypass")
        self.mode = mode

    def store(self, key, k, v):
        self.cache[key] = (k.detach().clone(), v.detach().clone())

    def get(self, key):
        return self.cache.get(key, (None, None))

    def clear(self):
        self.cache.clear()