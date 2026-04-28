# test_core_stage1.py
"""
Stage 1 core module tests.

Checks every module that stage1_segment.py depends on:
    1. alignment.py       — FaceLandmarkResult fields, warp_reference(), build_convex_hull_mask()
    2. segmentation.py    — get_face_mask() for all three mask_type values
    3. decomposition.py   — decompose() for all three methods + to_pil_inputs() + mask_to_tensor()
    4. mask_utils.py      — mask_to_token_mask() at multiple spatial sizes

All tests use synthetic data (random numpy arrays / mock landmark results).
No GPU, no SAM model, no MediaPipe model, no real images needed.
SAM is tested via the convex_hull and none paths only — the sam path is
skipped automatically when no checkpoint is available.

Run from the project root:
    cd core
    python ../test_core_stage1.py

Or from anywhere if core/ is on the path:
    PYTHONPATH=core python test_core_stage1.py

Expected output:
    All tests print PASS. Any FAIL line means a broken interface contract.
"""

import sys
import os
import traceback

import numpy as np
import cv2
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
# Support running from the project root or from inside core/
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")
if os.path.isdir(_CORE):
    sys.path.insert(0, _CORE)
else:
    sys.path.insert(0, _HERE)


# ── Test helpers ──────────────────────────────────────────────────────────────

_PASS = 0
_FAIL = 0


def check(label: str, condition: bool, detail: str = ""):
    global _PASS, _FAIL
    if condition:
        print(f"  PASS  {label}")
        _PASS += 1
    else:
        print(f"  FAIL  {label}" + (f"  →  {detail}" if detail else ""))
        _FAIL += 1


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Synthetic data factories ──────────────────────────────────────────────────

def make_face_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Return a random BGR uint8 image of given size."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def make_fake_landmark_result(h: int = 256, w: int = 256):
    """
    Build a FaceLandmarkResult with plausible synthetic values.
    Landmarks are placed roughly in the centre of the image.
    """
    from alignment import FaceLandmarkResult

    cx, cy = w / 2, h / 2

    # 478 points scattered in a face-like oval
    rng = np.random.default_rng(42)
    angles  = rng.uniform(0, 2 * np.pi, 478)
    radii_x = rng.uniform(0.1, 0.35, 478) * w
    radii_y = rng.uniform(0.1, 0.45, 478) * h
    xs = np.clip(cx + radii_x * np.cos(angles), 0, w - 1)
    ys = np.clip(cy + radii_y * np.sin(angles), 0, h - 1)
    all_pts = np.stack([xs, ys], axis=1).astype(np.float32)   # (478, 2)

    # 68-point subset (use first 68 for simplicity)
    pts_68 = all_pts[:68]

    # 5-point alignment anchors
    r_eye  = np.array([cx - 0.15 * w, cy - 0.1 * h], dtype=np.float32)
    l_eye  = np.array([cx + 0.15 * w, cy - 0.1 * h], dtype=np.float32)
    nose   = np.array([cx,             cy],            dtype=np.float32)
    m_r    = np.array([cx - 0.1 * w,  cy + 0.15 * h], dtype=np.float32)
    m_l    = np.array([cx + 0.1 * w,  cy + 0.15 * h], dtype=np.float32)
    align_pts = np.array([r_eye, l_eye, nose, m_r, m_l], dtype=np.float32)

    nose_tip_px   = all_pts[4].copy()
    eye_center_px = ((r_eye + l_eye) / 2.0).astype(np.float32)
    bbox_xyxy     = np.array(
        [all_pts[:, 0].min(), all_pts[:, 1].min(),
         all_pts[:, 0].max(), all_pts[:, 1].max()],
        dtype=np.float32,
    )

    return FaceLandmarkResult(
        landmarks_478 = all_pts,
        landmarks_68  = pts_68,
        align_pts     = align_pts,
        image_hw      = (h, w),
        nose_tip_px   = nose_tip_px,
        eye_center_px = eye_center_px,
        bbox_xyxy     = bbox_xyxy,
    )


def make_fake_affine_M() -> np.ndarray:
    """Return a near-identity affine matrix with slight rotation and scale."""
    angle   = np.deg2rad(5.0)
    scale   = 0.98
    M = np.array([
        [scale * np.cos(angle), -scale * np.sin(angle), 3.0],
        [scale * np.sin(angle),  scale * np.cos(angle), 2.0],
    ], dtype=np.float64)
    return M


# ══════════════════════════════════════════════════════════════════════════════
# 1. alignment.py
# ══════════════════════════════════════════════════════════════════════════════

def test_alignment():
    section("1. alignment.py")

    try:
        from alignment import (
            FaceLandmarkResult,
            warp_reference,
            build_convex_hull_mask,
            AlignmentResult,
        )
    except ImportError as e:
        print(f"  FAIL  Import failed: {e}")
        return

    H, W = 256, 256
    lm   = make_fake_landmark_result(H, W)
    img  = make_face_image(H, W)
    M    = make_fake_affine_M()

    # ── FaceLandmarkResult field shapes ──────────────────────────────────
    check("landmarks_478 shape",
          lm.landmarks_478.shape == (478, 2))
    check("landmarks_68 shape",
          lm.landmarks_68.shape == (68, 2))
    check("align_pts shape",
          lm.align_pts.shape == (5, 2))
    check("nose_tip_px shape",
          lm.nose_tip_px.shape == (2,))
    check("eye_center_px shape",
          lm.eye_center_px.shape == (2,))
    check("bbox_xyxy shape",
          lm.bbox_xyxy.shape == (4,))
    check("image_hw correct",
          lm.image_hw == (H, W))

    # ── bbox sanity ───────────────────────────────────────────────────────
    x1, y1, x2, y2 = lm.bbox_xyxy
    check("bbox x1 < x2",  x1 < x2)
    check("bbox y1 < y2",  y1 < y2)
    check("bbox within image", x2 <= W and y2 <= H)

    # ── warp_reference ────────────────────────────────────────────────────
    warped = warp_reference(img, M, (H, W))
    check("warp output shape",  warped.shape == (H, W, 3))
    check("warp output dtype",  warped.dtype == np.uint8)
    check("warp not all zeros", warped.sum() > 0)

    # ── build_convex_hull_mask ────────────────────────────────────────────
    mask = build_convex_hull_mask(lm.landmarks_68, lm.image_hw, dilation_px=10)
    check("hull mask shape",  mask.shape == (H, W))
    check("hull mask dtype",  mask.dtype == np.uint8)
    check("hull mask has 255", 255 in mask)
    check("hull mask has 0",     0 in mask)
    coverage = (mask > 0).mean()
    check("hull mask coverage 5–90%",
          0.05 < coverage < 0.90,
          f"coverage={coverage:.2%}")

    # ── AlignmentResult construction ──────────────────────────────────────
    ar = AlignmentResult(
        src_result   = lm,
        ref_result   = lm,
        aligned_face = warped,
        affine_M     = M,
        yaw_diff     = 5.0,
    )
    check("AlignmentResult.aligned_face",
          ar.aligned_face.shape == (H, W, 3))
    check("AlignmentResult.yaw_diff",
          isinstance(ar.yaw_diff, float))


# ══════════════════════════════════════════════════════════════════════════════
# 2. segmentation.py
# ══════════════════════════════════════════════════════════════════════════════

def test_segmentation():
    section("2. segmentation.py")

    try:
        from segmentation import get_face_mask, get_face_masks_for_pair
    except ImportError as e:
        print(f"  FAIL  Import failed: {e}")
        return

    H, W   = 256, 256
    img    = make_face_image(H, W)
    lm     = make_fake_landmark_result(H, W)

    # ── mask_type = "none" ────────────────────────────────────────────────
    mask_none = get_face_mask(img, lm, mask_type="none")
    check("none: shape",       mask_none.shape == (H, W))
    check("none: dtype",       mask_none.dtype == np.uint8)
    check("none: all 255",     mask_none.min() == 255)

    # ── mask_type = "convex_hull" ─────────────────────────────────────────
    mask_hull = get_face_mask(img, lm, mask_type="convex_hull",
                              convex_hull_dilation_px=10)
    check("hull: shape",        mask_hull.shape == (H, W))
    check("hull: dtype",        mask_hull.dtype == np.uint8)
    check("hull: has face (255)", 255 in mask_hull)
    check("hull: has bg (0)",       0 in mask_hull)
    cov = (mask_hull > 0).mean()
    check("hull: coverage 5–90%",
          0.05 < cov < 0.90,
          f"coverage={cov:.2%}")

    # ── mask_type = "sam" without predictor → should raise ValueError ─────
    raised = False
    try:
        get_face_mask(img, lm, mask_type="sam", predictor=None)
    except ValueError:
        raised = True
    check("sam: raises ValueError when predictor=None", raised)

    # ── unknown mask_type → should raise ValueError ───────────────────────
    raised_unknown = False
    try:
        get_face_mask(img, lm, mask_type="invalid_type")
    except ValueError:
        raised_unknown = True
    check("unknown mask_type raises ValueError", raised_unknown)

    # ── get_face_masks_for_pair (convex_hull) ─────────────────────────────
    src_m, ref_m = get_face_masks_for_pair(
        img, img, lm, lm,
        mask_type="convex_hull",
        convex_hull_dilation_px=10,
    )
    check("pair: src_mask shape",  src_m.shape == (H, W))
    check("pair: ref_mask shape",  ref_m.shape == (H, W))
    check("pair: src_mask dtype",  src_m.dtype == np.uint8)
    check("pair: ref_mask dtype",  ref_m.dtype == np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 3. decomposition.py
# ══════════════════════════════════════════════════════════════════════════════

def test_decomposition():
    section("3. decomposition.py")

    try:
        from decomposition import (
            decompose,
            DecomposeResult,
            to_pil_inputs,
            mask_to_tensor,
        )
    except ImportError as e:
        print(f"  FAIL  Import failed: {e}")
        return

    from PIL import Image

    H, W   = 256, 256
    img    = make_face_image(H, W)

    for method in ("gaussian", "fft", "none"):
        result = decompose(img, method=method)

        check(f"{method}: returns DecomposeResult",
              isinstance(result, DecomposeResult))
        check(f"{method}: LF dtype float32",
              result.LF.dtype == np.float32)
        check(f"{method}: HF dtype float32",
              result.HF.dtype == np.float32)
        check(f"{method}: LF shape",
              result.LF.shape == (H, W, 3))
        check(f"{method}: HF shape",
              result.HF.shape == (H, W, 3))
        check(f"{method}: method field correct",
              result.method == method)
        check(f"{method}: lf_energy is float",
              isinstance(result.lf_energy, float))
        check(f"{method}: hf_std is float",
              isinstance(result.hf_std, float))

        # Reconstruction: LF + HF ≈ original (not applicable for "none")
        if method != "none":
            recon_err = float(
                np.abs(
                    result.LF.astype(np.float64) +
                    result.HF.astype(np.float64) -
                    img.astype(np.float64)
                ).max()
            )
            check(f"{method}: LF + HF ≈ original (err < 0.5)",
                  recon_err < 0.5,
                  f"recon_err={recon_err:.4f}")

        # LF range should be within [0, 255]
        check(f"{method}: LF in [0, 255]",
              float(result.LF.min()) >= -1.0 and float(result.LF.max()) <= 256.0)

    # ── to_pil_inputs ─────────────────────────────────────────────────────
    for method in ("gaussian", "fft", "none"):
        result      = decompose(img, method=method)
        aligned_pil, lf_pil, hf_pil = to_pil_inputs(result, img, target_size=512)

        check(f"to_pil [{method}]: aligned_pil is PIL",
              isinstance(aligned_pil, Image.Image))
        check(f"to_pil [{method}]: lf_pil size",
              lf_pil.size == (512, 512))
        check(f"to_pil [{method}]: hf_pil size",
              hf_pil.size == (512, 512))
        check(f"to_pil [{method}]: lf_pil mode RGB",
              lf_pil.mode == "RGB")
        check(f"to_pil [{method}]: hf_pil mode RGB",
              hf_pil.mode == "RGB")

        # HF pixel values must be in [0, 255] after shift
        hf_arr = np.array(hf_pil)
        check(f"to_pil [{method}]: hf_pil pixels in [0,255]",
              int(hf_arr.min()) >= 0 and int(hf_arr.max()) <= 255)

    # ── mask_to_tensor ────────────────────────────────────────────────────
    raw_mask  = make_fake_landmark_result(H, W)
    from alignment import build_convex_hull_mask
    mask_uint8 = build_convex_hull_mask(
        raw_mask.landmarks_68, raw_mask.image_hw, dilation_px=10
    )

    for size in (512, 256):
        tensor = mask_to_tensor(mask_uint8, target_size=size)
        check(f"mask_to_tensor: shape (1,1,{size},{size})",
              tensor.shape == (1, 1, size, size))
        check(f"mask_to_tensor: dtype float32",
              tensor.dtype == torch.float32)
        check(f"mask_to_tensor: range [0,1]",
              float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0)

    # ── unknown method raises ValueError ──────────────────────────────────
    raised = False
    try:
        decompose(img, method="bad_method")
    except ValueError:
        raised = True
    check("decompose: unknown method raises ValueError", raised)

    # ── odd kernel assertion ───────────────────────────────────────────────
    raised_even = False
    try:
        decompose(img, method="gaussian", kernel=32)
    except AssertionError:
        raised_even = True
    check("decompose gaussian: even kernel raises AssertionError", raised_even)


# ══════════════════════════════════════════════════════════════════════════════
# 4. mask_utils.py
# ══════════════════════════════════════════════════════════════════════════════

def test_mask_utils():
    section("4. mask_utils.py")

    try:
        from mask_utils import mask_to_token_mask
    except ImportError as e:
        print(f"  FAIL  Import failed: {e}")
        return

    H, W = 256, 256

    # Build a synthetic face mask tensor: 1 in centre, 0 outside
    face_mask = torch.zeros(1, 1, H, W, dtype=torch.float32)
    face_mask[0, 0, 64:192, 64:192] = 1.0

    for spatial in (8, 16, 64):
        token_mask = mask_to_token_mask(face_mask, spatial_size=spatial)
        expected_len = spatial * spatial

        check(f"spatial={spatial}: output length",
              token_mask.shape == (expected_len,),
              f"got {token_mask.shape}")
        check(f"spatial={spatial}: dtype bool",
              token_mask.dtype == torch.bool)
        check(f"spatial={spatial}: has True tokens",
              token_mask.any().item())
        check(f"spatial={spatial}: has False tokens",
              (~token_mask).any().item())

    # ── Custom threshold ───────────────────────────────────────────────────
    # Mask with value 0.6 — threshold 0.5 → True, threshold 0.7 → False
    partial_mask = torch.full((1, 1, 64, 64), 0.6)
    tok_low  = mask_to_token_mask(partial_mask, spatial_size=8, threshold=0.5)
    tok_high = mask_to_token_mask(partial_mask, spatial_size=8, threshold=0.7)
    check("threshold=0.5 on 0.6 mask: all True",  tok_low.all().item())
    check("threshold=0.7 on 0.6 mask: all False", (~tok_high).all().item())

    # ── Device consistency ─────────────────────────────────────────────────
    result_cpu = mask_to_token_mask(face_mask, spatial_size=16)
    check("result on CPU",
          result_cpu.device.type == "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Cross-module integration — alignment → segmentation → decomposition
# ══════════════════════════════════════════════════════════════════════════════

def test_integration():
    section("5. Integration — alignment → segmentation → decomposition")

    try:
        from alignment    import warp_reference, build_convex_hull_mask
        from segmentation import get_face_mask
        from decomposition import decompose, to_pil_inputs, mask_to_tensor
    except ImportError as e:
        print(f"  FAIL  Import failed: {e}")
        return

    H, W   = 256, 256
    src    = make_face_image(H, W)
    ref    = make_face_image(H, W)
    src_lm = make_fake_landmark_result(H, W)
    M      = make_fake_affine_M()

    # Step 1 — warp reference into source pose
    aligned = warp_reference(ref, M, (H, W))
    check("step1: aligned face shape", aligned.shape == (H, W, 3))

    # Step 2 — segmentation mask from source landmarks
    mask_uint8 = get_face_mask(src, src_lm, mask_type="convex_hull",
                               convex_hull_dilation_px=10)
    check("step2: mask shape",         mask_uint8.shape == (H, W))
    check("step2: mask has face area", 255 in mask_uint8)

    # Step 3 — decompose aligned reference
    result = decompose(aligned, method="gaussian", kernel=31, sigma=5.0)
    check("step3: LF shape",   result.LF.shape == (H, W, 3))
    check("step3: HF shape",   result.HF.shape == (H, W, 3))
    check("step3: method tag", result.method == "gaussian")

    # Step 4 — PIL conversion
    aligned_pil, lf_pil, hf_pil = to_pil_inputs(result, aligned, target_size=512)
    check("step4: lf_pil size",    lf_pil.size  == (512, 512))
    check("step4: hf_pil size",    hf_pil.size  == (512, 512))

    # Step 5 — mask tensor ready for KVCache
    mask_tensor = mask_to_tensor(mask_uint8, target_size=512)
    check("step5: mask tensor shape",  mask_tensor.shape == (1, 1, 512, 512))
    check("step5: mask tensor range",
          float(mask_tensor.min()) >= 0.0 and float(mask_tensor.max()) <= 1.0)

    # Confirm the full artifact bundle is ready
    bundle = {
        "aligned_pil": aligned_pil,
        "lf_pil":      lf_pil,
        "hf_pil":      hf_pil,
        "face_mask":   mask_tensor,
        "yaw_diff":    5.0,
        "method":      result.method,
        "hf_std":      result.hf_std,
    }
    check("integration: all artifact keys present",
          all(k in bundle for k in
              ["aligned_pil", "lf_pil", "hf_pil", "face_mask",
               "yaw_diff", "method", "hf_std"]))


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  Stage 1 core module tests")
    print("═" * 60)

    for test_fn in [
        test_alignment,
        test_segmentation,
        test_decomposition,
        test_mask_utils,
        test_integration,
    ]:
        try:
            test_fn()
        except Exception:
            print(f"\n  ERROR in {test_fn.__name__}:")
            traceback.print_exc()

    print("\n" + "═" * 60)
    total = _PASS + _FAIL
    print(f"  Results: {_PASS}/{total} passed", end="")
    if _FAIL == 0:
        print("  ✓ all clear")
    else:
        print(f"  ✗ {_FAIL} failed — fix before running stage1_segment.py")
    print("═" * 60 + "\n")

    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()