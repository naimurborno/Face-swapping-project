# core/segmentation.py
"""
Face segmentation for the KV-injection face swap pipeline.

Single responsibility: produce a binary face mask for one image.

The mask_type ablation flag (from configs/default.yaml) selects the method:
    "sam"         — SAM with a MediaPipe-derived prompt (default, best quality)
    "convex_hull" — convex hull of 68 landmarks (fast, no SAM needed)
    "none"        — returns a full-image mask (global injection, no gating)

SAM prompt strategies (configs/default.yaml → stage1.sam.prompt_strategy):
    "nose_tip"   — single point at MediaPipe landmark index 4
                   Most reliable for frontal and near-frontal faces.
    "eye_center" — single point at the midpoint between both eye centroids
                   Slightly more central; useful when nose is partially occluded.
    "bbox"       — bounding box over all 478 landmarks
                   Use when point prompts produce incomplete masks on profile faces.

Supported SAM model types (paths.sam_model_type):
    "vit_h"   — ViT-H  (best quality, ~2.4 GB, needs ~6 GB VRAM)
    "vit_l"   — ViT-L  (good quality, ~1.2 GB)
    "vit_b"   — ViT-B  (faster,  ~375 MB)
    "mobile"  — MobileSAM (lightest, runs on CPU, ~40 MB)

Public API:
    load_sam_model()   — load SAM or MobileSAM predictor from checkpoint
    get_face_mask()    — main entry point: landmark result → binary mask (uint8)

Dependencies:
    SAM:       pip install git+https://github.com/facebookresearch/segment-anything.git
    MobileSAM: pip install git+https://github.com/ChaoningZhang/MobileSAM.git
    Both:      pip install opencv-python-headless numpy torch
"""

import numpy as np
import cv2
import torch
from typing import Optional

from core.alignment import FaceLandmarkResult, build_convex_hull_mask


# ── SAM loader ────────────────────────────────────────────────────────────────

def load_sam_model(
    checkpoint_path: str,
    model_type: str = "vit_h",
    device: Optional[str] = None,
):
    """
    Load a SAM or MobileSAM predictor and move it to the target device.

    This function is intentionally separated from get_face_mask() so you can
    load the model once and call get_face_mask() many times without reloading.
    Stage 1 loads it once, runs it on both source and reference, then deletes
    the predictor before Stage 2 loads the diffusion model.

    Args:
        checkpoint_path : Path to the .pth checkpoint file.
                          SAM:       sam_vit_h_4b8939.pth / sam_vit_l_0b3195.pth / sam_vit_b_01ec64.pth
                          MobileSAM: mobile_sam.pt
        model_type      : One of "vit_h" | "vit_l" | "vit_b" | "mobile".
                          Must match the checkpoint — mismatches raise RuntimeError.
        device          : "cuda" | "cpu" | None (auto-detects cuda if available).

    Returns:
        predictor : SamPredictor (SAM) or SamPredictor from MobileSAM,
                    already moved to device and ready for set_image() calls.

    Raises:
        ImportError  if the required SAM package is not installed.
        RuntimeError if the checkpoint does not match model_type.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "mobile":
        # MobileSAM has its own package with the same SamPredictor interface
        try:
            from mobile_sam import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "MobileSAM is not installed.\n"
                "  pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
            )
        sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    else:
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything is not installed.\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        if model_type not in sam_model_registry:
            raise RuntimeError(
                f"[segmentation] Unknown SAM model_type '{model_type}'.\n"
                f"  Valid options: {list(sam_model_registry.keys())}"
            )
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    print(
        f"[segmentation] Loaded SAM model_type='{model_type}' "
        f"on device='{device}' | checkpoint='{checkpoint_path}'"
    )
    return predictor


# ── Prompt builders ───────────────────────────────────────────────────────────

def _prompt_nose_tip(lm_result: FaceLandmarkResult):
    """
    Single point prompt at the nose tip (MediaPipe index 4).

    Most reliable prompt strategy for frontal and near-frontal faces.
    The nose tip is the most unambiguous interior face point — it sits
    centrally, is almost never occluded, and SAM confidently grows the
    face region from it.

    Returns:
        point_coords : (1, 2) float32   [[x, y]]
        point_labels : (1,)   int        [1]  (1 = foreground)
        box          : None
    """
    pt = lm_result.nose_tip_px.reshape(1, 2).astype(np.float32)
    return pt, np.array([1], dtype=np.int32), None


def _prompt_eye_center(lm_result: FaceLandmarkResult):
    """
    Single point prompt at the midpoint between both eye centroids.

    Useful when the nose is partially occluded (e.g. hand, scarf, mask).
    The eye midpoint is always an interior face point and rarely occluded.

    Returns:
        point_coords : (1, 2) float32
        point_labels : (1,)   int
        box          : None
    """
    pt = lm_result.eye_center_px.reshape(1, 2).astype(np.float32)
    return pt, np.array([1], dtype=np.int32), None


def _prompt_bbox(lm_result: FaceLandmarkResult):
    """
    Bounding box prompt over all 478 landmarks.

    More robust on profile faces where a single interior point may fall
    outside the visible face region. The bounding box gives SAM a tighter
    search region than a single point.

    The box is padded by 5% of its width/height on each side so that SAM
    does not clip the face boundary at the prompt edges.

    Returns:
        point_coords : None
        point_labels : None
        box          : (4,) float32  [x1, y1, x2, y2]  (SAM expects this shape)
    """
    x1, y1, x2, y2 = lm_result.bbox_xyxy.astype(np.float32)
    H, W = lm_result.image_hw

    pad_x = (x2 - x1) * 0.05
    pad_y = (y2 - y1) * 0.05

    x1 = float(np.clip(x1 - pad_x, 0, W - 1))
    y1 = float(np.clip(y1 - pad_y, 0, H - 1))
    x2 = float(np.clip(x2 + pad_x, 0, W - 1))
    y2 = float(np.clip(y2 + pad_y, 0, H - 1))

    box = np.array([x1, y1, x2, y2], dtype=np.float32)
    return None, None, box


_PROMPT_BUILDERS = {
    "nose_tip":   _prompt_nose_tip,
    "eye_center": _prompt_eye_center,
    "bbox":       _prompt_bbox,
}


# ── SAM mask selector ─────────────────────────────────────────────────────────

def _select_best_mask(
    masks:  np.ndarray,   # (N, H, W) bool
    scores: np.ndarray,   # (N,) float
    iou_threshold: float,
    stability_threshold: float,
    stab_scores: np.ndarray,  # (N,) float  — stability scores from SAM
) -> Optional[np.ndarray]:
    """
    From SAM's N candidate masks, pick the best one.

    Selection criteria (in order):
        1. Must pass both iou_threshold and stability_threshold.
        2. Among passing candidates, pick the one with the highest IoU score.

    For face prompts SAM usually returns 3 masks (whole face, face + hair,
    full head). Filtering by score selects the tightest semantic face region.

    Returns the selected mask as (H, W) uint8 [0/255], or None if all
    candidates fail the thresholds (caller should fall back to convex hull).
    """
    valid = []
    for i in range(len(masks)):
        if scores[i] >= iou_threshold and stab_scores[i] >= stability_threshold:
            valid.append((scores[i], i))

    if not valid:
        return None

    # Highest IoU score among valid candidates
    best_idx = max(valid, key=lambda x: x[0])[1]
    mask_bool = masks[best_idx]  # (H, W) bool

    return (mask_bool.astype(np.uint8) * 255)


# ── Post-processing ───────────────────────────────────────────────────────────

def _postprocess_mask(
    mask_uint8:  np.ndarray,   # (H, W) uint8 0/255
    dilation_px: int = 0,
) -> np.ndarray:
    """
    Clean up a raw SAM mask:
        1. Keep only the largest connected component (removes stray fragments).
        2. Fill holes inside the face region (cv2.RETR_EXTERNAL hull fill).
        3. Optionally dilate by dilation_px pixels to soften the boundary.

    Args:
        mask_uint8   : Raw SAM output mask.
        dilation_px  : Pixels to dilate. 0 = no dilation (SAM boundary is
                       already tight; dilation only needed for edge blending).

    Returns:
        Cleaned (H, W) uint8 mask, values 0/255.
    """
    # ── 1. Largest connected component ────────────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    if num_labels <= 1:
        # No components found — return as-is
        return mask_uint8

    # Component 0 is background; find largest foreground component
    areas = stats[1:, cv2.CC_STAT_AREA]          # skip background (index 0)
    largest_label = int(np.argmax(areas)) + 1    # +1 to offset background skip
    clean = np.where(labels == largest_label, np.uint8(255), np.uint8(0))

    # ── 2. Fill interior holes ─────────────────────────────────────────────
    # Flood fill from corner → inverted flood → OR with original
    flood = clean.copy()
    h, w  = flood.shape
    flood_canvas = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_canvas, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    clean     = cv2.bitwise_or(clean, flood_inv)

    # ── 3. Optional dilation ───────────────────────────────────────────────
    if dilation_px > 0:
        ksize  = dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        clean  = cv2.dilate(clean, kernel)

    return clean


# ── Main entry point ──────────────────────────────────────────────────────────

def get_face_mask(
    image_bgr:       np.ndarray,
    lm_result:       FaceLandmarkResult,
    mask_type:       str   = "sam",
    predictor               = None,      # SAM predictor — required when mask_type="sam"
    prompt_strategy: str   = "nose_tip", # "nose_tip" | "bbox" | "eye_center"
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    dilation_px:     int   = 0,          # post-process dilation (0 = off for SAM)
    convex_hull_dilation_px: int = 10,   # dilation specifically for convex_hull fallback
) -> np.ndarray:
    """
    Produce a binary face mask for one image.

    This is the single function called by stage1_segment.py. The mask_type
    argument maps directly to the ablation.mask_type flag in default.yaml so
    changing the ablation requires zero code changes here.

    Args:
        image_bgr       : BGR uint8 image to segment.
        lm_result       : FaceLandmarkResult from alignment.FaceLandmarkDetector.
                          Must match image_bgr (same image, same resolution).
        mask_type       : "sam" | "convex_hull" | "none"
        predictor       : SamPredictor loaded by load_sam_model(). Required when
                          mask_type="sam". Ignored otherwise.
        prompt_strategy : "nose_tip" | "bbox" | "eye_center"
                          Used only when mask_type="sam".
        pred_iou_thresh : SAM IoU threshold. Masks below this score are rejected.
                          Lower if SAM produces no valid mask on unusual poses.
        stability_score_thresh : SAM stability threshold. Lower if needed.
        dilation_px     : Post-process dilation applied to SAM mask.
                          0 by default — SAM boundary is already precise.
                          Increase if blending shows visible hard edges.
        convex_hull_dilation_px : Dilation for the convex hull fallback.
                          Should be larger than SAM dilation (hull is rougher).

    Returns:
        (H, W) uint8 mask, values 0 or 255.
            255 = face region (injection happens here)
              0 = background (injection is suppressed here)

    Behaviour on SAM failure:
        If SAM returns no mask passing the score thresholds, the function
        automatically falls back to convex_hull and prints a warning.
        This ensures the pipeline never crashes silently due to an unusual face.
    """
    H, W = image_bgr.shape[:2]

    # ── "none": full-image mask — global injection, no spatial gating ─────
    if mask_type == "none":
        print("[segmentation] mask_type='none' — returning full-image mask.")
        return np.full((H, W), 255, dtype=np.uint8)

    # ── "convex_hull": fast polygon fallback ──────────────────────────────
    if mask_type == "convex_hull":
        print("[segmentation] mask_type='convex_hull' — using MediaPipe hull.")
        return build_convex_hull_mask(
            lm_result.landmarks_68,
            lm_result.image_hw,
            dilation_px=convex_hull_dilation_px,
        )

    # ── "sam": SAM segmentation (main path) ───────────────────────────────
    if mask_type != "sam":
        raise ValueError(
            f"[segmentation] Unknown mask_type '{mask_type}'. "
            f"Choose: 'sam' | 'convex_hull' | 'none'"
        )

    if predictor is None:
        raise ValueError(
            "[segmentation] mask_type='sam' requires a loaded SAM predictor.\n"
            "  Call load_sam_model() first and pass the result as predictor=."
        )

    if prompt_strategy not in _PROMPT_BUILDERS:
        raise ValueError(
            f"[segmentation] Unknown prompt_strategy '{prompt_strategy}'.\n"
            f"  Choose: {list(_PROMPT_BUILDERS.keys())}"
        )

    # ── Set image into SAM (runs image encoder — the expensive step) ──────
    # SAM expects RGB uint8
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # ── Build prompt from MediaPipe landmarks ─────────────────────────────
    point_coords, point_labels, box = _PROMPT_BUILDERS[prompt_strategy](lm_result)

    print(
        f"[segmentation] SAM predict | strategy='{prompt_strategy}' | "
        f"{'point=' + str(point_coords.tolist()) if point_coords is not None else ''}"
        f"{'box=' + str(box.tolist()) if box is not None else ''}"
    )

    # ── SAM predict ───────────────────────────────────────────────────────
    predict_kwargs = dict(multimask_output=True)

    if point_coords is not None:
        predict_kwargs["point_coords"] = point_coords
        predict_kwargs["point_labels"] = point_labels

    if box is not None:
        # SAM expects box as (1, 4) tensor when passed alongside points,
        # but as (4,) numpy array when passed alone. We always pass as numpy.
        predict_kwargs["box"] = box

    masks, scores, logits = predictor.predict(**predict_kwargs)
    # masks : (N, H, W) bool
    # scores: (N,)      float  — predicted IoU for each mask

    # SAM does not directly expose stability scores from predict().
    # Approximate stability as 1.0 for all masks (only IoU threshold is used).
    # For automatic mask generation (SamAutomaticMaskGenerator) stability
    # scores are available, but single-prompt predict() does not return them.
    stab_scores = np.ones(len(masks), dtype=np.float32)

    # ── Select best mask ──────────────────────────────────────────────────
    mask_uint8 = _select_best_mask(
        masks, scores,
        iou_threshold=pred_iou_thresh,
        stability_threshold=stability_score_thresh,
        stab_scores=stab_scores,
    )

    if mask_uint8 is None:
        # SAM returned no mask passing thresholds — fall back to convex hull
        print(
            f"[segmentation] WARNING: SAM returned no mask above thresholds "
            f"(iou={pred_iou_thresh}, stab={stability_score_thresh}).\n"
            f"  Max SAM score: {scores.max():.3f}\n"
            f"  Falling back to convex hull mask.\n"
            f"  To avoid fallback: lower pred_iou_thresh in default.yaml, "
            f"or switch prompt_strategy from '{prompt_strategy}' to 'bbox'."
        )
        return build_convex_hull_mask(
            lm_result.landmarks_68,
            lm_result.image_hw,
            dilation_px=convex_hull_dilation_px,
        )

    best_score_idx = int(np.argmax(scores))
    print(
        f"[segmentation] SAM selected mask {best_score_idx}/{len(masks)} | "
        f"iou={scores[best_score_idx]:.3f} | "
        f"coverage={mask_uint8.astype(bool).mean()*100:.1f}%"
    )

    # ── Post-process: clean components + optional dilation ────────────────
    mask_clean = _postprocess_mask(mask_uint8, dilation_px=dilation_px)

    return mask_clean


# ── Batch convenience helper ──────────────────────────────────────────────────

def get_face_masks_for_pair(
    source_bgr:      np.ndarray,
    reference_bgr:   np.ndarray,
    src_lm:          FaceLandmarkResult,
    ref_lm:          FaceLandmarkResult,
    mask_type:       str   = "sam",
    predictor               = None,
    prompt_strategy: str   = "nose_tip",
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    dilation_px:     int   = 0,
    convex_hull_dilation_px: int = 10,
):
    """
    Run get_face_mask() on both source and reference images in one call.

    SAM's set_image() is called twice internally (once per image). The image
    encoder runs separately for each — this is correct behaviour because source
    and reference are different images.

    This is the function stage1_segment.py calls. It is separated from
    get_face_mask() purely for convenience; no new logic is introduced.

    Returns:
        src_mask : (H_src, W_src) uint8 mask for source image
        ref_mask : (H_ref, W_ref) uint8 mask for reference image
    """
    print("\n[segmentation] ── Source mask ──")
    src_mask = get_face_mask(
        source_bgr, src_lm,
        mask_type=mask_type,
        predictor=predictor,
        prompt_strategy=prompt_strategy,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        dilation_px=dilation_px,
        convex_hull_dilation_px=convex_hull_dilation_px,
    )

    print("\n[segmentation] ── Reference mask ──")
    ref_mask = get_face_mask(
        reference_bgr, ref_lm,
        mask_type=mask_type,
        predictor=predictor,
        prompt_strategy=prompt_strategy,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        dilation_px=dilation_px,
        convex_hull_dilation_px=convex_hull_dilation_px,
    )

    return src_mask, ref_mask