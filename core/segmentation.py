# core/segmentation.py
"""
Object segmentation for the Mixed-Frequency Prior Guided Inpainting pipeline.

Single responsibility: produce a binary object mask for a content image.

The mask defines where donor attributes are applied and where the content
image is frozen forever. Outside the mask, blended latent anchoring guarantees
pixel-identical preservation of the content image. Inside the mask, the
mixed-frequency prior guides what diffusion generates.

────────────────────────────────────────────────────────────────────────────
PROMPT MODES  (configs/default.yaml → stage1.sam.prompt_strategy)
────────────────────────────────────────────────────────────────────────────

  "text"          — TEXT-DRIVEN (new default, recommended)
                    You describe what to segment in plain language:
                        "the brick wall", "the sofa cushion", "the face"
                    Grounding DINO converts your text → bounding box(es).
                    SAM refines those boxes → tight pixel-level mask.
                    Requires: groundingdino + SAM (see Installation below).
                    Config key: stage1.sam.text_prompt  (required)

  "center_point"  — Automatic: single point at the image center.
                    Works for centered subjects. No text needed.
                    Fallback when text prompt is empty or DINO unavailable.

  "bbox"          — Automatic: bounding box with 5% inset.
                    More robust for off-center or large subjects.

  "none" (mask_type) — Full-image mask. No SAM, no DINO.
                    Entire image is editable. Fastest, zero dependencies.
                    Pass --mask-type none or set ablation.mask_type: none.

────────────────────────────────────────────────────────────────────────────
INSTALLATION
────────────────────────────────────────────────────────────────────────────

  For text-prompted segmentation (Grounded-SAM):

    # 1. Grounding DINO
    pip install groundingdino-py
    # Download weights:
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

    # 2. SAM (if not already installed)
    pip install git+https://github.com/facebookresearch/segment-anything.git
    # Download checkpoint: sam_vit_h_4b8939.pth

  For geometric-only prompts (center_point / bbox):
    pip install git+https://github.com/facebookresearch/segment-anything.git

  For MobileSAM (lighter alternative to SAM):
    pip install git+https://github.com/ChaoningZhang/MobileSAM.git

────────────────────────────────────────────────────────────────────────────
CONFIG EXAMPLE  (configs/default.yaml)
────────────────────────────────────────────────────────────────────────────

  ablation:
    mask_type: "sam"           # "sam" | "none"

  stage1:
    sam:
      prompt_strategy:        "text"        # "text" | "center_point" | "bbox"
      text_prompt:            "the brick wall"   # YOUR TEXT — describe the region

      # Grounding DINO paths (only needed for prompt_strategy: "text")
      grounding_dino_config:  "GroundingDINO_SwinT_OGC.py"
      grounding_dino_weights: "groundingdino_swint_ogc.pth"
      box_threshold:          0.35          # DINO box confidence cutoff
      text_threshold:         0.25          # DINO text confidence cutoff

      # SAM quality thresholds
      pred_iou_thresh:        0.88
      stability_score_thresh: 0.95
      dilation_px:            0

      # SAM model (used for all prompt strategies)
      # paths.sam_checkpoint and paths.sam_model_type in default.yaml

────────────────────────────────────────────────────────────────────────────
PUBLIC API
────────────────────────────────────────────────────────────────────────────
    load_sam_model()          — load SAM / MobileSAM predictor
    load_grounding_dino()     — load Grounding DINO model
    get_object_mask()         — main entry point: image → binary mask (uint8)
    mask_to_tensor()          — convert mask to (1,1,H,W) float32 tensor
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ══════════════════════════════════════════════════════════════════════════════
# SAM LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_sam_model(
    checkpoint_path: str,
    model_type: str = "vit_h",
    device: Optional[str] = None,
):
    """
    Load a SAM or MobileSAM predictor and move it to the target device.

    Args:
        checkpoint_path : Path to the .pth checkpoint file.
        model_type      : "vit_h" | "vit_l" | "vit_b" | "mobile"
        device          : "cuda" | "cpu" | None (auto-detects)

    Returns:
        SamPredictor ready for set_image() calls.

    Raises:
        ImportError  — if the required SAM package is not installed.
        RuntimeError — if the checkpoint does not match model_type.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "mobile":
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
        f"[segmentation] SAM loaded | model_type='{model_type}' "
        f"device='{device}' | checkpoint='{checkpoint_path}'"
    )
    return predictor


# ══════════════════════════════════════════════════════════════════════════════
# GROUNDING DINO LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_grounding_dino(
    config_path:  str,
    weights_path: str,
    device: Optional[str] = None,
):
    """
    Load a Grounding DINO model for text-to-box detection.

    Grounding DINO converts a text prompt ("the brick wall", "the face")
    into one or more bounding boxes in image space. Those boxes are then
    passed to SAM as spatial prompts, giving you text-driven segmentation
    without any fine-tuning.

    Args:
        config_path  : Path to the GroundingDINO config .py file.
                       e.g. "GroundingDINO_SwinT_OGC.py"
        weights_path : Path to the .pth checkpoint.
                       e.g. "groundingdino_swint_ogc.pth"
        device       : "cuda" | "cpu" | None (auto-detects)

    Returns:
        Grounding DINO model in eval mode, on device.

    Raises:
        ImportError if groundingdino is not installed.
        FileNotFoundError if config or weights path is missing.
    """
    import os
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[segmentation] Grounding DINO config not found: {config_path}\n"
            f"  Download from:\n"
            f"  https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/"
            f"main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"[segmentation] Grounding DINO weights not found: {weights_path}\n"
            f"  Download from:\n"
            f"  https://github.com/IDEA-Research/GroundingDINO/releases/"
            f"download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        )

    try:
        from groundingdino.util.inference import load_model
    except ImportError:
        raise ImportError(
            "groundingdino is not installed.\n"
            "  pip install groundingdino-py\n"
            "  or: pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(config_path, weights_path, device=device)
    model.eval()
    print(
        f"[segmentation] Grounding DINO loaded | device='{device}'\n"
        f"               config='{config_path}'\n"
        f"               weights='{weights_path}'"
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# TEXT → BOX  (Grounding DINO)
# ══════════════════════════════════════════════════════════════════════════════

def _text_to_boxes(
    dino_model,
    image_rgb:      np.ndarray,   # (H, W, 3) uint8
    text_prompt:    str,          # e.g. "the brick wall"
    box_threshold:  float = 0.35,
    text_threshold: float = 0.25,
) -> List[np.ndarray]:
    """
    Run Grounding DINO to detect bounding boxes from a text prompt.

    The text prompt is passed as-is to DINO. You can use natural language:
        "the brick wall"
        "face"
        "the sofa cushion fabric"
        "wooden floor"

    Multiple phrases can be joined with a period:
        "the wall . the floor"
    DINO will detect each phrase separately and return all boxes.

    Args:
        dino_model     : Loaded Grounding DINO model.
        image_rgb      : (H, W, 3) uint8 RGB image.
        text_prompt    : Natural language description of the region to segment.
        box_threshold  : Minimum box confidence score (default 0.35).
                         Lower to 0.20 if DINO misses a valid region.
                         Raise to 0.50 if DINO returns noisy boxes.
        text_threshold : Minimum text-box alignment score (default 0.25).

    Returns:
        List of (4,) float32 arrays in [x1, y1, x2, y2] pixel coordinates.
        Empty list if no boxes pass thresholds.
    """
    from groundingdino.util.inference import predict
    from groundingdino.util import box_ops
    import torchvision.transforms as T

    H, W = image_rgb.shape[:2]

    # DINO expects a specific normalised transform
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image_rgb)

    # Ensure prompt ends with a period (DINO tokenizer expects sentence format)
    prompt = text_prompt.strip()
    if not prompt.endswith("."):
        prompt = prompt + "."

    print(f"[segmentation] Grounding DINO | prompt='{prompt}' | "
          f"box_thresh={box_threshold} text_thresh={text_threshold}")

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model      = dino_model,
            image      = image_tensor,
            caption    = prompt,
            box_threshold  = box_threshold,
            text_threshold = text_threshold,
        )

    if boxes is None or len(boxes) == 0:
        print(f"[segmentation] Grounding DINO returned no boxes above thresholds.")
        return []

    # boxes are cx,cy,w,h normalised [0,1] — convert to pixel x1y1x2y2
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)   # (N, 4) normalised
    boxes_px   = boxes_xyxy * torch.tensor([W, H, W, H], dtype=torch.float32)
    boxes_px   = boxes_px.cpu().numpy()               # (N, 4) pixel coords

    print(
        f"[segmentation] DINO detected {len(boxes_px)} box(es) | "
        f"phrases={phrases} | scores={[f'{s:.2f}' for s in logits.tolist()]}"
    )

    return [boxes_px[i].astype(np.float32) for i in range(len(boxes_px))]


def _merge_boxes(boxes: List[np.ndarray]) -> np.ndarray:
    """
    Merge multiple DINO boxes into a single enclosing bounding box.

    When DINO detects multiple boxes for the same prompt (e.g. a wall with
    two disconnected sections), merging gives SAM a single clean box that
    covers the entire intended region.

    Args:
        boxes : List of (4,) [x1, y1, x2, y2] arrays.

    Returns:
        (4,) float32 array — merged [x1, y1, x2, y2].
    """
    arr = np.stack(boxes, axis=0)   # (N, 4)
    return np.array([
        arr[:, 0].min(),  # x1 = leftmost
        arr[:, 1].min(),  # y1 = topmost
        arr[:, 2].max(),  # x2 = rightmost
        arr[:, 3].max(),  # y2 = bottommost
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC PROMPT BUILDERS  (center_point / bbox — unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def _prompt_center_point(image_hw: Tuple[int, int]):
    """
    Single point prompt at the image center.

    Works well for centered subjects. SAM reliably grows the foreground
    region from a central interior point. Switch to 'bbox' or 'text' when
    the subject is not centered or when center_point selects the wrong region.

    Returns:
        point_coords : (1, 2) float32  [[cx, cy]]
        point_labels : (1,)   int32    [1]  (1 = foreground)
        box          : None
    """
    H, W = image_hw
    point_coords = np.array([[W / 2.0, H / 2.0]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    return point_coords, point_labels, None


def _prompt_bbox(image_hw: Tuple[int, int]):
    """
    Bounding box prompt over the full image with a 5% inset on each side.

    More robust than center_point for off-center or large subjects. The
    inset prevents the box from coinciding with the image edge, which can
    cause SAM to select the background rather than the subject.

    Returns:
        point_coords : None
        point_labels : None
        box          : (4,) float32  [x1, y1, x2, y2]
    """
    H, W = image_hw
    pad_x = W * 0.05
    pad_y = H * 0.05
    box = np.array([
        float(np.clip(pad_x,     0, W - 1)),
        float(np.clip(pad_y,     0, H - 1)),
        float(np.clip(W - pad_x, 0, W - 1)),
        float(np.clip(H - pad_y, 0, H - 1)),
    ], dtype=np.float32)
    return None, None, box


# ══════════════════════════════════════════════════════════════════════════════
# SAM MASK SELECTION + POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _select_best_mask(
    masks:               np.ndarray,   # (N, H, W) bool
    scores:              np.ndarray,   # (N,) float — predicted IoU
    iou_threshold:       float,
    stability_threshold: float,
    stab_scores:         np.ndarray,   # (N,) float — stability score
) -> Optional[np.ndarray]:
    """
    From SAM's N candidate masks, select the one with the highest IoU score
    that passes both iou_threshold and stability_threshold.

    Returns:
        (H, W) uint8 mask [0/255], or None if no candidate passes.
    """
    valid = [
        (scores[i], i)
        for i in range(len(masks))
        if scores[i] >= iou_threshold and stab_scores[i] >= stability_threshold
    ]
    if not valid:
        return None
    best_idx  = max(valid, key=lambda x: x[0])[1]
    return (masks[best_idx].astype(np.uint8) * 255)


def _postprocess_mask(
    mask_uint8:  np.ndarray,   # (H, W) uint8 0/255
    dilation_px: int = 0,
) -> np.ndarray:
    """
    Clean up a raw SAM mask:
        1. Keep only the largest connected component.
        2. Fill interior holes via flood-fill inversion.
        3. Optionally dilate by dilation_px pixels.

    Args:
        mask_uint8   : Raw SAM output, values 0/255.
        dilation_px  : Pixels to dilate outward. 0 = no dilation.
                       Increase to 8-16 if mask edges clip the subject.

    Returns:
        Cleaned (H, W) uint8 mask, values 0/255.
    """
    # 1. Largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    if num_labels <= 1:
        return mask_uint8

    areas         = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    clean         = np.where(labels == largest_label, np.uint8(255), np.uint8(0))

    # 2. Fill interior holes
    flood        = clean.copy()
    h, w         = flood.shape
    flood_canvas = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_canvas, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    clean     = cv2.bitwise_or(clean, flood_inv)

    # 3. Optional dilation
    if dilation_px > 0:
        ksize  = dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        clean  = cv2.dilate(clean, kernel)

    return clean


# ══════════════════════════════════════════════════════════════════════════════
# SAM PREDICT HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _run_sam_with_box(
    predictor:           object,         # SamPredictor with set_image() called
    box:                 np.ndarray,     # (4,) float32 [x1, y1, x2, y2]
    pred_iou_thresh:     float,
    stability_thresh:    float,
) -> Optional[np.ndarray]:
    """
    Run SAM predict with a bounding box prompt.

    Used by both the text-driven path (DINO box → SAM) and the geometric
    bbox path. Box input is (4,) [x1, y1, x2, y2] in pixel coordinates.

    Returns:
        (H, W) uint8 mask [0/255] or None.
    """
    print(f"[segmentation] SAM predict | box=[{box[0]:.0f},{box[1]:.0f},"
          f"{box[2]:.0f},{box[3]:.0f}]")

    masks, scores, _ = predictor.predict(
        box              = box,
        multimask_output = True,
    )
    stab_scores = np.ones(len(masks), dtype=np.float32)
    return _select_best_mask(
        masks, scores,
        iou_threshold       = pred_iou_thresh,
        stability_threshold = stability_thresh,
        stab_scores         = stab_scores,
    )


def _run_sam_with_point(
    predictor:           object,
    image_hw:            Tuple[int, int],
    prompt_strategy:     str,            # "center_point" | "bbox"
    pred_iou_thresh:     float,
    stability_thresh:    float,
) -> Optional[np.ndarray]:
    """
    Run SAM predict with a geometric prompt (center_point or bbox).

    Returns:
        (H, W) uint8 mask [0/255] or None.
    """
    builders = {"center_point": _prompt_center_point, "bbox": _prompt_bbox}
    point_coords, point_labels, box = builders[prompt_strategy](image_hw)

    desc = (f"point={point_coords.tolist()}" if point_coords is not None
            else f"box={box.tolist()}")
    print(f"[segmentation] SAM predict | strategy='{prompt_strategy}' | {desc}")

    predict_kwargs = dict(multimask_output=True)
    if point_coords is not None:
        predict_kwargs["point_coords"] = point_coords
        predict_kwargs["point_labels"] = point_labels
    if box is not None:
        predict_kwargs["box"] = box

    masks, scores, _ = predictor.predict(**predict_kwargs)
    stab_scores = np.ones(len(masks), dtype=np.float32)
    return _select_best_mask(
        masks, scores,
        iou_threshold       = pred_iou_thresh,
        stability_threshold = stability_thresh,
        stab_scores         = stab_scores,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def get_object_mask(
    image_bgr:              np.ndarray,
    mask_type:              str   = "sam",
    predictor                     = None,
    prompt_strategy:        str   = "text",
    text_prompt:            str   = "",
    dino_model                    = None,
    box_threshold:          float = 0.35,
    text_threshold:         float = 0.25,
    pred_iou_thresh:        float = 0.88,
    stability_score_thresh: float = 0.95,
    dilation_px:            int   = 0,
) -> np.ndarray:
    """
    Produce a binary object mask for a content image.

    This is the single function called by stage1_segment.py. Everything
    about how the mask is generated is controlled by the arguments here,
    which map 1:1 to configs/default.yaml fields.

    ── PROMPT STRATEGIES ────────────────────────────────────────────────────

    prompt_strategy="text"  (recommended — new default)

        You provide a plain-language description of the region to segment:
            "the brick wall"
            "the face"
            "the sofa fabric"
            "wooden floor planks"

        Grounding DINO detects a bounding box for your text. SAM refines
        that box into a tight pixel-level mask. If DINO finds multiple
        boxes (e.g. "the wall" returns two disconnected sections), they
        are merged into a single enclosing box before SAM runs.

        If DINO finds nothing above box_threshold:
            → automatically retries with prompt_strategy="center_point"
            → if that also fails, falls back to full-image mask

        Requires: dino_model to be loaded (load_grounding_dino()).
        If dino_model is None and prompt_strategy="text", the function
        falls back to "center_point" with a clear warning.

    prompt_strategy="center_point"

        Geometric fallback. SAM receives a single point at the image
        center. Works for centered subjects. No text or DINO needed.

    prompt_strategy="bbox"

        Geometric fallback. SAM receives a bounding box covering 90%
        of the image. More robust for off-center subjects.

    mask_type="none"

        No SAM, no DINO. Returns a full-image mask — the entire image
        is editable. Fastest option, zero dependencies.

    ── ARGS ─────────────────────────────────────────────────────────────────

    Args:
        image_bgr      : (H, W, 3) uint8 BGR content image.
        mask_type      : "sam" | "none"
        predictor      : SamPredictor from load_sam_model(). Required for
                         mask_type="sam".
        prompt_strategy: "text" | "center_point" | "bbox"
        text_prompt    : Natural language description of what to segment.
                         Required when prompt_strategy="text".
                         Examples: "the wall", "face", "sofa cushion"
        dino_model     : Grounding DINO model from load_grounding_dino().
                         Required when prompt_strategy="text".
        box_threshold  : DINO minimum box confidence (default 0.35).
                         Lower to ~0.20 if your region is not detected.
                         Raise to ~0.50 to reduce spurious boxes.
        text_threshold : DINO minimum text-alignment score (default 0.25).
        pred_iou_thresh: SAM IoU threshold (default 0.88).
                         Lower to 0.70 if SAM rejects valid masks.
        stability_score_thresh : SAM stability threshold (default 0.95).
        dilation_px    : Post-process dilation in pixels (default 0).
                         Increase to 8-16 if mask edges clip the subject.

    ── RETURNS ──────────────────────────────────────────────────────────────

    Returns:
        (H, W) uint8 mask, values 0 or 255.
            255 = editable  — donor attributes applied, diffusion fills this
              0 = frozen    — content image pixel-identical (blended anchoring)

    ── FALLBACK CHAIN ───────────────────────────────────────────────────────

    The fallback chain ensures the pipeline never crashes silently:
        text → DINO fails → retry center_point → retry bbox → full-image mask
        center_point → fails → retry bbox → full-image mask
        All fallbacks print a clear warning.
    """
    H, W = image_bgr.shape[:2]

    # ── mask_type="none": full-image mask, no SAM ─────────────────────────
    if mask_type == "none":
        print("[segmentation] mask_type='none' — returning full-image mask.")
        return np.full((H, W), 255, dtype=np.uint8)

    if mask_type != "sam":
        raise ValueError(
            f"[segmentation] Unknown mask_type '{mask_type}'. "
            f"Choose: 'sam' | 'none'"
        )

    if predictor is None:
        raise ValueError(
            "[segmentation] mask_type='sam' requires a loaded SAM predictor.\n"
            "  Call load_sam_model() first."
        )

    if prompt_strategy not in ("text", "center_point", "bbox"):
        raise ValueError(
            f"[segmentation] Unknown prompt_strategy '{prompt_strategy}'.\n"
            f"  Choose: 'text' | 'center_point' | 'bbox'"
        )

    # SAM expects RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    image_hw  = (H, W)

    mask_uint8 = None

    # ── TEXT PATH: Grounding DINO → box → SAM ────────────────────────────
    if prompt_strategy == "text":

        if not text_prompt.strip():
            print(
                "[segmentation] WARNING: prompt_strategy='text' but text_prompt "
                "is empty.\n"
                "  Falling back to prompt_strategy='center_point'.\n"
                "  Set stage1.sam.text_prompt in default.yaml or pass "
                "--text-prompt 'describe your region'."
            )
            prompt_strategy = "center_point"

        elif dino_model is None:
            print(
                "[segmentation] WARNING: prompt_strategy='text' but dino_model "
                "is None.\n"
                "  Load Grounding DINO with load_grounding_dino() and pass it "
                "as dino_model=.\n"
                "  Falling back to prompt_strategy='center_point'."
            )
            prompt_strategy = "center_point"

        else:
            # ── Grounding DINO: text → boxes ──────────────────────────────
            boxes = _text_to_boxes(
                dino_model  = dino_model,
                image_rgb   = image_rgb,
                text_prompt = text_prompt,
                box_threshold  = box_threshold,
                text_threshold = text_threshold,
            )

            if boxes:
                # Merge all detected boxes into one enclosing box, then SAM
                merged_box = _merge_boxes(boxes) if len(boxes) > 1 else boxes[0]
                print(
                    f"[segmentation] Using {'merged' if len(boxes) > 1 else 'single'} "
                    f"DINO box → SAM | "
                    f"box=[{merged_box[0]:.0f},{merged_box[1]:.0f},"
                    f"{merged_box[2]:.0f},{merged_box[3]:.0f}]"
                )
                mask_uint8 = _run_sam_with_box(
                    predictor        = predictor,
                    box              = merged_box,
                    pred_iou_thresh  = pred_iou_thresh,
                    stability_thresh = stability_score_thresh,
                )

            # DINO returned nothing → fall back to center_point
            if mask_uint8 is None:
                print(
                    f"[segmentation] WARNING: Grounding DINO found no boxes for "
                    f"prompt '{text_prompt}'.\n"
                    f"  Suggestions:\n"
                    f"  1. Lower box_threshold (current: {box_threshold}) to 0.20\n"
                    f"  2. Simplify the prompt (e.g. 'wall' instead of 'the brick wall')\n"
                    f"  3. Check that the object is clearly visible in the image\n"
                    f"  Falling back to prompt_strategy='center_point'."
                )
                prompt_strategy = "center_point"

    # ── GEOMETRIC PATHS: center_point / bbox (also fallback from text) ────
    if mask_uint8 is None and prompt_strategy in ("center_point", "bbox"):
        mask_uint8 = _run_sam_with_point(
            predictor        = predictor,
            image_hw         = image_hw,
            prompt_strategy  = prompt_strategy,
            pred_iou_thresh  = pred_iou_thresh,
            stability_thresh = stability_score_thresh,
        )

        # center_point failed → retry with bbox
        if mask_uint8 is None and prompt_strategy == "center_point":
            print(
                f"[segmentation] WARNING: SAM 'center_point' returned no mask "
                f"above thresholds.\n"
                f"  Retrying with prompt_strategy='bbox'."
            )
            mask_uint8 = _run_sam_with_point(
                predictor        = predictor,
                image_hw         = image_hw,
                prompt_strategy  = "bbox",
                pred_iou_thresh  = pred_iou_thresh,
                stability_thresh = stability_score_thresh,
            )

    # ── Final fallback: full-image mask ───────────────────────────────────
    if mask_uint8 is None:
        print(
            "[segmentation] WARNING: All SAM strategies failed.\n"
            "  Falling back to full-image mask (mask_type='none' behaviour).\n"
            "  Consider: lower pred_iou_thresh, lower box_threshold, "
            "or simplify text_prompt."
        )
        return np.full((H, W), 255, dtype=np.uint8)

    # ── Post-process: clean components + optional dilation ────────────────
    mask_clean = _postprocess_mask(mask_uint8, dilation_px=dilation_px)

    coverage = mask_clean.astype(bool).mean() * 100
    print(
        f"[segmentation] Mask ready | "
        f"coverage={coverage:.1f}% | dilation={dilation_px}px | "
        f"{'✓' if 3.0 < coverage < 97.0 else '⚠ unusual coverage — check mask'}"
    )

    return mask_clean


# ══════════════════════════════════════════════════════════════════════════════
# MASK TENSOR CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def mask_to_tensor(
    mask_uint8: np.ndarray,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Convert a uint8 numpy mask to a (1, 1, H, W) float32 tensor in [0, 1].

    This is the format expected by KVCache.face_mask and by the blended
    latent anchoring step in stage2_diffusion.py.

    Args:
        mask_uint8 : (H, W) uint8 mask produced by get_object_mask().
        device     : Target device. None = CPU.

    Returns:
        (1, 1, H, W) float32 tensor, values 0.0 (frozen) or 1.0 (editable).
    """
    t = torch.from_numpy(mask_uint8.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── segmentation.py smoke tests ──")

    H, W = 512, 512

    # T1: center_point prompt shapes
    coords, labels, box = _prompt_center_point((H, W))
    assert coords.shape == (1, 2)
    assert labels.shape == (1,)
    assert box is None
    assert coords[0, 0] == W / 2.0
    assert coords[0, 1] == H / 2.0
    print("T1 PASS — center_point shapes")

    # T2: bbox prompt shapes and inset
    coords2, labels2, box2 = _prompt_bbox((H, W))
    assert coords2 is None
    assert labels2 is None
    assert box2.shape == (4,)
    assert box2[0] > 0
    assert box2[2] < W
    print("T2 PASS — bbox prompt shapes and inset")

    # T3: mask_type="none" returns full-image mask
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    m = get_object_mask(dummy, mask_type="none")
    assert m.shape == (H, W)
    assert m.dtype == np.uint8
    assert (m == 255).all()
    print("T3 PASS — mask_type='none' returns full 255 mask")

    # T4: unknown mask_type raises ValueError
    try:
        get_object_mask(dummy, mask_type="convex_hull")
        assert False
    except ValueError as e:
        assert "convex_hull" in str(e)
    print("T4 PASS — unknown mask_type raises ValueError")

    # T5: unknown prompt_strategy raises ValueError
    class _MockPredictor:
        def set_image(self, *a): pass
        def predict(self, **kw):
            return (np.zeros((1, H, W), dtype=bool), np.array([0.9]), None)

    try:
        get_object_mask(dummy, mask_type="sam",
                        predictor=_MockPredictor(),
                        prompt_strategy="nose_tip")
        assert False
    except ValueError as e:
        assert "nose_tip" in str(e)
    print("T5 PASS — removed strategy raises ValueError")

    # T6: _select_best_mask returns None when below thresholds
    masks_arr  = np.ones((2, 4, 4), dtype=bool)
    scores_arr = np.array([0.5, 0.6])
    stab_arr   = np.ones(2, dtype=np.float32)
    result = _select_best_mask(masks_arr, scores_arr, 0.88, 0.9, stab_arr)
    assert result is None
    print("T6 PASS — _select_best_mask returns None below thresholds")

    # T7: _select_best_mask picks highest-scoring mask above threshold
    scores_pass = np.array([0.91, 0.95])
    result2 = _select_best_mask(masks_arr, scores_pass, 0.88, 0.9, stab_arr)
    assert result2 is not None
    assert result2.dtype == np.uint8
    print("T7 PASS — _select_best_mask picks highest IoU")

    # T8: _postprocess_mask largest component selection
    cm = np.zeros((20, 20), dtype=np.uint8)
    cm[2:8,   2:8]   = 255   # smaller (36px)
    cm[10:18, 10:18] = 255   # larger  (64px)
    r = _postprocess_mask(cm, dilation_px=0)
    assert r[3, 3]   == 0,   "small component should be removed"
    assert r[12, 12] == 255, "large component should be kept"
    print("T8 PASS — _postprocess_mask keeps largest component")

    # T9: mask_to_tensor shape and dtype
    mask9 = np.full((H, W), 255, dtype=np.uint8)
    t9    = mask_to_tensor(mask9)
    assert t9.shape == (1, 1, H, W)
    assert t9.dtype == torch.float32
    assert float(t9.max()) == 1.0
    print("T9 PASS — mask_to_tensor shape, dtype, value range")

    # T10: _merge_boxes produces enclosing box
    b1 = np.array([10., 20., 100., 150.], dtype=np.float32)
    b2 = np.array([80., 50., 200., 300.], dtype=np.float32)
    merged = _merge_boxes([b1, b2])
    assert merged[0] == 10.  # min x1
    assert merged[1] == 20.  # min y1
    assert merged[2] == 200. # max x2
    assert merged[3] == 300. # max y2
    print("T10 PASS — _merge_boxes produces correct enclosing box")

    # T11: text prompt fallback to center_point when dino_model is None
    # (uses mock predictor that returns a passing mask)
    class _MockPredictorPass:
        def set_image(self, *a): pass
        def predict(self, **kw):
            m = np.zeros((1, H, W), dtype=bool)
            m[0, 100:400, 100:400] = True
            return m, np.array([0.95]), None

    result_t11 = get_object_mask(
        dummy,
        mask_type       = "sam",
        predictor       = _MockPredictorPass(),
        prompt_strategy = "text",
        text_prompt     = "the wall",
        dino_model      = None,    # triggers fallback warning
    )
    assert result_t11.shape == (H, W)
    assert result_t11.dtype == np.uint8
    print("T11 PASS — text+no_dino falls back to center_point cleanly")

    # T12: empty text_prompt falls back to center_point
    result_t12 = get_object_mask(
        dummy,
        mask_type       = "sam",
        predictor       = _MockPredictorPass(),
        prompt_strategy = "text",
        text_prompt     = "",      # empty → fallback
        dino_model      = None,
    )
    assert result_t12.shape == (H, W)
    print("T12 PASS — empty text_prompt falls back cleanly")

    print("\nAll 12 tests passed.")