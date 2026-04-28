# core/alignment.py
"""
Face landmark detection and affine alignment.

Single responsibility: detect MediaPipe landmarks, estimate the affine
transform that maps reference pose → source pose, and warp the reference
image into source space.

Decomposition and masking are intentionally NOT here — they live in
core/decomposition.py and core/segmentation.py respectively.

Public API:
    FaceLandmarkResult  — dataclass holding all landmark arrays + image_hw
    FaceLandmarkDetector — MediaPipe Tasks API wrapper
    compute_affine_transform() — estimate (M, yaw_diff) from two results
    warp_reference()           — apply affine warp to reference image
    run_alignment()            — convenience: detect + warp in one call

Dependencies:
    pip install mediapipe opencv-python-headless
    Download face_landmarker.task:
        wget -O face_landmarker.task https://storage.googleapis.com/mediapipe-models/
        face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ── Landmark index constants ───────────────────────────────────────────────────
# These are fixed MediaPipe 478-point mesh indices.
# Do not change unless MediaPipe updates its topology.

ALIGN_POINTS = {
    # Eye center clusters — averaged to get a stable centroid per eye
    "right_eye_center": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "left_eye_center":  [362, 382, 381, 380, 374, 373, 390, 249, 263],
    # Single-point anchors
    "nose_tip":         [4],
    "mouth_right":      [61],
    "mouth_left":       [291],
}

# 68-point dlib-compatible subset of the 478-point mesh.
# Useful for convex hull masking and external tools expecting dlib format.
MP_68_INDICES = [
    162,  21,  54, 103,  67, 109,  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
     70,  63, 105,  66, 107, 336, 296, 334, 293, 300, 168,   6, 197, 195,   5,   4,  45,
    220, 115,  33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,  61,  40,  37,
      0, 267, 270, 291, 321, 314,  17,  84,  91,  78,  82,  87,  14, 317, 312, 308, 324,
]

# SAM point-prompt indices (used by segmentation.py — exposed here so the
# prompt strategy can derive coordinates directly from FaceLandmarkResult)
SAM_PROMPT_INDICES = {
    "nose_tip":    4,     # single nose tip point — reliable, unambiguous
    "eye_center":  None,  # computed as mean of right + left eye centroids
}


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class FaceLandmarkResult:
    """
    All face landmark data for one image, in pixel coordinates.

    landmarks_478 : (478, 2) float32  — full MediaPipe mesh, pixel coords
    landmarks_68  : (68,  2) float32  — dlib-compatible subset
    align_pts     : (5,   2) float32  — 5-point alignment anchors:
                                        [R_eye, L_eye, Nose, M_right, M_left]
    image_hw      : (H, W)  int tuple — original image dimensions
    nose_tip_px   : (2,)    float32  — nose tip pixel coord (landmark index 4)
    eye_center_px : (2,)    float32  — midpoint of both eye centroids
    bbox_xyxy     : (4,)    float32  — tight bounding box [x1, y1, x2, y2]
                                       over all 478 landmarks
    """
    landmarks_478 : np.ndarray   # (478, 2) float32
    landmarks_68  : np.ndarray   # (68,  2) float32
    align_pts     : np.ndarray   # (5,   2) float32
    image_hw      : Tuple[int, int]
    nose_tip_px   : np.ndarray   # (2,)  float32
    eye_center_px : np.ndarray   # (2,)  float32
    bbox_xyxy     : np.ndarray   # (4,)  float32  [x1, y1, x2, y2]


# ── Detector ───────────────────────────────────────────────────────────────────

class FaceLandmarkDetector:
    """
    MediaPipe Tasks API face landmark detector.

    Wraps FaceLandmarker and returns a FaceLandmarkResult with all derived
    quantities pre-computed so downstream modules (segmentation, alignment)
    never need to re-index MediaPipe landmarks themselves.

    Args:
        model_path : Path to face_landmarker.task file.
        min_face_detection_confidence : Detection threshold (default 0.3 is
            deliberately low to handle partially occluded or profile faces).
        min_face_presence_confidence  : Presence threshold.
    """

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
        min_face_detection_confidence: float = 0.3,
        min_face_presence_confidence:  float = 0.3,
    ):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> Optional[FaceLandmarkResult]:
        """
        Detect face landmarks in a BGR uint8 image.

        Returns FaceLandmarkResult if a face is found, None otherwise.
        Input must be BGR (OpenCV convention) — the function converts to RGB
        internally for MediaPipe.
        """
        H, W = image_bgr.shape[:2]
        rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_img)
        if not result.face_landmarks:
            return None

        face_lms = result.face_landmarks[0]

        # All 478 points in pixel coordinates
        all_pts = np.array(
            [(lm.x * W, lm.y * H) for lm in face_lms],
            dtype=np.float32,
        )  # (478, 2)

        # 68-point dlib-compatible subset
        pts_68 = all_pts[MP_68_INDICES]  # (68, 2)

        # 5-point alignment anchors
        def _centroid(indices: list) -> np.ndarray:
            return all_pts[indices].mean(axis=0)

        r_eye = _centroid(ALIGN_POINTS["right_eye_center"])
        l_eye = _centroid(ALIGN_POINTS["left_eye_center"])
        nose  = all_pts[ALIGN_POINTS["nose_tip"][0]]
        m_r   = all_pts[ALIGN_POINTS["mouth_right"][0]]
        m_l   = all_pts[ALIGN_POINTS["mouth_left"][0]]

        align_pts = np.array([r_eye, l_eye, nose, m_r, m_l], dtype=np.float32)

        # SAM prompt helpers
        nose_tip_px   = all_pts[4].copy()
        eye_center_px = ((r_eye + l_eye) / 2.0).astype(np.float32)

        # Tight bounding box over all 478 landmarks
        x1, y1 = all_pts.min(axis=0)
        x2, y2 = all_pts.max(axis=0)
        # Clamp to image bounds
        x1 = float(np.clip(x1, 0, W - 1))
        y1 = float(np.clip(y1, 0, H - 1))
        x2 = float(np.clip(x2, 0, W - 1))
        y2 = float(np.clip(y2, 0, H - 1))
        bbox_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)

        return FaceLandmarkResult(
            landmarks_478 = all_pts,
            landmarks_68  = pts_68,
            align_pts     = align_pts,
            image_hw      = (H, W),
            nose_tip_px   = nose_tip_px,
            eye_center_px = eye_center_px,
            bbox_xyxy     = bbox_xyxy,
        )


# ── Affine transform ───────────────────────────────────────────────────────────

def compute_affine_transform(
    src_result: FaceLandmarkResult,
    ref_result: FaceLandmarkResult,
    max_yaw_warning_deg: float = 35.0,
) -> Tuple[np.ndarray, float]:
    """
    Estimate the affine transform M that maps reference pose → source pose.

    Uses 5-point alignment anchors with RANSAC for robustness to detection
    noise. Partial affine (rotation + scale + translation, no shear) is used
    because faces are approximately rigid at moderate pose differences.

    Args:
        src_result           : Landmarks from the source image.
        ref_result           : Landmarks from the reference image.
        max_yaw_warning_deg  : Log a warning if estimated yaw gap exceeds this.

    Returns:
        M        : (2, 3) float64 affine matrix
        yaw_diff : float — rough yaw difference in degrees estimated from
                            the inter-eye width ratio.

    Raises:
        ValueError if RANSAC fails to find an affine transform. This can
        happen with extreme pose differences (>60°) or failed landmark
        detection on one image.
    """
    M, inliers = cv2.estimateAffinePartial2D(
        ref_result.align_pts,   # source points (reference)
        src_result.align_pts,   # destination points (source)
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )

    if M is None:
        raise ValueError(
            "[alignment] Affine estimation failed.\n"
            "  Likely causes:\n"
            "    1. Extreme pose difference (try images with < 35° yaw gap)\n"
            "    2. One detection had very low confidence\n"
            "    3. Partial occlusion at landmark anchor points"
        )

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    print(f"[alignment] Affine inliers: {n_inliers}/5")
    if n_inliers < 4:
        print(
            "[alignment] WARNING: fewer than 4 inliers — "
            "transform may be unreliable. Check landmark detection quality."
        )

    # ── Yaw estimation from inter-eye width ratio ──────────────────────────
    # When a face turns, the projected eye-width shrinks by cos(yaw).
    # Ratio of the narrower / wider gives cos(yaw_diff).
    def _eye_width(pts: np.ndarray) -> float:
        return float(abs(pts[1, 0] - pts[0, 0]))

    src_ew = _eye_width(src_result.align_pts)
    ref_ew = _eye_width(ref_result.align_pts)

    if max(src_ew, ref_ew) < 1e-6:
        yaw_diff = 0.0
    else:
        ratio    = min(src_ew, ref_ew) / max(src_ew, ref_ew)
        yaw_diff = float(np.degrees(np.arccos(np.clip(ratio, 0.0, 1.0))))

    # ── Decompose M for logging ────────────────────────────────────────────
    scale    = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    rotation = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    tx, ty   = float(M[0, 2]), float(M[1, 2])

    print(
        f"[alignment] scale={scale:.3f} | rotation={rotation:.1f}° | "
        f"tx={tx:.1f} ty={ty:.1f} | yaw_diff≈{yaw_diff:.1f}°"
    )

    if yaw_diff > max_yaw_warning_deg:
        print(
            f"[alignment] WARNING: yaw_diff {yaw_diff:.1f}° > {max_yaw_warning_deg}°.\n"
            f"  Affine warp is unreliable at this pose gap.\n"
            f"  Consider using a reference image closer in angle to the source."
        )

    return M, yaw_diff


# ── Warp ───────────────────────────────────────────────────────────────────────

def warp_reference(
    ref_img:   np.ndarray,
    M:         np.ndarray,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Warp reference image into source image space using affine matrix M.

    BORDER_REFLECT is used at boundaries so that edge regions have plausible
    pixel values rather than black borders, which would corrupt the VAE encoding
    of those areas.

    Args:
        ref_img   : BGR uint8 reference image.
        M         : (2, 3) float64 affine matrix from compute_affine_transform().
        target_hw : (H, W) output canvas size — matches source image dimensions.

    Returns:
        Warped reference as BGR uint8, same size as target_hw.
    """
    H, W = target_hw
    return cv2.warpAffine(
        ref_img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


# ── Convenience wrapper ────────────────────────────────────────────────────────

class AlignmentResult:
    """
    Everything produced by run_alignment(), ready for downstream modules.

    src_result    : FaceLandmarkResult for source image
    ref_result    : FaceLandmarkResult for reference image (original pose)
    aligned_face  : (H, W, 3) uint8  — reference warped into source pose
    affine_M      : (2, 3) float64   — affine matrix used for the warp
    yaw_diff      : float            — estimated yaw gap in degrees
    """
    __slots__ = ("src_result", "ref_result", "aligned_face", "affine_M", "yaw_diff")

    def __init__(self, src_result, ref_result, aligned_face, affine_M, yaw_diff):
        self.src_result   = src_result
        self.ref_result   = ref_result
        self.aligned_face = aligned_face
        self.affine_M     = affine_M
        self.yaw_diff     = yaw_diff


def run_alignment(
    source_bgr:    np.ndarray,
    reference_bgr: np.ndarray,
    model_path:    str   = "face_landmarker.task",
    min_face_detection_confidence: float = 0.3,
    min_face_presence_confidence:  float = 0.3,
    max_yaw_warning_deg:           float = 35.0,
) -> AlignmentResult:
    """
    Full alignment pipeline in one call.

    1. Detect landmarks in both images.
    2. Estimate affine transform (reference → source pose).
    3. Warp reference into source pose.

    Does NOT do decomposition or masking — call core/decomposition.py and
    core/segmentation.py separately after this returns.

    Args:
        source_bgr    : BGR uint8 — the pose/background to keep.
        reference_bgr : BGR uint8 — the face identity to transfer.
        model_path    : Path to face_landmarker.task.
        min_face_detection_confidence : Passed to FaceLandmarkDetector.
        min_face_presence_confidence  : Passed to FaceLandmarkDetector.
        max_yaw_warning_deg           : Passed to compute_affine_transform.

    Returns:
        AlignmentResult

    Raises:
        ValueError if face detection fails on either image, or if affine
        estimation fails.
    """
    detector = FaceLandmarkDetector(
        model_path=model_path,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
    )

    src_result = detector.detect(source_bgr)
    ref_result = detector.detect(reference_bgr)

    if src_result is None:
        raise ValueError(
            "[alignment] No face detected in source image.\n"
            "  Check image quality, lighting, and that the face is frontal."
        )
    if ref_result is None:
        raise ValueError(
            "[alignment] No face detected in reference image.\n"
            "  Check image quality, lighting, and that the face is frontal."
        )

    print(
        f"[alignment] Source   : {src_result.image_hw[1]}×{src_result.image_hw[0]}px | "
        f"bbox {src_result.bbox_xyxy.astype(int).tolist()}"
    )
    print(
        f"[alignment] Reference: {ref_result.image_hw[1]}×{ref_result.image_hw[0]}px | "
        f"bbox {ref_result.bbox_xyxy.astype(int).tolist()}"
    )

    M, yaw_diff = compute_affine_transform(
        src_result, ref_result,
        max_yaw_warning_deg=max_yaw_warning_deg,
    )

    aligned_face = warp_reference(reference_bgr, M, src_result.image_hw)

    return AlignmentResult(
        src_result   = src_result,
        ref_result   = ref_result,
        aligned_face = aligned_face,
        affine_M     = M,
        yaw_diff     = yaw_diff,
    )


# ── Convex hull mask (ablation fallback — lives here, used by segmentation.py) ─

def build_convex_hull_mask(
    landmarks_68: np.ndarray,
    image_hw:     Tuple[int, int],
    dilation_px:  int = 10,
) -> np.ndarray:
    """
    Binary face mask from convex hull of 68 landmarks.

    Used when ablation.mask_type = "convex_hull". When mask_type = "sam",
    segmentation.py produces the mask instead.

    Args:
        landmarks_68 : (68, 2) float32 landmark coordinates.
        image_hw     : (H, W) image dimensions.
        dilation_px  : Pixels to dilate outward to soften boundary seams.

    Returns:
        (H, W) uint8 mask, values 0/255.
    """
    H, W  = image_hw
    hull  = cv2.convexHull(landmarks_68.astype(np.float32))
    mask  = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    if dilation_px > 0:
        ksize  = dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask   = cv2.dilate(mask, kernel)

    return mask