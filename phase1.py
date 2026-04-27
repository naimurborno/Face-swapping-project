# phase1.py
"""
Phase 1: Face alignment + Gaussian frequency decomposition.

Extracted from Phase_One_Face_alignment_and_frequency_decomposition.ipynb.
All notebook-specific execution cells, hardcoded Colab paths, and
visualization-only code are removed. This is a pure importable module.

Public API (what run.py imports):
    FaceLandmarkDetector   — MediaPipe-based landmark detector
    align_and_decompose()  — full pipeline, returns DecomposeResult
    DecomposeResult        — NamedTuple: LF, HF, aligned_face, face_mask, yaw_diff

Dependencies:
    pip install mediapipe opencv-python-headless matplotlib
    wget -O face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, NamedTuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ── Landmark index constants ──────────────────────────────────────────────────

ALIGN_POINTS = {
    "right_eye_center": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "left_eye_center":  [362, 382, 381, 380, 374, 373, 390, 249, 263],
    "nose_tip":         [4],
    "mouth_right":      [61],
    "mouth_left":       [291],
}

MP_68_INDICES = [
    162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 168, 6, 197, 195, 5, 4, 45, 220, 115,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 61, 40, 37, 0, 267, 270,
    291, 321, 314, 17, 84, 91, 78, 82, 87, 14, 317, 312, 308, 324,
]


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class FaceLandmarkResult:
    landmarks_478: np.ndarray   # (478, 2) float32 — full MediaPipe mesh
    landmarks_68:  np.ndarray   # (68, 2)  float32 — dlib-compatible subset
    align_pts:     np.ndarray   # (5, 2)   float32 — 5-point alignment anchors
    image_hw:      tuple                            # (H, W)


class DecomposeResult(NamedTuple):
    LF:           np.ndarray   # (H, W, 3) float32  [0, 255]   — shape/tone
    HF:           np.ndarray   # (H, W, 3) float32  [-255, 255] — texture/detail
    aligned_face: np.ndarray   # (H, W, 3) uint8               — warped reference
    face_mask:    np.ndarray   # (H, W)    uint8    0/255       — convex hull mask
    yaw_diff:     float        # rough yaw difference in degrees


# ── Landmark detection ────────────────────────────────────────────────────────

class FaceLandmarkDetector:
    """
    MediaPipe Tasks API face landmark detector.

    Args:
        model_path: Path to face_landmarker.task file.
                    Download with:
                    wget -O face_landmarker.task https://storage.googleapis.com/
                    mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    """

    def __init__(self, model_path: str = "face_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> Optional[FaceLandmarkResult]:
        """
        Detect face landmarks in a BGR image.

        Returns FaceLandmarkResult or None if no face found.
        """
        H, W = image_bgr.shape[:2]
        rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detection_result = self.detector.detect(mp_img)
        if not detection_result.face_landmarks:
            return None

        face_lms = detection_result.face_landmarks[0]
        all_pts  = np.array(
            [(lm.x * W, lm.y * H) for lm in face_lms],
            dtype=np.float32
        )                                   # (478, 2)

        pts_68 = all_pts[MP_68_INDICES]    # (68, 2)

        def _centroid(indices):
            return all_pts[indices].mean(axis=0)

        align_pts = np.array([
            _centroid(ALIGN_POINTS["right_eye_center"]),
            _centroid(ALIGN_POINTS["left_eye_center"]),
            all_pts[ALIGN_POINTS["nose_tip"][0]],
            all_pts[ALIGN_POINTS["mouth_right"][0]],
            all_pts[ALIGN_POINTS["mouth_left"][0]],
        ], dtype=np.float32)                # (5, 2)

        return FaceLandmarkResult(
            landmarks_478=all_pts,
            landmarks_68=pts_68,
            align_pts=align_pts,
            image_hw=(H, W),
        )


# ── Affine alignment ──────────────────────────────────────────────────────────

def compute_affine_transform(
    src_result: FaceLandmarkResult,
    ref_result: FaceLandmarkResult,
) -> tuple:
    """
    Estimate affine matrix M that warps reference face → source pose.
    Uses 5-point alignment anchors (more stable than all 68 points).

    Returns:
        M        : (2, 3) float64 affine matrix
        yaw_diff : float — rough yaw difference in degrees
    """
    M, inliers = cv2.estimateAffinePartial2D(
        ref_result.align_pts,
        src_result.align_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )

    if M is None:
        raise ValueError(
            "Affine estimation failed. "
            "Likely cause: extreme pose difference or landmark detection error."
        )

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    print(f"[phase1] Affine inliers: {n_inliers}/5")
    if n_inliers < 4:
        print("[phase1] WARNING: fewer than 4 inliers — transform is unreliable.")

    # Rough yaw from eye-width ratio
    def _eye_width(pts):
        return abs(pts[1][0] - pts[0][0])

    ratio        = min(_eye_width(src_result.align_pts), _eye_width(ref_result.align_pts)) / \
                   max(_eye_width(src_result.align_pts), _eye_width(ref_result.align_pts))
    yaw_diff_deg = float(np.degrees(np.arccos(np.clip(ratio, 0.0, 1.0))))

    scale    = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    rotation = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    print(
        f"[phase1] scale={scale:.3f} | rotation={rotation:.1f}° | "
        f"yaw_diff≈{yaw_diff_deg:.1f}°"
    )
    if yaw_diff_deg > 35:
        print(f"[phase1] WARNING: yaw_diff {yaw_diff_deg:.1f}° > 35° — warp will degrade.")

    return M, yaw_diff_deg


# ── Warp + mask ───────────────────────────────────────────────────────────────

def warp_reference(
    ref_img:   np.ndarray,
    M:         np.ndarray,
    target_hw: tuple,
) -> np.ndarray:
    """Warp reference image into source pose using affine matrix M."""
    H, W = target_hw
    return cv2.warpAffine(
        ref_img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def build_face_mask(
    landmarks_68: np.ndarray,   # (68, 2) float32
    image_hw:     tuple,
    dilation_px:  int = 10,
) -> np.ndarray:
    """
    Binary mask of the face region from convex hull of 68 landmarks.
    Dilated slightly to avoid hard seams at the face boundary.

    Returns: (H, W) uint8, values 0/255.
    """
    H, W  = image_hw
    hull  = cv2.convexHull(landmarks_68.astype(np.float32))
    mask  = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )
    return cv2.dilate(mask, kernel)


# ── Frequency decomposition ───────────────────────────────────────────────────

def frequency_decompose(
    image:  np.ndarray,     # uint8 BGR
    kernel: int   = 31,
    sigma:  float = 5.0,
) -> tuple:
    """
    Gaussian frequency decomposition.

    Returns:
        LF : float32 [0, 255]      — low-frequency (structure, skin tone)
        HF : float32 [-255, 255]   — high-frequency residual (texture, detail)

    Why Gaussian and not FFT:
        Gaussian has a smooth rolloff — no ringing artifacts at boundaries.
        FFT with a rectangular mask causes Gibbs ringing which corrupts
        face textures and iris detail.
    """
    assert kernel % 2 == 1, f"kernel must be odd, got {kernel}"
    img_f = image.astype(np.float32)
    LF    = cv2.GaussianBlur(img_f, (kernel, kernel), sigmaX=sigma)
    HF    = img_f - LF
    return LF, HF


# ── Visualization (optional, does not affect pipeline) ────────────────────────

def visualize_landmarks(image_bgr: np.ndarray, result: FaceLandmarkResult):
    vis = image_bgr.copy()
    for (x, y) in result.landmarks_68:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
    labels = ["R_eye", "L_eye", "Nose", "M_right", "M_left"]
    for i, (x, y) in enumerate(result.align_pts):
        cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(vis, labels[i], (int(x)+6, int(y)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def visualize_decomposition(
    source_img:   np.ndarray,
    aligned_face: np.ndarray,
    LF:           np.ndarray,
    HF:           np.ndarray,
    face_mask:    np.ndarray,
):
    def _to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    HF_display = np.clip(HF + 128, 0, 255).astype(np.uint8)
    mask_overlay = source_img.copy()
    mask_overlay[face_mask > 0] = (
        mask_overlay[face_mask > 0] * 0.6 +
        np.array([0, 255, 0], dtype=np.float32) * 0.4
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    panels = [
        (_to_rgb(source_img),                                   "Source"),
        (_to_rgb(aligned_face),                                 "Aligned Ref"),
        (_to_rgb(np.clip(LF, 0, 255).astype(np.uint8)),        "LF (shape)"),
        (_to_rgb(HF_display),                                   "HF (texture)"),
        (_to_rgb(mask_overlay),                                 "Face Mask"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img); ax.set_title(title, fontsize=10); ax.axis("off")

    plt.suptitle(
        "Phase 1 sanity check\n"
        "LF should be smooth/blurry | HF should show edges + iris + texture",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    plt.show()

    print(
        f"[phase1] HF stats → mean: {HF.mean():.3f} (expect ~0) | "
        f"std: {HF.std():.2f} (expect >10) | "
        f"max_abs: {np.abs(HF).max():.1f}"
    )


# ── Public pipeline function ──────────────────────────────────────────────────

def align_and_decompose(
    source_img:    np.ndarray,
    reference_img: np.ndarray,
    gauss_kernel:  int   = 31,
    gauss_sigma:   float = 5.0,
    dilation_px:   int   = 10,
    visualize:     bool  = True,
    model_path:    str   = "face_landmarker.task",
) -> DecomposeResult:
    """
    Full Phase 1 pipeline:
        1. Detect landmarks in source and reference
        2. Estimate affine transform (reference → source pose)
        3. Warp reference into source pose
        4. Build face mask from source landmarks
        5. Gaussian LF/HF decomposition of the aligned reference

    Args:
        source_img    : BGR uint8 — defines the target pose/background
        reference_img : BGR uint8 — face whose identity we transfer
        gauss_kernel  : Gaussian kernel size (must be odd). Default 31.
        gauss_sigma   : Gaussian sigma. Default 5.0.
                        Larger kernel/sigma = smoother LF, more detail in HF.
        dilation_px   : Mask dilation in pixels. Default 10.
        visualize     : Whether to render the sanity check plot.
        model_path    : Path to face_landmarker.task model file.

    Returns:
        DecomposeResult(LF, HF, aligned_face, face_mask, yaw_diff)

    Raises:
        ValueError if face detection fails on either image, or if
        affine estimation fails (extreme pose mismatch).
    """
    detector = FaceLandmarkDetector(model_path)

    src_result = detector.detect(source_img)
    ref_result = detector.detect(reference_img)

    if src_result is None:
        raise ValueError("[phase1] No face detected in source image.")
    if ref_result is None:
        raise ValueError("[phase1] No face detected in reference image.")

    M, yaw_diff = compute_affine_transform(src_result, ref_result)

    aligned_face = warp_reference(reference_img, M, src_result.image_hw)
    face_mask    = build_face_mask(src_result.landmarks_68, src_result.image_hw, dilation_px)

    LF, HF = frequency_decompose(aligned_face, kernel=gauss_kernel, sigma=gauss_sigma)

    if visualize:
        visualize_decomposition(source_img, aligned_face, LF, HF, face_mask)

    return DecomposeResult(
        LF=LF,
        HF=HF,
        aligned_face=aligned_face,
        face_mask=face_mask,
        yaw_diff=yaw_diff,
    )