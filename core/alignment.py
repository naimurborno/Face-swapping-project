# core/alignment.py
"""
Donor-to-content spatial alignment for the Mixed-Frequency Prior Guided
Inpainting pipeline.

Single responsibility: register the donor image R into the content image S's
spatial coordinate system so that frequency decomposition and prior
construction in decomposition.py operate on spatially consistent signals.

WHY ALIGNMENT MATTERS FOR PRIOR CONSTRUCTION:
    build_prior() computes P = α·S_LF + β·R̃_LF + γ·R̃_HF pixel-by-pixel.
    If the donor image R is not registered to S's spatial layout, the
    frequency bands from R and S will be misaligned — edges from R will
    appear in the wrong positions relative to S_LF, producing a spatially
    incoherent prior that the inpainting model will treat as noise.
    Alignment ensures R̃ occupies the same spatial support as the editable
    mask region M before any frequency operations are performed.

ALIGNMENT METHODS (selected via configs/default.yaml → alignment.method):

    "resize"  — Resize donor to content canvas size.
                Correct when donor and content have the same object category
                and similar spatial layout (e.g. wall → wall, floor → floor).
                Fast, no dependencies beyond cv2. Default method.

    "tile"    — Tile donor to fill content canvas, then resize to target.
                Correct when donor is a tileable texture (fabric, stone, wood
                grain) that needs to cover a larger surface.
                Use when: donor is smaller than content, or texture has a
                repeating pattern that should cover the full masked region.

    "match"   — Feature-based alignment via ORB keypoint matching + affine
                RANSAC. Registers donor to content based on visual structure.
                Correct when donor and content share common visual features
                (e.g. same building, same object from different angles).
                Degrades gracefully to "resize" if match confidence is low.

    "flow"    — Dense optical flow warp (Farneback).
                Registers donor to content by dense pixel correspondence.
                Correct for highly similar images with smooth deformation
                (e.g. same face from slightly different expressions).
                Most expensive. Falls back to "resize" on flow failure.

    "affine"  — Explicit affine transform from user-supplied source and
                destination point pairs.
                Use when automatic methods fail and you have known
                correspondences (e.g. 4 corner points of a surface).

All methods return an AlignmentResult with the aligned donor image R̃ and
a record of which method was used and its quality metrics.

Public API:
    AlignmentResult          — dataclass: aligned_donor, method_used, meta
    align_resize()           — resize donor to content canvas
    align_tile()             — tile donor to fill content canvas
    align_feature_match()    — ORB feature matching + affine RANSAC
    align_flow()             — dense optical flow warp
    align_affine()           — explicit affine from point pairs
    run_donor_alignment()    — main entry point: dispatches by method flag

Dependencies:
    pip install opencv-python-headless numpy pillow
    No MediaPipe. No face landmarker model required.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from PIL import Image


# ── Output container ──────────────────────────────────────────────────────────

@dataclass
class AlignmentResult:
    """
    Output of run_donor_alignment(). Contains the aligned donor image and
    metadata about the alignment quality for logging to meta.json.

    aligned_donor : (H, W, 3) uint8 BGR — donor R̃ registered to content S.
                    Same spatial size as content image.
                    Ready for decomposition.build_prior().

    method_used   : str — which alignment method was actually executed.
                    May differ from requested method if fallback occurred
                    (e.g. "match" requested but fell back to "resize").

    fallback      : bool — True if the requested method failed and the
                    pipeline fell back to a simpler method.

    meta          : dict — method-specific quality metrics for logging.
                    resize  : {"scale_x", "scale_y"}
                    tile    : {"tile_x", "tile_y", "scale_x", "scale_y"}
                    match   : {"n_matches", "n_inliers", "inlier_ratio",
                               "scale", "rotation_deg", "tx", "ty"}
                    flow    : {"mean_flow_magnitude", "max_flow_magnitude"}
                    affine  : {"scale", "rotation_deg", "tx", "ty"}
    """
    aligned_donor : np.ndarray              # (H, W, 3) uint8 BGR
    method_used   : str
    fallback      : bool = False
    meta          : dict = field(default_factory=dict)


# ── Primitive: resize ─────────────────────────────────────────────────────────

def align_resize(
    donor_bgr:   np.ndarray,   # (H_d, W_d, 3) uint8
    content_hw:  Tuple[int, int],
) -> AlignmentResult:
    """
    Align donor to content by simple resize.

    The donor is resized to exactly (H, W) of the content image using
    LANCZOS interpolation. No spatial warping is applied — only scale.

    WHEN TO USE:
        - Donor and content have the same object category (wall → wall).
        - Donor occupies a similar layout to the masked region.
        - Speed is a priority and fine spatial alignment is not needed.

    Args:
        donor_bgr  : BGR uint8 donor image (any size).
        content_hw : (H, W) target size matching content image.

    Returns:
        AlignmentResult with method_used="resize".
    """
    H, W       = content_hw
    H_d, W_d   = donor_bgr.shape[:2]

    aligned = cv2.resize(donor_bgr, (W, H), interpolation=cv2.INTER_LANCZOS4)

    scale_x = W / W_d
    scale_y = H / H_d

    print(
        f"[alignment] resize | "
        f"donor {W_d}×{H_d} → content {W}×{H} | "
        f"scale_x={scale_x:.3f} scale_y={scale_y:.3f}"
    )

    return AlignmentResult(
        aligned_donor = aligned,
        method_used   = "resize",
        fallback      = False,
        meta          = {"scale_x": scale_x, "scale_y": scale_y},
    )


# ── Primitive: tile ───────────────────────────────────────────────────────────

def align_tile(
    donor_bgr:   np.ndarray,        # (H_d, W_d, 3) uint8
    content_hw:  Tuple[int, int],
    tile_scale:  float = 1.0,
) -> AlignmentResult:
    """
    Align donor by tiling it to fill the content canvas, then resize.

    Tiles the donor image in a regular grid until it covers at least
    (H, W) of the content, then crops and resizes to exactly (H, W).

    WHEN TO USE:
        - Donor is a repeating texture (fabric, brick, marble, wood grain).
        - Donor is smaller than the content masked region.
        - Texture should cover the full masked surface uniformly.

    tile_scale controls how large each tile appears relative to the canvas:
        1.0 → tile at native donor size (donor pixels map 1:1 to content pixels)
        0.5 → each tile occupies half the content canvas (2×2 tiling)
        2.0 → donor is scaled up 2× before tiling (larger texture features)

    Args:
        donor_bgr  : BGR uint8 donor image.
        content_hw : (H, W) target size matching content image.
        tile_scale : Scale factor applied to donor before tiling. Default 1.0.

    Returns:
        AlignmentResult with method_used="tile".
    """
    H, W = content_hw

    # Scale donor before tiling
    if tile_scale != 1.0:
        H_d_new = max(1, int(donor_bgr.shape[0] * tile_scale))
        W_d_new = max(1, int(donor_bgr.shape[1] * tile_scale))
        tile    = cv2.resize(donor_bgr, (W_d_new, H_d_new),
                             interpolation=cv2.INTER_LANCZOS4)
    else:
        tile = donor_bgr

    H_t, W_t = tile.shape[:2]

    # Compute how many tiles needed to cover content canvas
    n_y = int(np.ceil(H / H_t)) + 1
    n_x = int(np.ceil(W / W_t)) + 1

    # Tile by repeating
    tiled = np.tile(tile, (n_y, n_x, 1))   # (n_y*H_t, n_x*W_t, 3)

    # Crop to content canvas size
    tiled_cropped = tiled[:H, :W, :]

    # Final resize to exactly (H, W) — handles any rounding
    aligned = cv2.resize(tiled_cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

    print(
        f"[alignment] tile | "
        f"donor {donor_bgr.shape[1]}×{donor_bgr.shape[0]} | "
        f"tile_scale={tile_scale} | "
        f"grid={n_x}×{n_y} tiles | "
        f"output {W}×{H}"
    )

    return AlignmentResult(
        aligned_donor = aligned,
        method_used   = "tile",
        fallback      = False,
        meta          = {
            "tile_x"  : n_x,
            "tile_y"  : n_y,
            "scale_x" : W / donor_bgr.shape[1],
            "scale_y" : H / donor_bgr.shape[0],
        },
    )


# ── Primitive: feature matching ───────────────────────────────────────────────

def align_feature_match(
    donor_bgr:       np.ndarray,      # (H_d, W_d, 3) uint8
    content_bgr:     np.ndarray,      # (H, W, 3) uint8
    min_inlier_ratio: float = 0.25,   # below this, fall back to resize
    n_features:       int   = 1000,
) -> AlignmentResult:
    """
    Align donor to content via ORB keypoint matching and affine RANSAC.

    Detects ORB keypoints in both images, matches with BFMatcher, filters
    matches with Lowe's ratio test, and estimates a partial affine transform
    (rotation + scale + translation, no shear) via RANSAC.

    WHEN TO USE:
        - Donor and content share common visual features or structure.
        - Same object photographed from different angles or conditions.
        - Feature-level correspondences exist (edges, corners, blobs).

    FALLBACK:
        If fewer than 4 matches survive RANSAC, or the inlier ratio falls
        below min_inlier_ratio, the method falls back to "resize" and sets
        AlignmentResult.fallback=True. This ensures the pipeline never
        crashes on low-texture donors (solid colors, smooth gradients).

    Args:
        donor_bgr        : BGR uint8 donor image.
        content_bgr      : BGR uint8 content image (alignment target).
        min_inlier_ratio : Minimum fraction of good matches that must be
                           inliers for the transform to be accepted.
        n_features       : Number of ORB keypoints to detect per image.

    Returns:
        AlignmentResult with method_used="match" or "resize" (if fallback).
    """
    H, W = content_bgr.shape[:2]

    # Resize donor to content size first so feature scales are comparable
    donor_resized = cv2.resize(donor_bgr, (W, H), interpolation=cv2.INTER_LANCZOS4)

    # Convert to grayscale for feature detection
    gray_content = cv2.cvtColor(content_bgr,    cv2.COLOR_BGR2GRAY)
    gray_donor   = cv2.cvtColor(donor_resized,  cv2.COLOR_BGR2GRAY)

    # ── ORB detection ─────────────────────────────────────────────────────
    orb = cv2.ORB_create(nfeatures=n_features)
    kp_c, desc_c = orb.detectAndCompute(gray_content, None)
    kp_d, desc_d = orb.detectAndCompute(gray_donor,   None)

    # Handle degenerate cases (low-texture images produce no keypoints)
    if desc_c is None or desc_d is None or len(kp_c) < 4 or len(kp_d) < 4:
        print(
            f"[alignment] match | insufficient keypoints "
            f"(content={len(kp_c) if kp_c else 0}, "
            f"donor={len(kp_d) if kp_d else 0}) → fallback to resize"
        )
        result         = align_resize(donor_bgr, (H, W))
        result.fallback = True
        return result

    # ── BFMatcher with Lowe ratio test ────────────────────────────────────
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc_d, desc_c, k=2)

    # Lowe ratio test — keep matches where best is significantly better than second
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 4:
        print(
            f"[alignment] match | only {len(good)} good matches after "
            f"ratio test (need ≥ 4) → fallback to resize"
        )
        result          = align_resize(donor_bgr, (H, W))
        result.fallback = True
        return result

    # ── Extract matched keypoint coordinates ──────────────────────────────
    src_pts = np.float32([kp_d[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # ── Partial affine RANSAC ─────────────────────────────────────────────
    # estimateAffinePartial2D: rotation + uniform scale + translation only.
    # No shear. Appropriate for texture/material images where we do not want
    # perspective distortion introduced by the alignment.
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method        = cv2.RANSAC,
        ransacReprojThreshold = 5.0,
    )

    n_inliers    = int(inliers.sum()) if inliers is not None else 0
    inlier_ratio = n_inliers / len(good)

    print(
        f"[alignment] match | "
        f"good_matches={len(good)} | "
        f"inliers={n_inliers} ({inlier_ratio*100:.1f}%)"
    )

    # Check if transform quality is sufficient
    if M is None or inlier_ratio < min_inlier_ratio or n_inliers < 4:
        print(
            f"[alignment] match | inlier_ratio={inlier_ratio:.2f} < "
            f"{min_inlier_ratio} → fallback to resize"
        )
        result          = align_resize(donor_bgr, (H, W))
        result.fallback = True
        return result

    # ── Apply affine warp ─────────────────────────────────────────────────
    aligned = cv2.warpAffine(
        donor_resized, M, (W, H),
        flags      = cv2.INTER_LANCZOS4,
        borderMode = cv2.BORDER_REFLECT,
    )

    # Decompose M for logging
    scale       = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    rotation    = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    tx, ty      = float(M[0, 2]), float(M[1, 2])

    print(
        f"[alignment] match | "
        f"scale={scale:.3f} rotation={rotation:.1f}° "
        f"tx={tx:.1f} ty={ty:.1f}"
    )

    return AlignmentResult(
        aligned_donor = aligned,
        method_used   = "match",
        fallback      = False,
        meta          = {
            "n_matches"    : len(good),
            "n_inliers"    : n_inliers,
            "inlier_ratio" : round(inlier_ratio, 4),
            "scale"        : round(scale, 4),
            "rotation_deg" : round(rotation, 2),
            "tx"           : round(tx, 2),
            "ty"           : round(ty, 2),
        },
    )


# ── Primitive: dense optical flow ─────────────────────────────────────────────

def align_flow(
    donor_bgr:   np.ndarray,    # (H_d, W_d, 3) uint8
    content_bgr: np.ndarray,    # (H, W, 3) uint8
) -> AlignmentResult:
    """
    Align donor to content via dense optical flow warp (Farneback).

    Computes per-pixel flow from donor to content and applies a dense
    remap warp. Produces the most spatially precise alignment but is the
    most computationally expensive method and is sensitive to large
    appearance differences between donor and content.

    WHEN TO USE:
        - Donor and content are visually very similar (same scene, small
          deformation — e.g. same material under different lighting).
        - Fine-grained spatial correspondence is required.
        - Feature matching (ORB) fails because the images are too similar
          for keypoint detection but share dense visual structure.

    FALLBACK:
        If the flow magnitude is very low (images are already nearly aligned)
        or if remap produces mostly black output (flow diverged), the method
        falls back to "resize" with fallback=True.

    Args:
        donor_bgr   : BGR uint8 donor image.
        content_bgr : BGR uint8 content image (alignment target).

    Returns:
        AlignmentResult with method_used="flow" or "resize" (if fallback).
    """
    H, W = content_bgr.shape[:2]

    # Resize donor to content size before flow computation
    donor_resized = cv2.resize(donor_bgr, (W, H), interpolation=cv2.INTER_LANCZOS4)

    gray_content = cv2.cvtColor(content_bgr,   cv2.COLOR_BGR2GRAY)
    gray_donor   = cv2.cvtColor(donor_resized, cv2.COLOR_BGR2GRAY)

    # ── Farneback dense optical flow: donor → content ─────────────────────
    # Flow field gives per-pixel displacement from donor to content.
    flow = cv2.calcOpticalFlowFarneback(
        gray_donor,    # prev
        gray_content,  # next
        None,
        pyr_scale  = 0.5,
        levels     = 3,
        winsize    = 15,
        iterations = 3,
        poly_n     = 5,
        poly_sigma = 1.2,
        flags      = 0,
    )  # (H, W, 2) — flow[:,:,0]=dx, flow[:,:,1]=dy

    mean_mag = float(np.sqrt((flow ** 2).sum(axis=2)).mean())
    max_mag  = float(np.sqrt((flow ** 2).sum(axis=2)).max())

    print(
        f"[alignment] flow | "
        f"mean_magnitude={mean_mag:.2f}px | "
        f"max_magnitude={max_mag:.2f}px"
    )

    # ── Build remap grid ──────────────────────────────────────────────────
    # For each pixel (x, y) in output, sample from donor at (x - dx, y - dy)
    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (grid_x - flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y - flow[:, :, 1]).astype(np.float32)

    aligned = cv2.remap(
        donor_resized, map_x, map_y,
        interpolation = cv2.INTER_LANCZOS4,
        borderMode    = cv2.BORDER_REFLECT,
    )

    # Sanity check — if remap produced mostly black, flow diverged
    if aligned.mean() < 5.0:
        print("[alignment] flow | remap produced dark output → fallback to resize")
        result          = align_resize(donor_bgr, (H, W))
        result.fallback = True
        return result

    return AlignmentResult(
        aligned_donor = aligned,
        method_used   = "flow",
        fallback      = False,
        meta          = {
            "mean_flow_magnitude" : round(mean_mag, 3),
            "max_flow_magnitude"  : round(max_mag,  3),
        },
    )


# ── Primitive: explicit affine ────────────────────────────────────────────────

def align_affine(
    donor_bgr:   np.ndarray,          # (H_d, W_d, 3) uint8
    content_hw:  Tuple[int, int],
    src_points:  np.ndarray,          # (N≥3, 2) float32 — points in donor
    dst_points:  np.ndarray,          # (N≥3, 2) float32 — corresponding points in content
) -> AlignmentResult:
    """
    Align donor to content via an explicit affine transform from point pairs.

    Estimates the best-fit partial affine transform from N≥3 corresponding
    point pairs using least squares (no RANSAC — all points are trusted).

    WHEN TO USE:
        - Automatic methods fail and you have manually specified
          correspondences (e.g. 4 corners of a wall panel).
        - You have a known geometric relationship between donor and content
          (e.g. homography corners computed externally).
        - Precise user-guided alignment is needed.

    Args:
        donor_bgr   : BGR uint8 donor image.
        content_hw  : (H, W) output size matching content image.
        src_points  : (N, 2) float32 pixel coords in donor image.
        dst_points  : (N, 2) float32 corresponding pixel coords in content image.

    Returns:
        AlignmentResult with method_used="affine".

    Raises:
        ValueError if fewer than 3 point pairs are provided, or if affine
        estimation fails.
    """
    H, W = content_hw

    if len(src_points) < 3 or len(dst_points) < 3:
        raise ValueError(
            f"[alignment] affine requires ≥ 3 point pairs, "
            f"got src={len(src_points)} dst={len(dst_points)}."
        )
    if len(src_points) != len(dst_points):
        raise ValueError(
            f"[alignment] src_points ({len(src_points)}) and "
            f"dst_points ({len(dst_points)}) must have equal length."
        )

    src_pts = src_points.astype(np.float32).reshape(-1, 1, 2)
    dst_pts = dst_points.astype(np.float32).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method = cv2.LMEDS,   # least median of squares — robust to outliers
    )

    if M is None:
        raise ValueError(
            "[alignment] affine estimation failed. "
            "Check that point pairs are not collinear or degenerate."
        )

    # Resize donor to content first (src_points assumed in original donor coords)
    H_d, W_d = donor_bgr.shape[:2]
    scale_pre_x = W / W_d
    scale_pre_y = H / H_d
    donor_resized = cv2.resize(donor_bgr, (W, H), interpolation=cv2.INTER_LANCZOS4)

    aligned = cv2.warpAffine(
        donor_resized, M, (W, H),
        flags      = cv2.INTER_LANCZOS4,
        borderMode = cv2.BORDER_REFLECT,
    )

    scale    = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    rotation = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    tx, ty   = float(M[0, 2]), float(M[1, 2])

    print(
        f"[alignment] affine | "
        f"{len(src_points)} point pairs | "
        f"scale={scale:.3f} rotation={rotation:.1f}° "
        f"tx={tx:.1f} ty={ty:.1f}"
    )

    return AlignmentResult(
        aligned_donor = aligned,
        method_used   = "affine",
        fallback      = False,
        meta          = {
            "n_points"    : len(src_points),
            "scale"       : round(scale,    4),
            "rotation_deg": round(rotation, 2),
            "tx"          : round(tx,       2),
            "ty"          : round(ty,       2),
        },
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_donor_alignment(
    donor_bgr:         np.ndarray,
    content_bgr:       np.ndarray,
    method:            str   = "resize",
    tile_scale:        float = 1.0,
    min_inlier_ratio:  float = 0.25,
    n_features:        int   = 1000,
    src_points:        Optional[np.ndarray] = None,
    dst_points:        Optional[np.ndarray] = None,
) -> AlignmentResult:
    """
    Align donor image R to content image S's spatial coordinate system.

    Main entry point called by stage1_segment.py. Dispatches to the
    appropriate primitive based on the method argument, which maps to
    configs/default.yaml → alignment.method.

    All methods resize the output to content_bgr's exact spatial size
    so that the returned aligned_donor can be passed directly to
    decomposition.build_prior().

    Args:
        donor_bgr        : BGR uint8 donor image R (any size).
        content_bgr      : BGR uint8 content image S.
        method           : "resize" | "tile" | "match" | "flow" | "affine"
                           Default "resize" — safe for all input types.
        tile_scale       : Scale applied to each tile (method="tile" only).
                           1.0 = donor pixels map 1:1 to content pixels.
        min_inlier_ratio : Minimum inlier fraction to accept a feature match
                           (method="match" only). Below this, falls back to resize.
        n_features       : Number of ORB keypoints (method="match" only).
        src_points       : (N, 2) donor point coordinates (method="affine" only).
        dst_points       : (N, 2) content point coordinates (method="affine" only).

    Returns:
        AlignmentResult with aligned_donor, method_used, fallback, meta.

    Raises:
        ValueError for unknown method strings.
        ValueError if method="affine" and src_points/dst_points are not provided.
    """
    valid_methods = ("resize", "tile", "match", "flow", "affine")
    if method not in valid_methods:
        raise ValueError(
            f"[alignment] Unknown method '{method}'. "
            f"Choose: {' | '.join(valid_methods)}"
        )

    content_hw = content_bgr.shape[:2]   # (H, W)

    H_d, W_d = donor_bgr.shape[:2]
    H_c, W_c = content_hw

    print(
        f"[alignment] run_donor_alignment | "
        f"method={method} | "
        f"donor={W_d}×{H_d} content={W_c}×{H_c}"
    )

    # ── Dispatch ──────────────────────────────────────────────────────────

    if method == "resize":
        return align_resize(donor_bgr, content_hw)

    elif method == "tile":
        return align_tile(donor_bgr, content_hw, tile_scale=tile_scale)

    elif method == "match":
        return align_feature_match(
            donor_bgr, content_bgr,
            min_inlier_ratio = min_inlier_ratio,
            n_features       = n_features,
        )

    elif method == "flow":
        return align_flow(donor_bgr, content_bgr)

    elif method == "affine":
        if src_points is None or dst_points is None:
            raise ValueError(
                "[alignment] method='affine' requires src_points and "
                "dst_points to be provided. Pass corresponding pixel "
                "coordinate arrays for donor and content images."
            )
        return align_affine(donor_bgr, content_hw, src_points, dst_points)


# ── Convex hull mask (retained for ablation A3 — mask_type="convex_hull") ─────

def build_convex_hull_mask(
    points:      np.ndarray,        # (N, 2) float32 — any 2D point set
    image_hw:    Tuple[int, int],
    dilation_px: int = 10,
) -> np.ndarray:
    """
    Binary mask from convex hull of a 2D point set.

    Retained as ablation A3 baseline (mask_type="convex_hull") for
    comparison against SAM segmentation masks. In the new pipeline,
    SAM is the default mask source and this function is only called
    when ablation.mask_type = "convex_hull" is set in default.yaml.

    The point set can be any 2D coordinates defining the editable region:
    bounding box corners, landmark points, or user-specified anchors.

    Args:
        points      : (N, 2) float32 pixel coordinates defining the hull.
        image_hw    : (H, W) image dimensions for the output mask canvas.
        dilation_px : Pixels to dilate outward to soften boundary seams.
                      0 = tight hull boundary.

    Returns:
        (H, W) uint8 mask, values 0 or 255.
    """
    H, W  = image_hw
    pts   = points.astype(np.float32)
    hull  = cv2.convexHull(pts)
    mask  = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    if dilation_px > 0:
        ksize  = dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask   = cv2.dilate(mask, kernel)

    print(
        f"[alignment] convex_hull_mask | "
        f"hull_points={len(pts)} | "
        f"dilation_px={dilation_px} | "
        f"coverage={(mask > 0).mean()*100:.1f}%"
    )

    return mask


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    """
    Verify all alignment methods and helpers.
    Runs on CPU only — no GPU, no real images, no MediaPipe required.
    """
    print("[alignment] Running smoke test...\n")

    rng         = np.random.default_rng(0)
    content     = rng.integers(50, 200, (512, 512, 3), dtype=np.uint8)
    donor_same  = rng.integers(50, 200, (512, 512, 3), dtype=np.uint8)
    donor_small = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
    donor_large = rng.integers(50, 200, (768, 512, 3), dtype=np.uint8)

    target_hw = (512, 512)

    # ── resize ────────────────────────────────────────────────────────────
    for donor, label in [
        (donor_same,  "same-size"),
        (donor_small, "small"),
        (donor_large, "large"),
    ]:
        r = align_resize(donor, target_hw)
        assert r.aligned_donor.shape == (512, 512, 3), f"resize {label} shape wrong"
        assert r.aligned_donor.dtype == np.uint8
        assert r.method_used == "resize"
        assert not r.fallback
        print(f"  align_resize()  [{label}] | shape={r.aligned_donor.shape} ✓")

    # ── tile ──────────────────────────────────────────────────────────────
    for scale in [1.0, 0.5, 2.0]:
        r = align_tile(donor_small, target_hw, tile_scale=scale)
        assert r.aligned_donor.shape == (512, 512, 3), f"tile scale={scale} shape wrong"
        assert r.method_used == "tile"
        print(f"  align_tile()    [scale={scale}] | "
              f"shape={r.aligned_donor.shape} "
              f"grid={r.meta['tile_x']}×{r.meta['tile_y']} ✓")

    # ── match — high-texture images ───────────────────────────────────────
    # Use a high-contrast synthetic image so ORB finds keypoints
    content_hc = np.zeros((512, 512, 3), dtype=np.uint8)
    donor_hc   = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(0, 512, 32):
        content_hc[i:i+16, :] = 200
        donor_hc[:,  i:i+16]  = 200

    r = align_feature_match(donor_hc, content_hc)
    assert r.aligned_donor.shape == (512, 512, 3)
    assert r.method_used in ("match", "resize")   # may fallback on synthetic
    print(f"  align_feature_match() | method_used={r.method_used} "
          f"fallback={r.fallback} ✓")

    # ── match — low-texture fallback ──────────────────────────────────────
    flat_donor   = np.full((512, 512, 3), 128, dtype=np.uint8)
    flat_content = np.full((512, 512, 3), 100, dtype=np.uint8)
    r = align_feature_match(flat_donor, flat_content)
    assert r.aligned_donor.shape == (512, 512, 3)
    assert r.fallback   # must fallback on flat images
    print(f"  align_feature_match() [flat → fallback] | "
          f"fallback={r.fallback} method_used={r.method_used} ✓")

    # ── flow ──────────────────────────────────────────────────────────────
    r = align_flow(donor_same, content)
    assert r.aligned_donor.shape == (512, 512, 3)
    assert r.method_used in ("flow", "resize")
    print(f"  align_flow()    | method_used={r.method_used} "
          f"mean_flow={r.meta.get('mean_flow_magnitude', 'N/A')} ✓")

    # ── affine ────────────────────────────────────────────────────────────
    src_pts = np.array([[50, 50], [450, 50], [450, 450], [50, 450]], dtype=np.float32)
    dst_pts = np.array([[60, 60], [460, 60], [460, 460], [60, 460]], dtype=np.float32)
    r = align_affine(donor_same, target_hw, src_pts, dst_pts)
    assert r.aligned_donor.shape == (512, 512, 3)
    assert r.method_used == "affine"
    print(f"  align_affine()  | scale={r.meta['scale']:.3f} "
          f"rot={r.meta['rotation_deg']:.1f}° ✓")

    # ── affine ValueError: too few points ─────────────────────────────────
    try:
        align_affine(donor_same, target_hw,
                     np.array([[0, 0]], dtype=np.float32),
                     np.array([[0, 0]], dtype=np.float32))
        assert False, "should have raised ValueError"
    except ValueError:
        print(f"  align_affine()  [< 3 pts → ValueError] ✓")

    # ── run_donor_alignment dispatcher ───────────────────────────────────
    for method in ("resize", "tile", "match", "flow"):
        r = run_donor_alignment(donor_same, content, method=method)
        assert r.aligned_donor.shape == (512, 512, 3), \
            f"run_donor_alignment method={method} shape wrong"
        print(f"  run_donor_alignment(method={method}) | "
              f"used={r.method_used} fallback={r.fallback} ✓")

    # affine via run_donor_alignment
    r = run_donor_alignment(
        donor_same, content, method="affine",
        src_points=src_pts, dst_points=dst_pts,
    )
    assert r.method_used == "affine"
    print(f"  run_donor_alignment(method=affine) | used={r.method_used} ✓")

    # unknown method ValueError
    try:
        run_donor_alignment(donor_same, content, method="unknown")
        assert False, "should have raised ValueError"
    except ValueError:
        print(f"  run_donor_alignment(method=unknown → ValueError) ✓")

    # ── build_convex_hull_mask ────────────────────────────────────────────
    pts  = np.array([[100, 100], [400, 100], [400, 400], [100, 400]],
                    dtype=np.float32)
    mask = build_convex_hull_mask(pts, image_hw=(512, 512), dilation_px=10)
    assert mask.shape == (512, 512)
    assert mask.dtype == np.uint8
    assert mask.max() == 255
    assert mask.min() == 0
    print(f"  build_convex_hull_mask() | shape={mask.shape} "
          f"coverage={(mask>0).mean()*100:.1f}% ✓")

    # ── output spatial sizes are always exactly content size ──────────────
    for method in ("resize", "tile"):
        for donor in (donor_small, donor_large, donor_same):
            r = run_donor_alignment(donor, content, method=method)
            assert r.aligned_donor.shape[:2] == (512, 512), \
                f"Output size mismatch: method={method}"
    print(f"  output size consistency across all donor sizes ✓")

    print("\n[alignment] All smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()