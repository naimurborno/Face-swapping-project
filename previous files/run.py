# run.py
"""
Full pipeline orchestration:
    Phase 1  →  Bridge  →  Phase 3 injection pipeline

Execution order:
    1. Load images (OpenCV BGR — Phase 1 expects this)
    2. Run Phase 1: landmark detection, affine alignment, Gaussian LF/HF split
    3. Bridge: convert Phase 1 outputs to pipeline-ready PIL images + mask tensor
    4. Run pipeline in phase2 mode (ablation baseline)
    5. Run pipeline in phase3 mode (freq-decomposed injection)
    6. Display comparison
"""

import cv2
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Phase 1
from phase1 import align_and_decompose

# Bridge + pipeline
from phase_1 import phase1_to_pipeline
from pipeline import run_face_swap

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "stabilityai/stable-diffusion-2-1-base"
TARGET_SIZE     = 512       # 512 for sd-2-1-base, 768 for sd-2-1
SEED            = 42
STEPS           = 25
GUIDANCE        = 7.5
INJECTION_SCALE = 0.8       # Start here. Sweep [0.5, 1.0] for ablation.
GAUSS_KERNEL    = 31        # Phase 1 Gaussian kernel. Odd number. Larger = smoother LF.
GAUSS_SIGMA     = 5.0       # Phase 1 Gaussian sigma. Larger = more LF bleed into HF.

# ── 1. Load images (BGR for OpenCV / Phase 1) ─────────────────────────────────
source_bgr    = cv2.imread("source.jpg")
reference_bgr = cv2.imread("reference.jpg")

if source_bgr is None or reference_bgr is None:
    raise FileNotFoundError("source.jpg or reference.jpg not found. Check paths.")

# ── 2. Phase 1: landmark detection + alignment + decomposition ────────────────
LANDMARK_MODEL = "face_landmarker.task"   # path to downloaded .task file

print("── Phase 1: Face alignment + frequency decomposition ──")
phase1_result = align_and_decompose(
    source_img    = source_bgr,
    reference_img = reference_bgr,
    gauss_kernel  = GAUSS_KERNEL,
    gauss_sigma   = GAUSS_SIGMA,
    visualize     = True,           # Set False to skip the sanity check plot
    model_path    = LANDMARK_MODEL,
)

print(f"\nPhase 1 complete:")
print(f"  yaw_diff      : {phase1_result.yaw_diff:.1f}°  {'⚠ large' if phase1_result.yaw_diff > 35 else '✓ ok'}")
print(f"  LF shape      : {phase1_result.LF.shape}  dtype={phase1_result.LF.dtype}")
print(f"  HF std        : {phase1_result.HF.std():.2f}  (expect > 10 for textured face)")
print(f"  mask coverage : {(phase1_result.face_mask > 0).mean()*100:.1f}% of pixels")

# ── 3. Bridge: numpy BGR arrays → PIL RGB images + mask tensor ────────────────
print("\n── Bridge: converting Phase 1 outputs → pipeline inputs ──")
inputs = phase1_to_pipeline(
    source_bgr    = source_bgr,
    phase1_result = phase1_result,
    target_size   = TARGET_SIZE,
)

# ── 4. Load SD model ──────────────────────────────────────────────────────────
print("\n── Loading Stable Diffusion ──")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.unet.eval()

PROMPT = "portrait photo, sharp focus, photorealistic"

# ── 5. Phase 2 baseline (naive uniform injection, aligned reference, no freq split) ──
print("\n── Running Phase 2 baseline ──")
result_p2 = run_face_swap(
    pipe              = pipe,
    source_image      = inputs.source_pil,
    face_mask         = inputs.face_mask,
    prompt            = PROMPT,
    num_inference_steps = STEPS,
    guidance_scale    = GUIDANCE,
    seed              = SEED,
    injection_scale   = INJECTION_SCALE,
    mode              = "phase2",
    aligned_image     = inputs.aligned_pil,   # whole aligned reference, no split
)

# ── 6. Phase 3 (freq-decomposed, depth-aware, temporally-annealed injection) ──
print("\n── Running Phase 3 (freq-decomposed) ──")
result_p3 = run_face_swap(
    pipe              = pipe,
    source_image      = inputs.source_pil,
    face_mask         = inputs.face_mask,
    prompt            = PROMPT,
    num_inference_steps = STEPS,
    guidance_scale    = GUIDANCE,
    seed              = SEED,
    injection_scale   = INJECTION_SCALE,
    mode              = "phase3",
    lf_image          = inputs.lf_pil,        # Gaussian LF → deep layers
    hf_image          = inputs.hf_pil,        # Gaussian HF → shallow layers
)

# ── 7. Display ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
panels = [
    (inputs.source_pil,   "Source"),
    (inputs.aligned_pil,  f"Aligned Ref\n(yaw Δ={phase1_result.yaw_diff:.1f}°)"),
    (inputs.lf_pil,       "LF (structure)"),
    (result_p2,           "Phase 2\n(naive injection)"),
    (result_p3,           f"Phase 3\n(freq-decomposed, scale={INJECTION_SCALE})"),
]
for ax, (img, title) in zip(axes, panels):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("phase3_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → phase3_comparison.png")