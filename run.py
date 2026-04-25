# run.py
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from pipeline import run_face_swap
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "stabilityai/stable-diffusion-2-1-base"   # 512px
SIZE     = 512
SEED     = 42

# ── Load model ────────────────────────────────────────────────────────────────
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.unet.eval()

# ── Images ────────────────────────────────────────────────────────────────────
source_image    = Image.open("source.jpg").convert("RGB").resize((SIZE, SIZE))
reference_image = Image.open("reference.jpg").convert("RGB").resize((SIZE, SIZE))

# ── Face mask ─────────────────────────────────────────────────────────────────
# Replace with MediaPipe or SAM output for real experiments.
mask = torch.zeros(1, 1, SIZE, SIZE)
mask[:, :, 100:400, 150:360] = 1.0

# ── Run Phase 2 baseline ──────────────────────────────────────────────────────
result_p2 = run_face_swap(
    pipe=pipe,
    source_image=source_image,
    reference_image=reference_image,
    face_mask=mask,
    prompt="portrait photo, sharp focus",
    num_inference_steps=25,
    guidance_scale=7.5,
    seed=SEED,
    mode="phase2",                  # naive uniform injection
    injection_scale=1.0,
)

# ── Run Phase 3 (freq-decomposed) ────────────────────────────────────────────
result_p3 = run_face_swap(
    pipe=pipe,
    source_image=source_image,
    reference_image=reference_image,
    face_mask=mask,
    prompt="portrait photo, sharp focus",
    num_inference_steps=25,
    guidance_scale=7.5,
    seed=SEED,
    mode="phase3",                  # freq-decomposed depth-aware injection
    injection_scale=0.8,            # start conservative; sweep [0.5, 1.0]
    cutoff_ratio=0.1,               # LF/HF boundary; sweep [0.05, 0.15]
)

# ── Display ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(source_image);    axes[0].set_title("Source");         axes[0].axis("off")
axes[1].imshow(reference_image); axes[1].set_title("Reference");      axes[1].axis("off")
axes[2].imshow(result_p2);       axes[2].set_title("Phase 2 (naive)");axes[2].axis("off")
axes[3].imshow(result_p3);       axes[3].set_title("Phase 3 (freq)"); axes[3].axis("off")
plt.tight_layout()
plt.savefig("phase3_comparison.png", dpi=150)
plt.show()

print("Saved → phase3_comparison.png")