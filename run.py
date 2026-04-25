# run.py
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from pipeline import run_face_swap
import matplotlib.pyplot as plt

# ── Load model ────────────────────────────────────────────────────────────────
model_id = "stabilityai/stable-diffusion-2-1-base"  # base = 512px, fits in 16GB
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.unet.eval()

# ── Images ────────────────────────────────────────────────────────────────────
SIZE = 512  # use 512 for sd-2-1-base; change to 768 for sd-2-1
source_image    = Image.open("source.jpg").convert("RGB").resize((SIZE, SIZE))
reference_image = Image.open("reference.jpg").convert("RGB").resize((SIZE, SIZE))

# ── Face mask — crude bounding box, replace with MediaPipe later ──────────────
mask = torch.zeros(1, 1, SIZE, SIZE)
# Adjust these coords to roughly cover the face in YOUR source image
mask[:, :, 100:400, 150:360] = 1.0

# ── Run ───────────────────────────────────────────────────────────────────────
result = run_face_swap(
    pipe=pipe,
    source_image=source_image,
    reference_image=reference_image,
    face_mask=mask,
    prompt="portrait photo, sharp focus",
    num_inference_steps=25,
    guidance_scale=7.5,
    seed=42,
)

# ── Show ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(source_image);    axes[0].set_title("Source");    axes[0].axis("off")
axes[1].imshow(result);          axes[1].set_title("Output");    axes[1].axis("off")
axes[2].imshow(reference_image); axes[2].set_title("Reference"); axes[2].axis("off")
plt.tight_layout()
plt.savefig("phase2_result.png", dpi=150)
plt.show()