# run.py
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from pipeline import run_face_swap
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# ── 1. Load model ──────────────────────────────────────────────────────────────
model_id = "stabilityai/stable-diffusion-2-1"   # or 2-base for 512px
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# ── 2. Load images ─────────────────────────────────────────────────────────────
source_image    = Image.open("source.jpg").resize((768, 768))
reference_image = Image.open("reference.jpg").resize((768, 768))

# ── 3. Build face mask ─────────────────────────────────────────────────────────
# Option A: simple bounding box mask (replace with MediaPipe/dlib later)
import torch
mask = torch.zeros(1, 1, 768, 768)
# Example: face occupies rows 150-600, cols 200-570
mask[:, :, 150:600, 200:570] = 1.0

# Option B (recommended): use MediaPipe face detection
# pip install mediapipe
# from mediapipe_mask import get_face_mask
# mask = get_face_mask(source_image)  # returns (1,1,768,768) tensor

# ── 4. Run ─────────────────────────────────────────────────────────────────────
result = run_face_swap(
    pipe=pipe,
    source_image=source_image,
    reference_image=reference_image,
    face_mask=mask,
    prompt="a portrait photo, high quality",
    num_inference_steps=30,       # 30 steps fine for baseline
    guidance_scale=7.5,
    seed=42,
)

# ── 5. Side-by-side comparison ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(source_image);    axes[0].set_title("Source");    axes[0].axis("off")
axes[1].imshow(result);          axes[1].set_title("Output");    axes[1].axis("off")
axes[2].imshow(reference_image); axes[2].set_title("Reference"); axes[2].axis("off")
plt.tight_layout()
plt.savefig("phase2_result.png", dpi=150)
plt.show()
print("Saved to phase2_result.png")