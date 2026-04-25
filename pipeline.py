# pipeline.py
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from attention_processor import KVInjectionProcessor
from kv_cache import KVCache


def install_processors(unet, kv_cache: KVCache):
    """
    Replace every self-attention (attn1) processor in the U-Net
    with our KVInjectionProcessor, keyed by layer name.
    """
    for name, module in unet.named_modules():
        # In diffusers SD2, self-attention is attn1
        # Cross-attention is attn2 — we leave that alone
        if name.endswith(".attn1"):
            processor = KVInjectionProcessor(
                layer_name=name,
                kv_cache=kv_cache,
            )
            module.set_processor(processor)
    print(f"[INFO] Installed KVInjectionProcessor on all attn1 layers.")


@torch.no_grad()
def run_face_swap(
    pipe,                       # loaded StableDiffusionPipeline
    source_image,               # PIL Image
    reference_image,            # PIL Image
    face_mask,                  # torch.Tensor (1,1,H,W) float [0,1]
    prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
):
    device = pipe.device
    kv_cache = KVCache()
    kv_cache.face_mask = face_mask.to(device)

    # Install our processors once
    install_processors(pipe.unet, kv_cache)

    # Encode both images to latents
    def encode_image(img):
        img_tensor = pipe.image_processor.preprocess(img).to(device, dtype=pipe.vae.dtype)
        latent = pipe.vae.encode(img_tensor).latent_dist.sample()
        return latent * pipe.vae.config.scaling_factor

    src_latent = encode_image(source_image)
    ref_latent = encode_image(reference_image)

    # Set up scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Encode prompt (can be empty string for unconditional)
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=guidance_scale > 1.0,
        negative_prompt="",
    )

    # Initialize noisy latent from source
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn_like(src_latent, generator=generator)
    # Start denoising from t_start — use full schedule
    noisy_src = pipe.scheduler.add_noise(
        src_latent, noise, timesteps[:1]
    )
    latents = noisy_src

    # Denoising loop
    for i, t in enumerate(timesteps):
        # ── REFERENCE PASS: store K,V from reference at this timestep ──
        noisy_ref = pipe.scheduler.add_noise(
            ref_latent, torch.randn_like(ref_latent, generator=generator), t.unsqueeze(0)
        )
        kv_cache.set_mode("store")
        kv_cache.clear()

        ref_input = torch.cat([noisy_ref] * 2) if guidance_scale > 1.0 else noisy_ref
        _ = pipe.unet(
            ref_input,
            t,
            encoder_hidden_states=prompt_embeds if guidance_scale > 1.0
                                  else negative_prompt_embeds,
        ).sample

        # ── SOURCE PASS: inject stored K,V during source denoising ──
        kv_cache.set_mode("inject")

        latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        noise_pred = pipe.unet(
            latent_input,
            t,
            encoder_hidden_states=prompt_embeds if guidance_scale > 1.0
                                  else negative_prompt_embeds,
        ).sample

        # CFG
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode final latent
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    return image