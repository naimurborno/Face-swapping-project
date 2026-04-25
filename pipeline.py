# pipeline.py
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from patch_unet import patch_unet_attention
from kv_cache import KVCache


@torch.no_grad()
def run_face_swap(
    pipe,
    source_image,
    reference_image,
    face_mask,              # torch.Tensor (1,1,H,W) float [0,1]
    prompt="",
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42,
):
    device = pipe.device

    # ── Patch UNet once ──────────────────────────────────────────────────────
    patch_unet_attention(pipe.unet)

    kv_cache = KVCache()
    kv_cache.face_mask = face_mask.to(device)

    # ── Encode images to latents ─────────────────────────────────────────────
    def encode(img):
        t = pipe.image_processor.preprocess(img).to(device, dtype=pipe.vae.dtype)
        lat = pipe.vae.encode(t).latent_dist.sample()
        return lat * pipe.vae.config.scaling_factor

    src_lat = encode(source_image)
    ref_lat = encode(reference_image)

    # ── Prompt embeddings ────────────────────────────────────────────────────
    do_cfg = guidance_scale > 1.0
    prompt_embeds, neg_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt="",
    )
    # For CFG: stack [neg, pos]
    text_embeds = torch.cat([neg_embeds, prompt_embeds]) if do_cfg else prompt_embeds

    # ── Scheduler ────────────────────────────────────────────────────────────
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # ── Init noisy source latent ─────────────────────────────────────────────
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn_like(src_lat, generator=gen)
    latents = pipe.scheduler.add_noise(src_lat, noise, timesteps[:1])

    # ── Denoising loop ───────────────────────────────────────────────────────
    for t in timesteps:

        # ── Pass 1: Reference → store K,V ───────────────────────────────────
        ref_noise = torch.randn_like(ref_lat, generator=gen)
        noisy_ref = pipe.scheduler.add_noise(ref_lat, ref_noise, t.unsqueeze(0))

        kv_cache.set_mode("store")
        kv_cache.clear()

        ref_input = torch.cat([noisy_ref] * 2) if do_cfg else noisy_ref
        pipe.unet(
            ref_input, t,
            encoder_hidden_states=text_embeds if do_cfg else prompt_embeds,
            cross_attention_kwargs={"kv_cache": kv_cache},
        )

        # ── Pass 2: Source → inject K,V → predict noise ──────────────────────
        kv_cache.set_mode("inject")

        src_input = torch.cat([latents] * 2) if do_cfg else latents
        noise_pred = pipe.unet(
            src_input, t,
            encoder_hidden_states=text_embeds if do_cfg else prompt_embeds,
            cross_attention_kwargs={"kv_cache": kv_cache},
        ).sample

        # CFG
        if do_cfg:
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (cond - uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # ── Decode ───────────────────────────────────────────────────────────────
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents).sample
    return pipe.image_processor.postprocess(image, output_type="pil")[0]