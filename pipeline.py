# pipeline.py
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from patch_unet import patch_unet_attention
from kv_cache import KVCache
from freq_utils import encode_freq_latents


@torch.no_grad()
def run_face_swap(
    pipe,
    source_image,
    reference_image,
    face_mask,                    # torch.Tensor (1,1,H,W) float [0,1]
    prompt="",
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42,
    cutoff_ratio=0.1,             # LF/HF split threshold (see freq_utils)
    injection_scale=1.0,          # Global multiplier on both lambdas — tune [0.5, 1.0]
    mode="phase3",                # "phase2" = naive uniform, "phase3" = freq-decomposed
):
    """
    Phase 3 face swap pipeline.

    Denoising loop (per step):
        Pass 1 — store LF:  noisy LF reference → U-Net (mode=store, freq=lf)
        Pass 2 — store HF:  noisy HF reference → U-Net (mode=store, freq=hf)
        Pass 3 — inject:    noisy source latent → U-Net (mode=inject)
                            deep layers   ← LF cache × λ_LF(t)
                            shallow layers← HF cache × λ_HF(t)

    Phase 2 fallback (mode="phase2"):
        Stores full reference (no freq split) and injects uniformly.
        Use this as the ablation baseline without changing run.py.

    Args:
        injection_scale: Scales both lambdas. Below 0.8 is safer if you see
                         identity bleed artifacts at shallow layers.
    """
    device = pipe.device

    # ── Patch UNet (idempotent: skip if already patched) ─────────────────────
    from kv_attention import KVInjectionAttention
    already_patched = any(
        isinstance(m, KVInjectionAttention)
        for m in pipe.unet.modules()
    )
    if not already_patched:
        patch_unet_attention(pipe.unet)

    kv_cache = KVCache()
    kv_cache.face_mask = face_mask.to(device)

    # ── Encode images to latents ──────────────────────────────────────────────
    def encode(img):
        t = pipe.image_processor.preprocess(img).to(device, dtype=pipe.vae.dtype)
        lat = pipe.vae.encode(t).latent_dist.sample()
        return lat * pipe.vae.config.scaling_factor

    src_lat = encode(source_image)

    if mode == "phase3":
        # Preprocess reference → pixel tensor for FFT
        ref_pixel = pipe.image_processor.preprocess(reference_image).to(device, dtype=pipe.vae.dtype)
        lf_lat, hf_lat = encode_freq_latents(pipe.vae, ref_pixel, cutoff_ratio)
    else:
        # Phase 2 baseline: single reference encoding
        ref_lat = encode(reference_image)

    # ── Prompt embeddings ─────────────────────────────────────────────────────
    do_cfg = guidance_scale > 1.0
    prompt_embeds, neg_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt="",
    )
    text_embeds = torch.cat([neg_embeds, prompt_embeds]) if do_cfg else prompt_embeds
    ref_embeds  = neg_embeds if do_cfg else prompt_embeds   # uncond for ref encoding

    # ── Scheduler ────────────────────────────────────────────────────────────
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    T = len(timesteps)

    # ── Init noisy source latent ──────────────────────────────────────────────
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn_like(src_lat, generator=gen)
    latents = pipe.scheduler.add_noise(src_lat, noise, timesteps[:1])

    # ── Denoising loop ────────────────────────────────────────────────────────
    for step_idx, t in enumerate(timesteps):

        # Update temporal annealing lambdas (both scaled by injection_scale)
        kv_cache.set_lambdas(step_idx, T)
        kv_cache.lambda_lf *= injection_scale
        kv_cache.lambda_hf *= injection_scale

        kv_cache.clear()

        if mode == "phase3":
            # ── Pass 1: Noisy LF reference → store LF K,V ────────────────────
            lf_noise = torch.randn_like(lf_lat, generator=gen)
            noisy_lf  = pipe.scheduler.add_noise(lf_lat, lf_noise, t.unsqueeze(0))

            kv_cache.set_mode("store")
            kv_cache.set_freq_mode("lf")
            _unet_pass(pipe, noisy_lf, t, ref_embeds, kv_cache, do_cfg=False)

            # ── Pass 2: Noisy HF reference → store HF K,V ────────────────────
            hf_noise = torch.randn_like(hf_lat, generator=gen)
            noisy_hf  = pipe.scheduler.add_noise(hf_lat, hf_noise, t.unsqueeze(0))

            kv_cache.set_mode("store")
            kv_cache.set_freq_mode("hf")
            _unet_pass(pipe, noisy_hf, t, ref_embeds, kv_cache, do_cfg=False)

        else:
            # ── Phase 2 baseline: single reference store ──────────────────────
            ref_noise = torch.randn_like(ref_lat, generator=gen)
            noisy_ref = pipe.scheduler.add_noise(ref_lat, ref_noise, t.unsqueeze(0))

            kv_cache.set_mode("store")
            kv_cache.set_freq_mode("lf")   # lf slot used as the single store
            _unet_pass(pipe, noisy_ref, t, ref_embeds, kv_cache, do_cfg=False)

            # Mirror to hf slot so inject code finds something regardless of depth
            kv_cache._hf_cache = {k: v for k, v in kv_cache._lf_cache.items()}
            # Override lambdas: uniform (phase 2 has no temporal annealing)
            kv_cache.lambda_lf = injection_scale
            kv_cache.lambda_hf = injection_scale

        # ── Pass 3: Source → inject → predict noise ───────────────────────────
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

        torch.cuda.empty_cache()

    # ── Decode ────────────────────────────────────────────────────────────────
    latents = latents / pipe.vae.config.scaling_factor
    image   = pipe.vae.decode(latents).sample
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


# ── Internal helper ───────────────────────────────────────────────────────────

def _unet_pass(pipe, latent, t, text_embeds, kv_cache, do_cfg: bool):
    """
    Single U-Net forward pass for store phases.
    Always run without CFG (reference has no classifier guidance need).
    The result is discarded — we only care about the K,V side-effect.
    """
    pipe.unet(
        latent, t,
        encoder_hidden_states=text_embeds,
        cross_attention_kwargs={"kv_cache": kv_cache},
    )