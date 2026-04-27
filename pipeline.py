# pipeline.py
import torch
from patch_unet import patch_unet_attention
from kv_cache import KVCache
from phase_1 import encode_pil_to_latent


@torch.no_grad()
def run_face_swap(
    pipe,
    source_image,               # PIL RGB — source pose/background
    face_mask,                  # torch.Tensor (1,1,H,W) float [0,1] from phase1_bridge
    prompt="",
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42,
    injection_scale=1.0,        # Global lambda multiplier. Start at 0.8, tune [0.5, 1.0]
    mode="phase3",              # "phase3" = freq-decomposed | "phase2" = naive baseline

    # Phase 3 inputs — from phase1_bridge.PipelineInputs
    lf_image=None,              # PIL RGB — Gaussian LF component of aligned reference
    hf_image=None,              # PIL RGB — Gaussian HF component (shifted to [0,255])

    # Phase 2 input — from phase1_bridge.PipelineInputs
    aligned_image=None,         # PIL RGB — aligned reference (whole, no freq split)
):
    """
    Face swap pipeline integrating Phase 1 alignment + Phase 3 freq-decomposed KV injection.

    Phase 1 does the heavy lifting before this function is called:
        - MediaPipe landmark detection
        - Affine alignment of reference → source pose
        - Gaussian LF/HF decomposition
        - Convex hull face mask

    This function handles:
        - VAE encoding of pre-aligned, pre-decomposed images
        - 3-pass denoising loop (store LF → store HF → inject)
        - Depth-aware, temporally-annealed KV injection

    Denoising loop per step:
        Pass 1 — store LF:   noisy LF reference → U-Net  (deep layers capture structure)
        Pass 2 — store HF:   noisy HF reference → U-Net  (shallow layers capture texture)
        Pass 3 — inject:     noisy source latent → U-Net
                             deep layers   ← LF cache × λ_LF(t)
                             shallow layers← HF cache × λ_HF(t)

    Phase 2 mode uses a single aligned reference (no freq split) for ablation comparison.
    """
    device = pipe.device

    # ── Validate inputs ───────────────────────────────────────────────────────
    if mode == "phase3" and (lf_image is None or hf_image is None):
        raise ValueError(
            "phase3 mode requires lf_image and hf_image. "
            "Run phase1_bridge.phase1_to_pipeline() first and pass result.lf_pil / result.hf_pil."
        )
    if mode == "phase2" and aligned_image is None:
        raise ValueError(
            "phase2 mode requires aligned_image. "
            "Pass result.aligned_pil from phase1_bridge.phase1_to_pipeline()."
        )

    # ── Patch UNet once (idempotent) ──────────────────────────────────────────
    from kv_attention import KVInjectionAttention
    already_patched = any(isinstance(m, KVInjectionAttention) for m in pipe.unet.modules())
    if not already_patched:
        patch_unet_attention(pipe.unet)

    kv_cache = KVCache()
    kv_cache.face_mask = face_mask.to(device)

    # ── Encode all images to latents ──────────────────────────────────────────
    # Source — what we're denoising
    src_lat = encode_pil_to_latent(pipe.vae, source_image, pipe.image_processor)

    if mode == "phase3":
        lf_lat = encode_pil_to_latent(pipe.vae, lf_image,  pipe.image_processor)
        hf_lat = encode_pil_to_latent(pipe.vae, hf_image,  pipe.image_processor)
    else:
        ref_lat = encode_pil_to_latent(pipe.vae, aligned_image, pipe.image_processor)

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
    # Reference store passes always use uncond — no guidance needed there
    ref_embeds  = neg_embeds if do_cfg else prompt_embeds

    # ── Scheduler ─────────────────────────────────────────────────────────────
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    T = len(timesteps)

    # ── Init noisy source latent ──────────────────────────────────────────────
    gen    = torch.Generator(device=device).manual_seed(seed)
    noise  = torch.randn_like(src_lat, generator=gen)
    latents = pipe.scheduler.add_noise(src_lat, noise, timesteps[:1])

    # ── Denoising loop ────────────────────────────────────────────────────────
    for step_idx, t in enumerate(timesteps):

        # Temporal annealing: λ_LF dominant early, λ_HF dominant late
        kv_cache.set_lambdas(step_idx, T)
        kv_cache.lambda_lf *= injection_scale
        kv_cache.lambda_hf *= injection_scale

        kv_cache.clear()

        if mode == "phase3":
            # Pass 1 — LF store: reference structure into deep layers
            lf_noise = torch.randn_like(lf_lat, generator=gen)
            noisy_lf = pipe.scheduler.add_noise(lf_lat, lf_noise, t.unsqueeze(0))
            kv_cache.set_mode("store")
            kv_cache.set_freq_mode("lf")
            _store_pass(pipe, noisy_lf, t, ref_embeds, kv_cache)

            # Pass 2 — HF store: reference texture into shallow layers
            hf_noise = torch.randn_like(hf_lat, generator=gen)
            noisy_hf = pipe.scheduler.add_noise(hf_lat, hf_noise, t.unsqueeze(0))
            kv_cache.set_mode("store")
            kv_cache.set_freq_mode("hf")
            _store_pass(pipe, noisy_hf, t, ref_embeds, kv_cache)

        else:
            # Phase 2 baseline: single whole-reference store, uniform injection
            ref_noise = torch.randn_like(ref_lat, generator=gen)
            noisy_ref = pipe.scheduler.add_noise(ref_lat, ref_noise, t.unsqueeze(0))
            kv_cache.set_mode("store")
            kv_cache.set_freq_mode("lf")
            _store_pass(pipe, noisy_ref, t, ref_embeds, kv_cache)

            # Mirror lf → hf so both depth categories find cached features
            kv_cache._hf_cache = dict(kv_cache._lf_cache)
            # Flat lambdas (no annealing in Phase 2)
            kv_cache.lambda_lf = injection_scale
            kv_cache.lambda_hf = injection_scale

        # Pass 3 — inject: source denoising step with reference KV blended in
        kv_cache.set_mode("inject")
        src_input  = torch.cat([latents] * 2) if do_cfg else latents
        noise_pred = pipe.unet(
            src_input, t,
            encoder_hidden_states=text_embeds if do_cfg else prompt_embeds,
            cross_attention_kwargs={"kv_cache": kv_cache},
        ).sample

        if do_cfg:
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (cond - uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        torch.cuda.empty_cache()

    # ── Decode ────────────────────────────────────────────────────────────────
    latents = latents / pipe.vae.config.scaling_factor
    image   = pipe.vae.decode(latents).sample
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


# ── Internal ──────────────────────────────────────────────────────────────────

def _store_pass(pipe, latent, t, ref_embeds, kv_cache):
    """
    Single no-grad U-Net forward pass for KV store phases.
    Output is discarded — side effect (cache population) is all we need.
    Always runs without CFG.
    """
    pipe.unet(
        latent, t,
        encoder_hidden_states=ref_embeds,
        cross_attention_kwargs={"kv_cache": kv_cache},
    )