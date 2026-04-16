from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler
from das.diffusers_patch.pipeline_using_SMC_SDXL import pipeline_using_smc_sdxl
import torch
import numpy as np
import das.rewards as rewards
from PIL import Image
import os

################### Configuration ###################
kl_coeff = 0.0001
n_steps = 100
num_particles = 4
batch_p = 1
tempering_gamma = 0.008

prompt = "A close up of a handpalm with leaves growing from it."
repeated_prompts = [prompt] * batch_p

# Use a differentiable reward for steering during denoising.
steer_reward_fn = rewards.PickScore(device="cuda")
def image_reward_fn(images):
    scores = steer_reward_fn(images, repeated_prompts)
    if isinstance(scores, tuple):
        scores = scores[-1]
    return torch.as_tensor(scores)

# Use ImageReward only for evaluation on decoded timestep images.
eval_reward_fn = rewards.ImageReward(device="cuda")


def eval_image_reward(images, prompts):
    scores = eval_reward_fn(images, prompts)
    if isinstance(scores, tuple):
        scores = scores[-1]
    return torch.as_tensor(scores)


def decode_latents_sdxl(pipe, latents):
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast

    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != pipe.vae.dtype:
        if torch.backends.mps.is_available():
            pipe.vae = pipe.vae.to(latents.dtype)
        else:
            latents = latents.to(pipe.vae.dtype)

    has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None

    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipe.vae.config.scaling_factor

    with torch.no_grad():
        image = pipe.vae.decode(latents).sample

    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)

    do_denormalize = [True] * image.shape[0]
    image = pipe.image_processor.postprocess(image, output_type="pt", do_denormalize=do_denormalize)
    return image

################### Initialize ###################
log_dir_sdxl_smc = "logs/DAS_SDXL/pick/qualitative"
os.makedirs(log_dir_sdxl_smc, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(n_steps)
pipe.to("cuda")
pipe.vae.to(dtype=torch.float32)
pipe.text_encoder.to(dtype=torch.float32)

################### Inference ###################
image, log_w, normalized_w, all_latents, all_log_w, all_resample_indices, ess_trace, scale_factor_trace, rewards_trace, manifold_deviation_trace, log_prob_diffusion_trace = pipeline_using_smc_sdxl(
    pipe,
    prompt=prompt,
    negative_prompt="",
    num_inference_steps=n_steps,
    output_type="pt",
    # SMC parameters
    num_particles=num_particles,
    batch_p=batch_p,
    tempering_gamma=tempering_gamma,
    reward_fn=image_reward_fn,
    kl_coeff=kl_coeff,
    show_intermediate_rewards=True,
)

# Evaluate decoded timestep latents using ImageReward.
image_reward_max_trace = []
image_reward_mean_trace = []
for step_idx, step_latents in enumerate(all_latents):
    step_latents = step_latents.to("cuda")
    step_images = decode_latents_sdxl(pipe, step_latents)
    step_prompts = [prompt] * step_images.shape[0]
    step_scores = eval_image_reward(step_images, step_prompts).detach().cpu()

    image_reward_max_trace.append(step_scores.max().item())
    image_reward_mean_trace.append(step_scores.mean().item())

    print(
        f"[ImageReward][step {step_idx+1:03d}] "
        f"scores={step_scores.tolist()} max={step_scores.max().item():.6f} mean={step_scores.mean().item():.6f}"
    )

# Final image metrics
steer_reward = image_reward_fn(image).item()
final_image_reward = eval_image_reward(image, [prompt]).item()

np.save(f"{log_dir_sdxl_smc}/image_reward_max_trace.npy", np.array(image_reward_max_trace))
np.save(f"{log_dir_sdxl_smc}/image_reward_mean_trace.npy", np.array(image_reward_mean_trace))

image = (image[0].cpu().numpy() * 255).transpose(1, 2, 0).round().astype(np.uint8)
image = Image.fromarray(image)
image.save(
    f"{log_dir_sdxl_smc}/{prompt} | PickScore: {steer_reward:.6f} | ImageReward: {final_image_reward:.6f}.png"
)
print(
    f"Saved in {log_dir_sdxl_smc}/{prompt} | PickScore: {steer_reward:.6f} | ImageReward: {final_image_reward:.6f}.png"
)