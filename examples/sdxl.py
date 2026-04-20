import re
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler
from das.diffusers_patch.pipeline_using_SMC_SDXL import pipeline_using_smc_sdxl
import torch
import numpy as np
import das.rewards as rewards
from PIL import Image
import os

import argparse
import json
import matplotlib.pyplot as plt

################### Configuration ###################
prompt = "A close up of a handpalm with leaves growing from it."

def main():
        def slugify(text, max_len=60):
            slug = re.sub(r'[^a-zA-Z0-9]+', '-', text.strip().lower())
            slug = slug.strip('-')
            return slug[:max_len].rstrip('-') or 'prompt'

        prompt_slug = slugify(prompt)
        prompt_dir = os.path.join(output_dir, prompt_slug)
        os.makedirs(prompt_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description="Run SDXL example with a given prompt.")
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for image generation')
    parser.add_argument('--log_json', type=str, default=None, help='Path to save intermediate rewards log as JSON')
    parser.add_argument('--output_dir', type=str, default="logs/DAS_SDXL/pick/qualitative", help='Output directory')
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--num_particles', type=int, default=4)
    parser.add_argument('--batch_p', type=int, default=1)
    parser.add_argument('--kl_coeff', type=float, default=1.)
    parser.add_argument('--tempering_gamma', type=float, default=0.008)
    args = parser.parse_args()

    prompt = args.prompt
    batch_p = args.batch_p
    n_steps = args.n_steps
    num_particles = args.num_particles
    kl_coeff = args.kl_coeff
    tempering_gamma = args.tempering_gamma
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
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


    # Initialize pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(n_steps)
    pipe.to("cuda")
    pipe.vae.to(dtype=torch.float32)
    pipe.text_encoder.to(dtype=torch.float32)

    # Inference
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


    # Evaluate decoded timestep latents using ImageReward (after steer) and PickScore (before steer)
    image_reward_max_trace = []
    image_reward_mean_trace = []
    pickscore_max_trace = []
    pickscore_mean_trace = []
    step_logs = []
    for step_idx, step_latents in enumerate(all_latents):
        step_latents = step_latents.to("cuda")
        step_images = decode_latents_sdxl(pipe, step_latents)
        step_prompts = [prompt] * step_images.shape[0]
        # After steer: ImageReward
        step_scores = eval_image_reward(step_images, step_prompts).detach().cpu()
        image_reward_max_trace.append(step_scores.max().item())
        image_reward_mean_trace.append(step_scores.mean().item())
        # Before steer: PickScore
        pick_scores = image_reward_fn(step_images).detach().cpu()
        pickscore_max_trace.append(pick_scores.max().item())
        pickscore_mean_trace.append(pick_scores.mean().item())
        step_logs.append({
            "step": step_idx+1,
            "image_reward_scores": step_scores.tolist(),
            "image_reward_max": step_scores.max().item(),
            "image_reward_mean": step_scores.mean().item(),
            "pickscore_scores": pick_scores.tolist(),
            "pickscore_max": pick_scores.max().item(),
            "pickscore_mean": pick_scores.mean().item()
        })

    # Final image metrics
    steer_reward = image_reward_fn(image).item()
    final_image_reward = eval_image_reward(image, [prompt]).item()



    # Save traces as .npy
    np.save(os.path.join(prompt_dir, "image_reward_max_trace.npy"), np.array(image_reward_max_trace))
    np.save(os.path.join(prompt_dir, "image_reward_mean_trace.npy"), np.array(image_reward_mean_trace))
    np.save(os.path.join(prompt_dir, "pickscore_max_trace.npy"), np.array(pickscore_max_trace))
    np.save(os.path.join(prompt_dir, "pickscore_mean_trace.npy"), np.array(pickscore_mean_trace))

    # Plot and save separate reward traces
    # 1. ImageReward (after steer)
    plt.figure(figsize=(8, 5))
    plt.plot(image_reward_max_trace, label='ImageReward Max', marker='o')
    plt.plot(image_reward_mean_trace, label='ImageReward Mean', marker='x')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('ImageReward (After Steer) at Each Timestep')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plot_path_after = os.path.join(prompt_dir, "reward_trace_after.png")
    plt.tight_layout()
    plt.savefig(plot_path_after)
    plt.close()

    # 2. PickScore (before steer)
    plt.figure(figsize=(8, 5))
    plt.plot(pickscore_max_trace, label='PickScore Max', marker='o', color='tab:orange')
    plt.plot(pickscore_mean_trace, label='PickScore Mean', marker='x', color='tab:green')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('PickScore (Before Steer) at Each Timestep')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plot_path_before = os.path.join(prompt_dir, "reward_trace_before.png")
    plt.tight_layout()
    plt.savefig(plot_path_before)
    plt.close()

    # Save step logs as JSON if requested
    if args.log_json:
        with open(os.path.join(prompt_dir, "intermediate_rewards.json"), 'w') as f:
            json.dump(step_logs, f, indent=2)

    # Save image
    image_np = (image[0].cpu().numpy() * 255).transpose(1, 2, 0).round().astype(np.uint8)
    image_pil = Image.fromarray(image_np)
    image_filename = os.path.join(prompt_dir, f"image_PickScore_{steer_reward:.6f}_ImageReward_{final_image_reward:.6f}.png")
    image_pil.save(image_filename)

    # Print only summary info
    print(f"Prompt: {prompt}")
    print(f"Saved image: {image_filename}")
    print(f"PickScore: {steer_reward:.6f} | ImageReward: {final_image_reward:.6f}")
    print(f"Reward trace plot (after steer) saved to: {plot_path_after}")
    print(f"Reward trace plot (before steer) saved to: {plot_path_before}")
    print(f"All outputs saved in: {prompt_dir}")


if __name__ == "__main__":
    main()