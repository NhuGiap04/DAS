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
    prompt_slug = slugify(prompt)
    prompt_dir = os.path.join(output_dir, prompt_slug)
    os.makedirs(prompt_dir, exist_ok=True)
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
    image, log_w, normalized_w, all_latents, all_log_w, all_resample_indices, ess_trace, scale_factor_trace, rewards_trace, manifold_deviation_trace, log_prob_diffusion_trace, all_after_steer_latents = pipeline_using_smc_sdxl(
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
        return_after_steer_latents=True,
    )


    # Evaluate decoded timestep latents before and after steer for both rewards.
    eval_reward_before_max_trace = []
    eval_reward_before_mean_trace = []
    eval_reward_after_max_trace = []
    eval_reward_after_mean_trace = []
    steer_reward_before_max_trace = []
    steer_reward_before_mean_trace = []
    steer_reward_after_max_trace = []
    steer_reward_after_mean_trace = []
    step_logs = []
    for step_idx, (step_latents_before, step_latents_after) in enumerate(zip(all_latents, all_after_steer_latents)):
        step_latents_before = step_latents_before.to("cuda")
        step_latents_after = step_latents_after.to("cuda")
        step_images_before = decode_latents_sdxl(pipe, step_latents_before)
        step_images_after = decode_latents_sdxl(pipe, step_latents_after)
        step_prompts_before = [prompt] * step_images_before.shape[0]
        step_prompts_after = [prompt] * step_images_after.shape[0]

        # Eval Reward before/after steer
        eval_scores_before = eval_image_reward(step_images_before, step_prompts_before).detach().cpu()
        eval_scores_after = eval_image_reward(step_images_after, step_prompts_after).detach().cpu()
        eval_reward_before_max_trace.append(eval_scores_before.max().item())
        eval_reward_before_mean_trace.append(eval_scores_before.mean().item())
        eval_reward_after_max_trace.append(eval_scores_after.max().item())
        eval_reward_after_mean_trace.append(eval_scores_after.mean().item())

        # Steer Reward before/after steer
        steer_scores_before = image_reward_fn(step_images_before).detach().cpu()
        steer_scores_after = image_reward_fn(step_images_after).detach().cpu()
        steer_reward_before_max_trace.append(steer_scores_before.max().item())
        steer_reward_before_mean_trace.append(steer_scores_before.mean().item())
        steer_reward_after_max_trace.append(steer_scores_after.max().item())
        steer_reward_after_mean_trace.append(steer_scores_after.mean().item())

        step_logs.append({
            "step": step_idx+1,
            "eval_reward_before_scores": eval_scores_before.tolist(),
            "eval_reward_before_max": eval_scores_before.max().item(),
            "eval_reward_before_mean": eval_scores_before.mean().item(),
            "eval_reward_after_scores": eval_scores_after.tolist(),
            "eval_reward_after_max": eval_scores_after.max().item(),
            "eval_reward_after_mean": eval_scores_after.mean().item(),
            "steer_reward_before_scores": steer_scores_before.tolist(),
            "steer_reward_before_max": steer_scores_before.max().item(),
            "steer_reward_before_mean": steer_scores_before.mean().item(),
            "steer_reward_after_scores": steer_scores_after.tolist(),
            "steer_reward_after_max": steer_scores_after.max().item(),
            "steer_reward_after_mean": steer_scores_after.mean().item(),
        })

    # Final image metrics
    steer_reward = image_reward_fn(image).item()
    final_image_reward = eval_image_reward(image, [prompt]).item()



    # Save traces as .npy
    np.save(os.path.join(prompt_dir, "eval_reward_before_max_trace.npy"), np.array(eval_reward_before_max_trace))
    np.save(os.path.join(prompt_dir, "eval_reward_before_mean_trace.npy"), np.array(eval_reward_before_mean_trace))
    np.save(os.path.join(prompt_dir, "eval_reward_after_max_trace.npy"), np.array(eval_reward_after_max_trace))
    np.save(os.path.join(prompt_dir, "eval_reward_after_mean_trace.npy"), np.array(eval_reward_after_mean_trace))
    np.save(os.path.join(prompt_dir, "steer_reward_before_max_trace.npy"), np.array(steer_reward_before_max_trace))
    np.save(os.path.join(prompt_dir, "steer_reward_before_mean_trace.npy"), np.array(steer_reward_before_mean_trace))
    np.save(os.path.join(prompt_dir, "steer_reward_after_max_trace.npy"), np.array(steer_reward_after_max_trace))
    np.save(os.path.join(prompt_dir, "steer_reward_after_mean_trace.npy"), np.array(steer_reward_after_mean_trace))

    # Backward-compatible aliases for existing artifact collectors.
    np.save(os.path.join(prompt_dir, "image_reward_max_trace.npy"), np.array(eval_reward_before_max_trace))
    np.save(os.path.join(prompt_dir, "image_reward_mean_trace.npy"), np.array(eval_reward_before_mean_trace))
    np.save(os.path.join(prompt_dir, "pickscore_max_trace.npy"), np.array(steer_reward_before_max_trace))
    np.save(os.path.join(prompt_dir, "pickscore_mean_trace.npy"), np.array(steer_reward_before_mean_trace))

    # Plot 1: Eval Reward (Before/After Steer), mean and max.
    plt.figure(figsize=(8, 5))
    plt.plot(eval_reward_before_max_trace, label='Before Max', marker='o')
    plt.plot(eval_reward_before_mean_trace, label='Before Mean', marker='x')
    plt.plot(eval_reward_after_max_trace, label='After Max', marker='s')
    plt.plot(eval_reward_after_mean_trace, label='After Mean', marker='^')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Eval Reward: Before vs After Steer')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plot_path_eval_before_after = os.path.join(prompt_dir, "reward_trace_eval_before_after.png")
    plt.tight_layout()
    plt.savefig(plot_path_eval_before_after)
    plt.close()

    # Plot 2: Steer Reward (Before/After Steer), mean and max.
    plt.figure(figsize=(8, 5))
    plt.plot(steer_reward_before_max_trace, label='Before Max', marker='o', color='tab:orange')
    plt.plot(steer_reward_before_mean_trace, label='Before Mean', marker='x', color='tab:green')
    plt.plot(steer_reward_after_max_trace, label='After Max', marker='s', color='tab:red')
    plt.plot(steer_reward_after_mean_trace, label='After Mean', marker='^', color='tab:blue')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Steer Reward: Before vs After Steer')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plot_path_steer_before_after = os.path.join(prompt_dir, "reward_trace_steer_before_after.png")
    plt.tight_layout()
    plt.savefig(plot_path_steer_before_after)
    plt.close()

    # Save step logs as JSON if requested
    if args.log_json:
        log_json_parent = os.path.dirname(args.log_json)
        if log_json_parent:
            os.makedirs(log_json_parent, exist_ok=True)
        with open(args.log_json, 'w') as f:
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
    print(f"Eval reward trace plot (before/after steer) saved to: {plot_path_eval_before_after}")
    print(f"Steer reward trace plot (before/after steer) saved to: {plot_path_steer_before_after}")
    print(f"All outputs saved in: {prompt_dir}")


if __name__ == "__main__":
    main()
