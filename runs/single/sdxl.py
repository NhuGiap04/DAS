import argparse
import json
import os
import re
import time

import das.rewards as rewards
import numpy as np
import torch
from das.diffusers_patch.pipeline_using_SMC_SDXL import pipeline_using_smc_sdxl
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from PIL import Image

################### Configuration ###################
prompt = "A close up of a handpalm with leaves growing from it."


def slugify(text, max_len=60):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    slug = slug.strip("-")
    return slug[:max_len].rstrip("-") or "prompt"


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_to_pil(image):
    image_np = (image.detach().cpu().numpy() * 255).transpose(1, 2, 0).round().astype(np.uint8)
    return Image.fromarray(image_np)


def main():
    parser = argparse.ArgumentParser(description="Run SDXL SMC for a single prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="logs/DAS_SDXL/pick/qualitative", help="Output directory")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--batch_p", type=int, default=1)
    parser.add_argument("--kl_coeff", type=float, default=1.0)
    parser.add_argument("--tempering_gamma", type=float, default=0.008)
    parser.add_argument("--save-final-artifacts", action="store_true", help="Save final particle images and rewards JSON.")
    args = parser.parse_args()

    prompt = args.prompt
    batch_p = args.batch_p
    n_steps = args.n_steps
    num_particles = args.num_particles
    kl_coeff = args.kl_coeff
    tempering_gamma = args.tempering_gamma

    ################### Rewards ###################
    config_reward_fn = rewards.PickScore(device="cuda")

    def config_image_reward_fn(images):
        prompt_batch = [prompt] * int(images.shape[0])
        scores = config_reward_fn(images, prompt_batch)
        if isinstance(scores, tuple):
            scores = scores[-1]
        return torch.as_tensor(scores)

    eval_reward_fn = rewards.ImageReward(device="cuda")

    def eval_image_reward_fn(images):
        prompt_batch = [prompt] * int(images.shape[0])
        scores = eval_reward_fn(images, prompt_batch)
        if isinstance(scores, tuple):
            scores = scores[-1]
        return torch.as_tensor(scores)

    ################### Initialize ###################
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(n_steps)
    pipe.to("cuda")
    pipe.vae.to(dtype=torch.float32)
    pipe.text_encoder.to(dtype=torch.float32)

    ################### Inference ###################
    cuda_sync()
    inference_start = time.perf_counter()
    (
        image,
        log_w,
        normalized_w,
        all_latents,
        all_log_w,
        all_resample_indices,
        ess_trace,
        scale_factor_trace,
        manifold_deviation_trace,
        log_prob_diffusion_trace,
        final_particle_images,
    ) = pipeline_using_smc_sdxl(
        pipe,
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=n_steps,
        output_type="pt",
        # SMC parameters
        num_particles=num_particles,
        batch_p=batch_p,
        tempering_gamma=tempering_gamma,
        reward_fn=config_image_reward_fn,
        kl_coeff=kl_coeff,
        return_final_particles=True,
    )
    cuda_sync()
    inference_seconds = time.perf_counter() - inference_start

    config_rewards = config_image_reward_fn(final_particle_images).detach().cpu().to(torch.float32)
    eval_rewards = eval_image_reward_fn(final_particle_images).detach().cpu().to(torch.float32)
    log_w_cpu = log_w.detach().cpu().to(torch.float32).flatten()
    normalized_w_cpu = normalized_w.detach().cpu().to(torch.float32).flatten()
    common_count = min(
        int(final_particle_images.shape[0]),
        int(config_rewards.numel()),
        int(eval_rewards.numel()),
        int(log_w_cpu.numel()),
        int(normalized_w_cpu.numel()),
    )
    if common_count <= 0:
        raise RuntimeError("No final particles available for reward logging.")
    best_particle_index = int(torch.argmax(eval_rewards[:common_count]).item())

    if args.save_final_artifacts:
        prompt_dir = os.path.join(args.output_dir, slugify(prompt))
        final_particles_dir = os.path.join(prompt_dir, "final_particles")
        os.makedirs(final_particles_dir, exist_ok=True)

        particle_records = []
        for idx in range(common_count):
            image_path = os.path.join(final_particles_dir, f"final_particle_{idx:03d}.png")
            tensor_to_pil(final_particle_images[idx]).save(image_path)
            particle_records.append(
                {
                    "particle_index": idx,
                    "image_path": image_path,
                    "config_reward": float(config_rewards[idx].item()),
                    "eval_reward": float(eval_rewards[idx].item()),
                    "log_w": float(log_w_cpu[idx].item()),
                    "normalized_w": float(normalized_w_cpu[idx].item()),
                    "is_best_by_eval_reward": idx == best_particle_index,
                }
            )

        final_rewards_path = os.path.join(prompt_dir, "final_rewards.json")
        final_rewards_payload = {
            "prompt": prompt,
            "reward_names": {
                "config_reward": "PickScore",
                "eval_reward": "ImageReward",
            },
            "smc_config": {
                "n_steps": n_steps,
                "num_particles": num_particles,
                "batch_p": batch_p,
                "kl_coeff": kl_coeff,
                "tempering_gamma": tempering_gamma,
            },
            "best_particle_index": best_particle_index,
            "best_particle_config_reward": float(config_rewards[best_particle_index].item()),
            "best_particle_eval_reward": float(eval_rewards[best_particle_index].item()),
            "inference_seconds": inference_seconds,
            "particles": particle_records,
        }
        with open(final_rewards_path, "w", encoding="utf-8") as f:
            json.dump(final_rewards_payload, f, indent=2)
        print(f"Saved final particle images in: {final_particles_dir}")
        print(f"Saved final rewards: {final_rewards_path}")

    print(f"Prompt: {prompt}")
    print(
        f"Best particle {best_particle_index}: "
        f"PickScore={float(config_rewards[best_particle_index].item()):.6f} | "
        f"ImageReward={float(eval_rewards[best_particle_index].item()):.6f}"
    )
    print(f"Inference time: {inference_seconds:.4f}s")


if __name__ == "__main__":
    main()
