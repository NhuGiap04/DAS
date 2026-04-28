import argparse
import json
import os
import random
import re
import time

import das.rewards as rewards
import numpy as np
import torch
from das.diffusers_patch.pipeline_using_SMC import pipeline_using_smc
from diffusers import DDIMScheduler, DiffusionPipeline
from PIL import Image

################### Configuration ###################
kl_coeff = 0.0001
n_steps = 100
num_particles = 4
batch_p = 2
tempering_gamma = 0.008

prompt = "cat and a dog"
log_dir_sd_smc = "logs/DAS_SD/pick/qualitative"


def slugify(text, max_len=60):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    slug = slug.strip("-")
    return slug[:max_len].rstrip("-") or "prompt"


def tensor_to_pil(image):
    image_np = (image.detach().cpu().numpy() * 255).transpose(1, 2, 0).round().astype(np.uint8)
    return Image.fromarray(image_np)


def main():
    parser = argparse.ArgumentParser(description="Run SD SMC for a single prompt.")
    parser.add_argument("--prompt", type=str, default=prompt, help="Prompt for image generation")
    parser.add_argument("--output_dir", type=str, default=log_dir_sd_smc, help="Output directory")
    parser.add_argument("--n_steps", type=int, default=n_steps)
    parser.add_argument("--num_particles", type=int, default=num_particles)
    parser.add_argument("--batch_p", type=int, default=batch_p)
    parser.add_argument("--kl_coeff", type=float, default=kl_coeff)
    parser.add_argument("--tempering_gamma", type=float, default=tempering_gamma)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling.")
    parser.add_argument("--save-final-artifacts", action="store_true", help="Save final particle images and rewards JSON.")
    args = parser.parse_args()

    run_prompt = args.prompt
    run_n_steps = args.n_steps
    run_num_particles = args.num_particles
    run_batch_p = args.batch_p
    run_kl_coeff = args.kl_coeff
    run_tempering_gamma = args.tempering_gamma
    run_seed = args.seed

    if run_seed is not None:
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

    ################### Rewards ###################
    config_reward_fn = rewards.PickScore(device="cuda")
    eval_reward_fn = rewards.ImageReward(device="cuda")

    def config_image_reward_fn(images):
        prompt_batch = [run_prompt] * int(images.shape[0])
        scores = config_reward_fn(images, prompt_batch)
        if isinstance(scores, tuple):
            scores = scores[-1]
        return torch.as_tensor(scores)

    def eval_image_reward_fn(images):
        prompt_batch = [run_prompt] * int(images.shape[0])
        scores = eval_reward_fn(images, prompt_batch)
        if isinstance(scores, tuple):
            scores = scores[-1]
        return torch.as_tensor(scores)

    ################### Initialize ###################
    pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(run_n_steps)
    pipe.to("cuda", torch.float16)
    generator = torch.Generator(device="cuda").manual_seed(run_seed) if run_seed is not None else None

    ################### Inference ###################
    start = time.perf_counter()
    with torch.autocast("cuda"):
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
        ) = pipeline_using_smc(
            pipe,
            prompt=run_prompt,
            negative_prompt="",
            num_inference_steps=run_n_steps,
            generator=generator,
            output_type="pt",
            # SMC parameters
            num_particles=run_num_particles,
            batch_p=run_batch_p,
            tempering_gamma=run_tempering_gamma,
            reward_fn=config_image_reward_fn,
            kl_coeff=run_kl_coeff,
            verbose=True,
            return_final_particles=True,
        )
    inference_seconds = time.perf_counter() - start

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
    best_particle_index = int(torch.argmax(eval_rewards[:common_count]).item())

    if args.save_final_artifacts:
        prompt_dir = os.path.join(args.output_dir, slugify(run_prompt))
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
            "prompt": run_prompt,
            "reward_names": {
                "config_reward": "PickScore",
                "eval_reward": "ImageReward",
            },
            "smc_config": {
                "n_steps": run_n_steps,
                "num_particles": run_num_particles,
                "batch_p": run_batch_p,
                "kl_coeff": run_kl_coeff,
                "tempering_gamma": run_tempering_gamma,
                "seed": run_seed,
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

    print(
        f"Best particle {best_particle_index}: "
        f"PickScore={float(config_rewards[best_particle_index].item()):.6f} | "
        f"ImageReward={float(eval_rewards[best_particle_index].item()):.6f}"
    )


if __name__ == "__main__":
    main()
