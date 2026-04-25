import argparse
import json
import os
import re
import time

import das.rewards as rewards
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from PIL import Image


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
    parser = argparse.ArgumentParser(description="Run baseline SDXL without reward steering.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="logs/baseline_sdxl", help="Output directory")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic generation.")
    parser.add_argument("--save-final-artifacts", action="store_true", help="Save final images and rewards JSON.")
    args = parser.parse_args()

    config_reward_fn = rewards.PickScore(device="cuda")

    def config_image_reward_fn(images):
        prompt_batch = [args.prompt] * int(images.shape[0])
        scores = config_reward_fn(images, prompt_batch)
        if isinstance(scores, tuple):
            scores = scores[-1]
        return torch.as_tensor(scores)

    eval_reward_fn = rewards.ImageReward(device="cuda")

    def eval_image_reward_fn(images):
        prompt_batch = [args.prompt] * int(images.shape[0])
        scores = eval_reward_fn(images, prompt_batch)
        if isinstance(scores, tuple):
            scores = scores[-1]
        return torch.as_tensor(scores)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(args.n_steps)
    pipe.to("cuda")
    pipe.vae.to(dtype=torch.float32)
    pipe.text_encoder.to(dtype=torch.float32)

    cuda_sync()
    start = time.perf_counter()
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
    output = pipe(
        prompt=args.prompt,
        negative_prompt="",
        num_inference_steps=args.n_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        generator=generator,
        output_type="pt",
    )
    cuda_sync()
    inference_seconds = time.perf_counter() - start
    final_images = output.images

    config_rewards = config_image_reward_fn(final_images).detach().cpu().to(torch.float32)
    eval_rewards = eval_image_reward_fn(final_images).detach().cpu().to(torch.float32)
    common_count = min(int(final_images.shape[0]), int(config_rewards.numel()), int(eval_rewards.numel()))
    if common_count <= 0:
        raise RuntimeError("No final images available for reward logging.")
    best_image_index = int(torch.argmax(eval_rewards[:common_count]).item())

    if args.save_final_artifacts:
        prompt_dir = os.path.join(args.output_dir, slugify(args.prompt))
        final_particles_dir = os.path.join(prompt_dir, "final_particles")
        os.makedirs(final_particles_dir, exist_ok=True)

        particle_records = []
        for idx in range(common_count):
            image_path = os.path.join(final_particles_dir, f"final_particle_{idx:03d}.png")
            tensor_to_pil(final_images[idx]).save(image_path)
            particle_records.append(
                {
                    "particle_index": idx,
                    "image_path": image_path,
                    "config_reward": float(config_rewards[idx].item()),
                    "eval_reward": float(eval_rewards[idx].item()),
                    "is_best_by_eval_reward": idx == best_image_index,
                }
            )

        final_rewards_path = os.path.join(prompt_dir, "final_rewards.json")
        final_rewards_payload = {
            "prompt": args.prompt,
            "generation_type": "baseline",
            "steering_enabled": False,
            "reward_names": {
                "config_reward": "PickScore",
                "eval_reward": "ImageReward",
            },
            "baseline_config": {
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "n_steps": args.n_steps,
                "num_images": args.num_images,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
            },
            "best_particle_index": best_image_index,
            "best_particle_config_reward": float(config_rewards[best_image_index].item()),
            "best_particle_eval_reward": float(eval_rewards[best_image_index].item()),
            "inference_seconds": inference_seconds,
            "particles": particle_records,
        }
        with open(final_rewards_path, "w", encoding="utf-8") as f:
            json.dump(final_rewards_payload, f, indent=2)
        print(f"Saved final images in: {final_particles_dir}")
        print(f"Saved final rewards: {final_rewards_path}")

    print(f"Prompt: {args.prompt}")
    print(
        f"Best image {best_image_index}: "
        f"PickScore={float(config_rewards[best_image_index].item()):.6f} | "
        f"ImageReward={float(eval_rewards[best_image_index].item()):.6f}"
    )
    print(f"Inference time: {inference_seconds:.4f}s")


if __name__ == "__main__":
    main()
