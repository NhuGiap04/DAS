import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import das.rewards as rewards
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from PIL import Image
from runs.reward_eval import (
    STEERING_REWARD,
    best_index_for_reward,
    build_final_rewards_payload,
    build_particle_record,
    common_reward_count,
    evaluate_all_rewards,
    reward_summary_row_from_values,
    rewards_to_lists,
    write_reward_summary_csv,
    write_reward_summary_stats_csv,
)


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
    parser.add_argument("--save-final-artifacts", action="store_true", help="Save final images, rewards JSON, and reward CSVs.")
    args = parser.parse_args()

    config_reward_fn = rewards.PickScore(device="cuda")

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

    score_map = evaluate_all_rewards(
        final_images,
        args.prompt,
        device="cuda",
        prebuilt_reward_fns={STEERING_REWARD: config_reward_fn},
    )
    common_count = common_reward_count(final_images, score_map)
    if common_count <= 0:
        raise RuntimeError("No final images available for reward logging.")
    reward_values = rewards_to_lists(score_map, common_count)
    best_image_index = best_index_for_reward(reward_values)

    if args.save_final_artifacts:
        prompt_dir = args.output_dir
        os.makedirs(prompt_dir, exist_ok=True)

        particle_records = []
        for idx in range(common_count):
            image_path = os.path.join(prompt_dir, f"final_particle_{idx:03d}.png")
            tensor_to_pil(final_images[idx]).save(image_path)
            particle_records.append(
                build_particle_record(idx, image_path, reward_values, best_image_index)
            )

        final_rewards_path = os.path.join(prompt_dir, "final_rewards.json")
        final_rewards_payload = build_final_rewards_payload(
            prompt=args.prompt,
            reward_values=reward_values,
            selected_best_index=best_image_index,
            particles=particle_records,
            inference_seconds=inference_seconds,
            config_key="baseline_config",
            config={
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "n_steps": args.n_steps,
                "num_images": args.num_images,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
            },
            extra_fields={
                "generation_type": "baseline",
                "steering_enabled": False,
            },
        )
        with open(final_rewards_path, "w", encoding="utf-8") as f:
            json.dump(final_rewards_payload, f, indent=2)
        reward_summary_row = reward_summary_row_from_values(
            prompt=args.prompt,
            reward_values=reward_values,
            best_particle_index=best_image_index,
            final_rewards_path=Path(final_rewards_path).resolve(),
            elapsed=inference_seconds,
        )
        reward_summary_csv_path = os.path.join(prompt_dir, "reward_summary.csv")
        reward_summary_stats_csv_path = os.path.join(prompt_dir, "reward_summary_stats.csv")
        write_reward_summary_csv(Path(reward_summary_csv_path), [reward_summary_row])
        write_reward_summary_stats_csv(Path(reward_summary_stats_csv_path), [reward_summary_row])
        print(f"Saved final images in: {prompt_dir}")
        print(f"Saved final rewards: {final_rewards_path}")
        print(f"Saved reward summary CSV: {reward_summary_csv_path}")
        print(f"Saved reward statistics CSV: {reward_summary_stats_csv_path}")

    print(f"Prompt: {args.prompt}")
    best_scores = " | ".join(
        f"{reward_key}={values[best_image_index]:.6f}" for reward_key, values in reward_values.items()
    )
    print(f"Best image {best_image_index}: {best_scores}")
    print(f"Inference time: {inference_seconds:.4f}s")


if __name__ == "__main__":
    main()
