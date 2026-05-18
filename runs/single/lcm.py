import argparse
import json
import os
import random
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
from das.diffusers_patch.pipeline_using_SMC_LCM import pipeline_using_smc_lcm
from diffusers import DiffusionPipeline
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

################### Configuration ###################
kl_coeff = 0.0001
n_steps = 8 # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
num_particles = 4
batch_p = 1
tempering_gamma = 0.1

# prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
prompt = "a black motorcycle is parked by the side of the road"
log_dir_lcm_smc = "logs/DAS_LCM/pick/qualitative"


def slugify(text, max_len=60):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    slug = slug.strip("-")
    return slug[:max_len].rstrip("-") or "prompt"


def tensor_to_pil(image):
    image_np = (image.detach().cpu().numpy() * 255).transpose(1, 2, 0).round().astype(np.uint8)
    return Image.fromarray(image_np)


def main():
    parser = argparse.ArgumentParser(description="Run LCM SMC for a single prompt.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling.")
    parser.add_argument("--save-final-artifacts", action="store_true", help="Save final particle images, rewards JSON, and reward CSVs.")
    args = parser.parse_args()
    run_seed = args.seed

    if run_seed is not None:
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

    ################### Rewards ###################
    config_reward_fn = rewards.PickScore(device="cuda")

    def config_image_reward_fn(images):
        prompt_batch = [prompt] * int(images.shape[0])
        scores = config_reward_fn(images, prompt_batch)
        if isinstance(scores, tuple):
            scores = scores[-1]
        return torch.as_tensor(scores)

    ################### Initialize ###################
    pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")

    # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
    pipe.to("cuda", torch.float32)
    pipe.vae.to(dtype=torch.float32)
    pipe.text_encoder.to(dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(run_seed) if run_seed is not None else None

    ################### Inference ###################
    start = time.perf_counter()
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
    ) = pipeline_using_smc_lcm(
        pipe,
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=n_steps,
        generator=generator,
        eta=0.5,
        output_type="pt",
        # SMC parameters
        num_particles=num_particles,
        batch_p=batch_p,
        tempering_gamma=tempering_gamma,
        reward_fn=config_image_reward_fn,
        kl_coeff=kl_coeff,
        verbose=True,
        return_final_particles=True,
    )
    inference_seconds = time.perf_counter() - start

    score_map = evaluate_all_rewards(
        final_particle_images,
        prompt,
        device="cuda",
        prebuilt_reward_fns={STEERING_REWARD: config_reward_fn},
    )
    log_w_cpu = log_w.detach().cpu().to(torch.float32).flatten()
    normalized_w_cpu = normalized_w.detach().cpu().to(torch.float32).flatten()
    common_count = common_reward_count(final_particle_images, score_map, log_w_cpu, normalized_w_cpu)
    if common_count <= 0:
        raise RuntimeError("No final particles available for reward logging.")
    reward_values = rewards_to_lists(score_map, common_count)
    best_particle_index = best_index_for_reward(reward_values)

    if args.save_final_artifacts:
        prompt_dir = os.path.join(log_dir_lcm_smc, slugify(prompt))
        os.makedirs(prompt_dir, exist_ok=True)

        particle_records = []
        for idx in range(common_count):
            image_path = os.path.join(prompt_dir, f"final_particle_{idx:03d}.png")
            tensor_to_pil(final_particle_images[idx]).save(image_path)
            particle_records.append(
                build_particle_record(
                    idx,
                    image_path,
                    reward_values,
                    best_particle_index,
                    extras={
                        "log_w": float(log_w_cpu[idx].item()),
                        "normalized_w": float(normalized_w_cpu[idx].item()),
                    },
                )
            )

        final_rewards_path = os.path.join(prompt_dir, "final_rewards.json")
        final_rewards_payload = build_final_rewards_payload(
            prompt=prompt,
            reward_values=reward_values,
            selected_best_index=best_particle_index,
            particles=particle_records,
            inference_seconds=inference_seconds,
            config_key="smc_config",
            config={
                "n_steps": n_steps,
                "num_particles": num_particles,
                "batch_p": batch_p,
                "kl_coeff": kl_coeff,
                "tempering_gamma": tempering_gamma,
                "eta": 0.5,
                "seed": run_seed,
            },
        )
        with open(final_rewards_path, "w", encoding="utf-8") as f:
            json.dump(final_rewards_payload, f, indent=2)
        reward_summary_row = reward_summary_row_from_values(
            prompt=prompt,
            reward_values=reward_values,
            best_particle_index=best_particle_index,
            final_rewards_path=Path(final_rewards_path).resolve(),
            elapsed=inference_seconds,
        )
        reward_summary_csv_path = os.path.join(prompt_dir, "reward_summary.csv")
        reward_summary_stats_csv_path = os.path.join(prompt_dir, "reward_summary_stats.csv")
        write_reward_summary_csv(Path(reward_summary_csv_path), [reward_summary_row])
        write_reward_summary_stats_csv(Path(reward_summary_stats_csv_path), [reward_summary_row])
        print(f"Saved final particle images in: {prompt_dir}")
        print(f"Saved final rewards: {final_rewards_path}")
        print(f"Saved reward summary CSV: {reward_summary_csv_path}")
        print(f"Saved reward statistics CSV: {reward_summary_stats_csv_path}")

    best_scores = " | ".join(
        f"{reward_key}={values[best_particle_index]:.6f}" for reward_key, values in reward_values.items()
    )
    print(f"Best particle {best_particle_index}: {best_scores}")


if __name__ == "__main__":
    main()
