from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


REWARD_SPECS = (
    ("pickscore", "PickScore", "PickScore"),
    ("imagereward", "ImageReward", "ImageReward"),
    ("clip", "CLIP", "clip_score"),
    ("aesthetic", "Aesthetic", "aesthetic_score"),
    ("hpsv2", "HPSv2", "hps_score"),
)

DEFAULT_SELECTION_REWARD = "imagereward"
STEERING_REWARD = "pickscore"


def reward_names() -> "OrderedDict[str, str]":
    return OrderedDict((key, name) for key, name, _ in REWARD_SPECS)


def _torch():
    import torch

    return torch


def _reward_builder(builder_name: str) -> Any:
    import das.rewards as rewards

    return getattr(rewards, builder_name)


def _as_score_tensor(scores: Any) -> torch.Tensor:
    torch = _torch()
    if isinstance(scores, tuple):
        scores = scores[-1]
    return torch.as_tensor(scores).detach().cpu().to(torch.float32).flatten()


def score_images(
    reward_fn: Any,
    images: torch.Tensor,
    prompt: str,
) -> torch.Tensor:
    torch = _torch()
    prompt_batch = [prompt] * int(images.shape[0])
    with torch.no_grad():
        scores = reward_fn(images, prompt_batch)
    return _as_score_tensor(scores)


def build_reward_fn(reward_key: str, device: str = "cuda") -> Any:
    for key, _, builder_name in REWARD_SPECS:
        if key == reward_key:
            return _reward_builder(builder_name)(device=device)
    raise KeyError(f"Unknown reward key: {reward_key}")


def evaluate_all_rewards(
    images: torch.Tensor,
    prompt: str,
    device: str = "cuda",
    prebuilt_reward_fns: Optional[Mapping[str, Any]] = None,
) -> "OrderedDict[str, torch.Tensor]":
    torch = _torch()
    prebuilt_reward_fns = prebuilt_reward_fns or {}
    score_map: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for reward_key, _, builder_name in REWARD_SPECS:
        reward_fn = prebuilt_reward_fns.get(reward_key)
        owns_reward_fn = reward_fn is None
        if reward_fn is None:
            reward_fn = _reward_builder(builder_name)(device=device)
        score_map[reward_key] = score_images(reward_fn, images, prompt)
        if owns_reward_fn:
            del reward_fn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return score_map


def common_reward_count(images: torch.Tensor, score_map: Mapping[str, torch.Tensor], *extra_tensors: torch.Tensor) -> int:
    counts = [int(images.shape[0])]
    counts.extend(int(scores.numel()) for scores in score_map.values())
    counts.extend(int(tensor.numel()) for tensor in extra_tensors)
    return min(counts)


def rewards_to_lists(score_map: Mapping[str, torch.Tensor], count: int) -> "OrderedDict[str, List[float]]":
    return OrderedDict(
        (reward_key, [float(value.item()) for value in scores[:count]])
        for reward_key, scores in score_map.items()
    )


def best_index_for_reward(
    reward_values: Mapping[str, Sequence[float]],
    reward_key: str = DEFAULT_SELECTION_REWARD,
) -> int:
    values = reward_values.get(reward_key)
    if not values:
        reward_key = next(iter(reward_values))
        values = reward_values[reward_key]
    return max(range(len(values)), key=lambda idx: values[idx])


def reward_summary(reward_values: Mapping[str, Sequence[float]]) -> "OrderedDict[str, Dict[str, Any]]":
    summary: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for reward_key, values in reward_values.items():
        if not values:
            continue
        summary[reward_key] = {
            "reward_name": reward_names().get(reward_key, reward_key),
            "mean_reward": sum(values) / len(values),
            "max_reward": max(values),
            "best_particle_index": max(range(len(values)), key=lambda idx: values[idx]),
        }
    return summary


def build_particle_record(
    particle_index: int,
    image_path: str,
    reward_values: Mapping[str, Sequence[float]],
    selected_best_index: int,
    extras: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    per_particle_rewards = OrderedDict(
        (reward_key, float(values[particle_index]))
        for reward_key, values in reward_values.items()
        if particle_index < len(values)
    )
    record: Dict[str, Any] = {
        "particle_index": particle_index,
        "image_path": image_path,
        "rewards": per_particle_rewards,
        "config_reward": per_particle_rewards.get(STEERING_REWARD),
        "eval_reward": per_particle_rewards.get(DEFAULT_SELECTION_REWARD),
        "is_best_by_eval_reward": particle_index == selected_best_index,
    }
    if extras:
        record.update(extras)
    return record


def build_final_rewards_payload(
    prompt: str,
    reward_values: Mapping[str, Sequence[float]],
    selected_best_index: int,
    particles: Sequence[Mapping[str, Any]],
    inference_seconds: float,
    config_key: str,
    config: Mapping[str, Any],
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "reward_names": reward_names(),
        config_key: dict(config),
        "best_particle_index": selected_best_index,
        "best_particle_rewards": OrderedDict(
            (reward_key, float(values[selected_best_index]))
            for reward_key, values in reward_values.items()
            if selected_best_index < len(values)
        ),
        "per_reward_summary": reward_summary(reward_values),
        "inference_seconds": inference_seconds,
        "particles": list(particles),
    }
    payload["best_particle_config_reward"] = payload["best_particle_rewards"].get(STEERING_REWARD)
    payload["best_particle_eval_reward"] = payload["best_particle_rewards"].get(DEFAULT_SELECTION_REWARD)
    if extra_fields:
        payload.update(extra_fields)
    return payload


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _reward_values_from_payload(payload: Mapping[str, Any]) -> "OrderedDict[str, List[float]]":
    particles = payload.get("particles", [])
    names = payload.get("reward_names", {})
    reward_keys: List[str] = []
    if isinstance(names, Mapping):
        reward_keys.extend(str(key) for key in names.keys())
    for particle in particles:
        rewards_map = particle.get("rewards", {}) if isinstance(particle, Mapping) else {}
        if isinstance(rewards_map, Mapping):
            for key in rewards_map.keys():
                if str(key) not in reward_keys:
                    reward_keys.append(str(key))

    if reward_keys:
        values: "OrderedDict[str, List[float]]" = OrderedDict((key, []) for key in reward_keys)
        for particle in particles:
            rewards_map = particle.get("rewards", {}) if isinstance(particle, Mapping) else {}
            for key in reward_keys:
                if isinstance(rewards_map, Mapping) and rewards_map.get(key) is not None:
                    values[key].append(float(rewards_map[key]))
        values = OrderedDict((key, vals) for key, vals in values.items() if vals)
        if values:
            return values

    legacy_values: "OrderedDict[str, List[float]]" = OrderedDict()
    legacy_map = (("imagereward", "eval_reward"), ("pickscore", "config_reward"))
    for reward_key, particle_key in legacy_map:
        vals = [
            float(particle[particle_key])
            for particle in particles
            if isinstance(particle, Mapping) and particle.get(particle_key) is not None
        ]
        if vals:
            legacy_values[reward_key] = vals
    return legacy_values


def reward_summary_row_from_json(
    final_rewards_path: Path,
    run_index: Optional[int] = None,
    run_name: Optional[str] = None,
    prompt: Optional[str] = None,
    elapsed: Optional[float] = None,
) -> Dict[str, Any]:
    payload = json.loads(final_rewards_path.read_text(encoding="utf-8"))
    reward_values = _reward_values_from_payload(payload)
    if not reward_values:
        raise ValueError("final_rewards.json has no particle reward values")

    particles = payload.get("particles", [])
    row: Dict[str, Any] = {
        "index": run_index,
        "run_name": run_name,
        "prompt": prompt if prompt is not None else payload.get("prompt"),
        "num_particles": len(particles),
        "best_particle_index": payload.get("best_particle_index"),
        "elapsed_seconds": elapsed,
        "final_rewards_json": str(final_rewards_path),
    }
    for reward_key, values in reward_values.items():
        row[f"mean_{reward_key}_reward"] = _mean(values)
        row[f"max_{reward_key}_reward"] = max(values)
    return row


def max_reward_row_from_json(
    final_rewards_path: Path,
    run_index: Optional[int] = None,
    run_name: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    payload = json.loads(final_rewards_path.read_text(encoding="utf-8"))
    reward_values = _reward_values_from_payload(payload)
    if not reward_values:
        raise ValueError("final_rewards.json has no particle reward values")

    particles = payload.get("particles", [])
    row: Dict[str, Any] = {
        "index": run_index,
        "run_name": run_name,
        "prompt": prompt if prompt is not None else payload.get("prompt"),
        "num_particles": len(particles),
        "final_rewards_json": str(final_rewards_path),
    }
    for reward_key, values in reward_values.items():
        best_idx = max(range(len(values)), key=lambda idx: values[idx])
        row[f"max_{reward_key}_reward"] = float(values[best_idx])
        row[f"best_{reward_key}_particle_index"] = best_idx
    return row


def reward_summary_row_from_values(
    prompt: str,
    reward_values: Mapping[str, Sequence[float]],
    best_particle_index: int,
    final_rewards_path: Path,
    run_index: Optional[int] = None,
    run_name: Optional[str] = None,
    elapsed: Optional[float] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "index": run_index,
        "run_name": run_name,
        "prompt": prompt,
        "num_particles": max((len(values) for values in reward_values.values()), default=0),
        "best_particle_index": best_particle_index,
        "elapsed_seconds": elapsed,
        "final_rewards_json": str(final_rewards_path),
    }
    for reward_key, values in reward_values.items():
        if not values:
            continue
        row[f"mean_{reward_key}_reward"] = _mean(values)
        row[f"max_{reward_key}_reward"] = max(values)
    return row


def _ordered_fieldnames(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    base = ["index", "run_name", "prompt", "num_particles"]
    reward_fields: List[str] = []
    tail = ["best_particle_index", "elapsed_seconds", "final_rewards_json"]
    for row in rows:
        for key in row.keys():
            if key.startswith("mean_") or key.startswith("max_"):
                if key not in reward_fields:
                    reward_fields.append(key)
    ordered_reward_fields: List[str] = []
    for reward_key, _, _ in REWARD_SPECS:
        for prefix in ("mean", "max"):
            field = f"{prefix}_{reward_key}_reward"
            if field in reward_fields:
                ordered_reward_fields.append(field)
    for field in reward_fields:
        if field not in ordered_reward_fields:
            ordered_reward_fields.append(field)
    fieldnames = [field for field in base if any(field in row for row in rows)]
    fieldnames.extend(ordered_reward_fields)
    fieldnames.extend(field for field in tail if any(field in row for row in rows))
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_reward_summary_csv(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _ordered_fieldnames(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ordered_max_reward_fieldnames(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    base = ["index", "run_name", "prompt", "num_particles"]
    reward_fields: List[str] = []
    tail = ["final_rewards_json"]
    for row in rows:
        for key in row.keys():
            if key.startswith("max_") or (key.startswith("best_") and key.endswith("_particle_index")):
                if key not in reward_fields:
                    reward_fields.append(key)
    ordered_reward_fields: List[str] = []
    for reward_key, _, _ in REWARD_SPECS:
        for field in (f"max_{reward_key}_reward", f"best_{reward_key}_particle_index"):
            if field in reward_fields:
                ordered_reward_fields.append(field)
    for field in reward_fields:
        if field not in ordered_reward_fields:
            ordered_reward_fields.append(field)
    fieldnames = [field for field in base if any(field in row for row in rows)]
    fieldnames.extend(ordered_reward_fields)
    fieldnames.extend(field for field in tail if any(field in row for row in rows))
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_max_reward_csv(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _ordered_max_reward_fieldnames(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reward_statistics_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    stats_rows: List[Dict[str, Any]] = []
    for reward_key, reward_name, _ in REWARD_SPECS:
        mean_field = f"mean_{reward_key}_reward"
        max_field = f"max_{reward_key}_reward"
        mean_values = [float(row[mean_field]) for row in rows if row.get(mean_field) is not None]
        max_values = [float(row[max_field]) for row in rows if row.get(max_field) is not None]
        if not mean_values or not max_values:
            continue
        stats_rows.append(
            {
                "reward_key": reward_key,
                "reward_name": reward_name,
                "num_prompts": len(mean_values),
                "mean_of_mean_rewards": _mean(mean_values),
                "mean_of_max_rewards": _mean(max_values),
            }
        )
    return stats_rows


def write_reward_summary_stats_csv(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    stats_rows = reward_statistics_rows(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["reward_key", "reward_name", "num_prompts", "mean_of_mean_rewards", "mean_of_max_rewards"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)
