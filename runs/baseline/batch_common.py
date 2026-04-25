import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _supports_color() -> bool:
    return sys.stdout.isatty()


class _Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


USE_COLOR = _supports_color()


def _c(text: str, *styles: str) -> str:
    if not USE_COLOR or not styles:
        return text
    return "".join(styles) + text + _Style.RESET


def _title(text: str) -> None:
    line = "=" * len(text)
    print(_c(line, _Style.CYAN, _Style.BOLD))
    print(_c(text, _Style.CYAN, _Style.BOLD))
    print(_c(line, _Style.CYAN, _Style.BOLD))


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    slug = slug.strip("-")
    if not slug:
        return "prompt"
    return slug[:max_len].rstrip("-")


def _truncate(text: str, max_len: int = 72) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _extract_prompts_from_json(data: Any) -> List[str]:
    if isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            return [item.strip() for item in data if item.strip()]
        prompts: List[str] = []
        for item in data:
            if isinstance(item, str):
                prompt = item.strip()
                if prompt:
                    prompts.append(prompt)
                continue
            if isinstance(item, dict):
                for key in ("prompt", "text"):
                    value = item.get(key)
                    if isinstance(value, str):
                        prompt = value.strip()
                        if prompt:
                            prompts.append(prompt)
                            break
                else:
                    raise ValueError("JSON list entries must be strings or objects with 'prompt'/'text'.")
                continue
            raise ValueError("Unsupported JSON list entry type for prompts.")
        return prompts
    if isinstance(data, dict):
        if "prompts" in data:
            return _extract_prompts_from_json(data["prompts"])
        raise ValueError("JSON object must contain a 'prompts' field.")
    raise ValueError("JSON prompt file must be a list or an object with 'prompts'.")


def load_prompts(file_path: Path) -> List[str]:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        prompts = []
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
        return prompts
    if suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return _extract_prompts_from_json(data)
    raise ValueError(f"Unsupported prompt file extension: {suffix}. Use .txt or .json")


def _build_baseline_cmd(args, prompt, run_output_dir):
    cmd = [
        args.python,
        str(args.baseline_script),
        "--prompt",
        prompt,
        "--output_dir",
        str(run_output_dir),
        "--n_steps",
        str(args.n_steps),
        "--num_images",
        str(args.num_images),
        "--guidance_scale",
        str(args.guidance_scale),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.save_final_artifacts:
        cmd.append("--save-final-artifacts")
    return cmd


def _collect_run_artifacts(run_output_dir: Path, run_name: str, prompt: str, artifacts_root: Path) -> Dict[str, Any]:
    run_artifacts_dir = artifacts_root / run_name
    particles_dir = run_artifacts_dir / "final_particles"
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)

    copied_files = {
        "final_rewards_json": None,
    }
    final_particle_images: List[str] = []
    missing_files: List[str] = []

    final_rewards_matches = list(run_output_dir.rglob("final_rewards.json"))
    if final_rewards_matches:
        src = max(final_rewards_matches, key=lambda p: p.stat().st_mtime)
        dst = run_artifacts_dir / "final_rewards.json"
        shutil.copy2(src, dst)
        copied_files["final_rewards_json"] = str(dst)
    else:
        missing_files.append("final_rewards.json")

    particle_image_candidates = sorted(
        [p for p in run_output_dir.rglob("final_particle_*.png")],
        key=lambda p: p.name,
    )
    if particle_image_candidates:
        particles_dir.mkdir(parents=True, exist_ok=True)
        for src in particle_image_candidates:
            dst = particles_dir / src.name
            shutil.copy2(src, dst)
            final_particle_images.append(str(dst))
    else:
        missing_files.append("final particle images (final_particle_*.png)")

    return {
        "run_name": run_name,
        "prompt": prompt,
        "run_output_dir": str(run_output_dir),
        "artifacts_dir": str(run_artifacts_dir),
        "files": copied_files,
        "final_particle_images": final_particle_images,
        "missing": missing_files,
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _reward_summary_row(
    final_rewards_path: Path,
    run_index: int,
    run_name: str,
    prompt: str,
    elapsed: float,
) -> Dict[str, Any]:
    payload = json.loads(final_rewards_path.read_text(encoding="utf-8"))
    particles = payload.get("particles", [])
    eval_rewards = [
        float(particle["eval_reward"])
        for particle in particles
        if particle.get("eval_reward") is not None
    ]
    steer_rewards = [
        float(particle["config_reward"])
        for particle in particles
        if particle.get("config_reward") is not None
    ]
    if not eval_rewards:
        raise ValueError("final_rewards.json has no particle eval_reward values")
    if not steer_rewards:
        raise ValueError("final_rewards.json has no particle config_reward values")

    return {
        "index": run_index,
        "run_name": run_name,
        "prompt": prompt,
        "num_particles": len(particles),
        "mean_eval_reward": _mean(eval_rewards),
        "max_eval_reward": max(eval_rewards),
        "mean_steer_reward": _mean(steer_rewards),
        "max_steer_reward": max(steer_rewards),
        "best_particle_index": payload.get("best_particle_index"),
        "elapsed_seconds": elapsed,
        "final_rewards_json": str(final_rewards_path),
    }


def _write_reward_summary_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "run_name",
        "prompt",
        "num_particles",
        "mean_eval_reward",
        "max_eval_reward",
        "mean_steer_reward",
        "max_steer_reward",
        "best_particle_index",
        "elapsed_seconds",
        "final_rewards_json",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows):
    if not rows:
        return
    print()
    _title("Batch Summary")
    headers = ["Idx", "Status", "Time(s)", "Prompt"]
    table_data = []
    for row in rows:
        table_data.append([
            str(row["index"]),
            row["status"],
            f"{row['elapsed']:.2f}",
            _truncate(row["prompt"], 68),
        ])
    col_widths = [len(h) for h in headers]
    for r in table_data:
        for i, cell in enumerate(r):
            col_widths[i] = max(col_widths[i], len(cell))

    def format_row(cells):
        padded = [cells[i].ljust(col_widths[i]) for i in range(len(cells))]
        return " | ".join(padded)

    print(format_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in table_data:
        status = r[1]
        if status == "OK":
            r[1] = _c(status, _Style.GREEN, _Style.BOLD)
        elif status == "DRY":
            r[1] = _c(status, _Style.YELLOW, _Style.BOLD)
        else:
            r[1] = _c(status, _Style.RED, _Style.BOLD)
        print(format_row(r))


def parse_args(model_label: str, default_script: Path, default_output_dir: Path):
    parser = argparse.ArgumentParser(
        description=f"Run baseline {model_label} over prompts from a .txt or .json file.",
    )
    parser.add_argument("--prompts_file", type=Path, required=True, help="Path to .txt or .json prompts file.")
    parser.add_argument("--baseline_script", type=Path, default=default_script, help=f"Path to the single-prompt baseline {model_label} script.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch the baseline script.")
    parser.add_argument("--output_dir", type=Path, default=default_output_dir)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None, help="Random seed passed to each baseline generation run.")
    parser.add_argument("--save-final-artifacts", action="store_true", help="Save and collect final images and rewards JSON.")
    parser.add_argument("--start_index", type=int, default=0, help="Start from this 0-based prompt index.")
    parser.add_argument("--max_prompts", type=int, default=None, help="Limit number of prompts to run.")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop batch on first failed prompt.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--log_dir", type=Path, default=None, help="Where to save per-run stdout/stderr logs (default: <output-dir>/_batch_logs).")
    return parser.parse_args()


def run_batch(model_label: str, default_script: Path, default_output_dir: Path) -> int:
    args = parse_args(model_label, default_script, default_output_dir)
    if not args.prompts_file.exists():
        print(_c(f"Prompt file not found: {args.prompts_file}", _Style.RED, _Style.BOLD))
        return 2
    if not args.baseline_script.exists():
        print(_c(f"Baseline script not found: {args.baseline_script}", _Style.RED, _Style.BOLD))
        return 2
    prompts = load_prompts(args.prompts_file)
    if not prompts:
        print(_c("No prompts were loaded from input file.", _Style.RED, _Style.BOLD))
        return 2
    if args.start_index < 0 or args.start_index >= len(prompts):
        print(_c("--start-index is out of range.", _Style.RED, _Style.BOLD))
        return 2
    end_index = len(prompts)
    if args.max_prompts is not None:
        if args.max_prompts < 1:
            print(_c("--max-prompts must be >= 1.", _Style.RED, _Style.BOLD))
            return 2
        end_index = min(end_index, args.start_index + args.max_prompts)

    selected_prompts = prompts[args.start_index:end_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or (args.output_dir / "_batch_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root = args.output_dir / "_batch_artifacts"
    if args.save_final_artifacts:
        artifacts_root.mkdir(parents=True, exist_ok=True)
    artifacts_manifest_path = artifacts_root / "manifest.jsonl"
    reward_summary_csv_path = artifacts_root / "reward_summary.csv"

    _title(f"Baseline {model_label} Batch Runner")
    print(f"Prompt file : {args.prompts_file}")
    print(f"Script      : {args.baseline_script}")
    print(f"Runs        : {len(selected_prompts)} (from index {args.start_index})")
    print(f"Output root : {args.output_dir}")
    print(f"Log dir     : {log_dir}")
    print(f"Artifacts   : {artifacts_root if args.save_final_artifacts else 'disabled'}")
    print()

    rows = []
    reward_summary_rows = []
    success_count = 0
    batch_start = time.time()
    total_runs = len(selected_prompts)
    for local_idx, prompt in enumerate(selected_prompts, start=1):
        global_idx = args.start_index + local_idx - 1
        prompt_slug = _slugify(prompt)
        run_name = f"run_{global_idx:04d}_{prompt_slug}"
        run_output_dir = args.output_dir / run_name
        cmd = _build_baseline_cmd(args, prompt, run_output_dir)
        print(_c(f"[{local_idx:03d}/{total_runs:03d}]", _Style.BOLD), _truncate(prompt, 100))
        print(_c("  output:", _Style.DIM), run_output_dir)
        if args.dry_run:
            print(_c("  dry-run command:", _Style.DIM), shlex.join(cmd))
            success_count += 1
            rows.append({
                "index": global_idx,
                "status": "DRY",
                "elapsed": 0.0,
                "prompt": prompt,
            })
            continue

        start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        stdout_path = log_dir / f"{run_name}.stdout.log"
        stderr_path = log_dir / f"{run_name}.stderr.log"
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")

        if proc.returncode == 0:
            success_count += 1
            rows.append({
                "index": global_idx,
                "status": "OK",
                "elapsed": elapsed,
                "prompt": prompt,
            })
            print(_c(f"  status: {_c('OK', _Style.GREEN, _Style.BOLD)}  time: {elapsed:.2f}s", _Style.DIM))
            if args.save_final_artifacts:
                artifact_manifest = _collect_run_artifacts(
                    run_output_dir=run_output_dir,
                    run_name=run_name,
                    prompt=prompt,
                    artifacts_root=artifacts_root,
                )
                with artifacts_manifest_path.open("a", encoding="utf-8") as mf:
                    mf.write(json.dumps(artifact_manifest) + "\n")
                if artifact_manifest["missing"]:
                    print(_c("  artifact warnings:", _Style.YELLOW, _Style.BOLD), "; ".join(artifact_manifest["missing"]))
                if artifact_manifest["files"].get("final_rewards_json"):
                    print(_c("  final rewards:", _Style.DIM), artifact_manifest["files"]["final_rewards_json"])
                    try:
                        reward_summary_rows.append(
                            _reward_summary_row(
                                final_rewards_path=Path(artifact_manifest["files"]["final_rewards_json"]),
                                run_index=global_idx,
                                run_name=run_name,
                                prompt=prompt,
                                elapsed=elapsed,
                            )
                        )
                    except Exception as exc:
                        print(_c("  reward summary warning:", _Style.YELLOW, _Style.BOLD), str(exc))
                if artifact_manifest.get("final_particle_images"):
                    print(
                        _c("  final images:", _Style.DIM),
                        f"{len(artifact_manifest['final_particle_images'])} files",
                    )
        else:
            rows.append({
                "index": global_idx,
                "status": "FAIL",
                "elapsed": elapsed,
                "prompt": prompt,
            })
            print(_c(f"  status: {_c('FAIL', _Style.RED, _Style.BOLD)}  time: {elapsed:.2f}s  code: {proc.returncode}", _Style.DIM))
            stderr_tail = (proc.stderr or "").splitlines()[-20:]
            if stderr_tail:
                print(_c("  stderr tail:", _Style.YELLOW, _Style.BOLD))
                for line in stderr_tail:
                    print("   ", line)
            if args.stop_on_error:
                print(_c("Stopping due to --stop-on-error", _Style.YELLOW, _Style.BOLD))
                break
        print()

    total_elapsed = time.time() - batch_start
    _print_summary(rows)
    if args.save_final_artifacts and reward_summary_rows:
        _write_reward_summary_csv(reward_summary_csv_path, reward_summary_rows)
    print()
    _title("Result")
    print(f"Succeeded : {success_count}/{len(rows)}")
    print(f"Failed    : {len(rows) - success_count}")
    print(f"Wall time : {total_elapsed:.2f}s")
    print(f"Logs      : {log_dir}")
    if args.save_final_artifacts:
        print(f"Artifacts : {artifacts_root}")
        print(f"Manifest  : {artifacts_manifest_path}")
        if reward_summary_rows:
            print(f"Rewards CSV: {reward_summary_csv_path}")
    else:
        print("Artifacts : disabled")
    return 0 if success_count == len(rows) else 1
