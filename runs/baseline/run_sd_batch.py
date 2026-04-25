from pathlib import Path

from batch_common import run_batch


if __name__ == "__main__":
    raise SystemExit(
        run_batch(
            model_label="SD",
            default_script=Path("runs/baseline/sd.py"),
            default_output_dir=Path("logs/baseline_sd_batch"),
        )
    )
