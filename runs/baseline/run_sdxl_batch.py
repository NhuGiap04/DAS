from pathlib import Path

from batch_common import run_batch


if __name__ == "__main__":
    raise SystemExit(
        run_batch(
            model_label="SDXL",
            default_script=Path("runs/baseline/sdxl.py"),
            default_output_dir=Path("logs/baseline_sdxl_batch"),
        )
    )
