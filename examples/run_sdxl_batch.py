
import subprocess
import os
from rich.console import Console
from rich.table import Table
import tempfile
import argparse
import json


def load_prompts(prompts_arg, prompts_file):
    import json as _json
    if prompts_file:
        if prompts_file.endswith('.json'):
            with open(prompts_file, 'r') as f:
                data = _json.load(f)
            # Accept list of strings or list of dicts with 'prompt' key
            if isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    return data
                elif all(isinstance(item, dict) and 'prompt' in item for item in data):
                    return [item['prompt'] for item in data]
                else:
                    raise ValueError("JSON file must be a list of strings or list of dicts with 'prompt' key.")
            else:
                raise ValueError("JSON file must be a list.")
        else:
            # Assume txt: one prompt per line
            with open(prompts_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
    if prompts_arg:
        return prompts_arg
    # Default prompts
    return [
        "A close up of a handpalm with leaves growing from it.",
        "A futuristic cityscape at sunset.",
        "A cat riding a skateboard.",
        "A serene mountain landscape with a lake.",
    ]

def main():
    parser = argparse.ArgumentParser(description="Batch run SDXL for multiple prompts.")
    parser.add_argument('--prompts', nargs='+', help='List of prompts (overrides --prompts_file)')
    parser.add_argument('--prompts_file', type=str, help='File with one prompt per line')
    parser.add_argument('--output_dir', type=str, default="logs/DAS_SDXL/pick/qualitative", help='Output directory')
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--num_particles', type=int, default=4)
    parser.add_argument('--batch_p', type=int, default=1)
    parser.add_argument('--kl_coeff', type=float, default=1.)
    parser.add_argument('--tempering_gamma', type=float, default=0.008)
    parser.add_argument('--script', type=str, default=None, help='Path to sdxl.py (default: examples/sdxl.py)')
    args = parser.parse_args()

    SCRIPT = args.script or os.path.join(os.path.dirname(__file__), "sdxl.py")
    PROMPTS = load_prompts(args.prompts, args.prompts_file)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Prompt", style="dim", width=40)
    table.add_column("PickScore", justify="right")
    table.add_column("ImageReward", justify="right")
    table.add_column("Image Path", width=40)
    table.add_column("Log Path", width=30)

    for prompt in PROMPTS:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as log_file:
            log_json = log_file.name
        cmd = [
            "python3", SCRIPT,
            "--prompt", prompt,
            "--log_json", log_json,
            "--output_dir", args.output_dir,
            "--n_steps", str(args.n_steps),
            "--num_particles", str(args.num_particles),
            "--batch_p", str(args.batch_p),
            "--kl_coeff", str(args.kl_coeff),
            "--tempering_gamma", str(args.tempering_gamma)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        pickscore = "-"
        imagereward = "-"
        image_path = "-"
        for line in result.stdout.splitlines():
            if line.startswith("PickScore"):
                parts = line.split("|")
                if len(parts) == 2:
                    pickscore = parts[0].split(":")[-1].strip()
                    imagereward = parts[1].split(":")[-1].strip()
            elif line.startswith("Saved image"):
                image_path = line.split(":", 1)[-1].strip()
        table.add_row(prompt, pickscore, imagereward, image_path, log_json)

    console.print("\n[bold green]SDXL Batch Run Results[/bold green]")
    console.print(table)
    console.print("\n[dim]Intermediate rewards logs are saved as JSON for each prompt.[/dim]")

if __name__ == "__main__":
    main()
