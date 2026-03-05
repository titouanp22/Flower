from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_steps(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("steps list must not be empty")
    if any(v <= 0 for v in values):
        raise ValueError("all steps must be positive")
    return values


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run FLOWER generation for several delta_t values (delta_t = 1/steps) "
            "and save intermediate step1/step2/step3 debug images."
        )
    )
    parser.add_argument("--steps-list", type=str, default="50,100,200")
    parser.add_argument("--dataset", type=str, default="afhq_cat")
    parser.add_argument("--model", type=str, default="flow_indp")
    parser.add_argument("--problem", type=str, default="superresolution")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument("--batch-size-ip", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--step-debug-stride", type=int, default=10)

    args = parser.parse_args()
    steps_values = parse_steps(args.steps_list)

    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"

    for steps in steps_values:
        delta_t = 1.0 / float(steps)
        print(f"\n=== FLOWER run: steps={steps} (delta_t={delta_t:.6f}) ===")
        cmd = [
            sys.executable,
            str(main_py),
            "--opts",
            "dataset", args.dataset,
            "eval_split", args.eval_split,
            "model", args.model,
            "problem", args.problem,
            "method", "flower",
            "num_samples", str(args.num_samples),
            "max_batch", str(args.max_batch),
            "batch_size_ip", str(args.batch_size_ip),
            "steps", str(steps),
            "save_step_debug", "True",
            "step_debug_stride", str(args.step_debug_stride),
            "device", args.device,
        ]
        run_cmd(cmd, cwd=project_root)


if __name__ == "__main__":
    main()
