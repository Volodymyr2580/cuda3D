#!/usr/bin/env python3
"""Safe wrapper for true multi-GPU CUDA3D shot batching.

The wrapper refuses same-GPU oversubscription.  It prepares an input copy with
the final gpus_p_node line set to the number of requested visible GPUs, then can
optionally call tools/run_benchmark.py.  By default it runs in dry-run mode.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CASE_INPUTS = {
    "perf_1gpu_6shots": (
        Path("benchmarks/cases/perf_1gpu_6shots"),
        "input_perf_1gpu_6shots.in",
    ),
}


def visible_gpu_count_from_nvidia_smi() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return 0
    if result.returncode != 0:
        return 0
    return sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU "))


def parse_gpu_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def write_input_variant(case_dir: Path, input_name: str, gpu_count: int) -> str:
    src = case_dir / input_name
    lines = src.read_text(encoding="utf-8").splitlines()
    last = None
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            last = idx
            break
    if last is None:
        raise ValueError(f"{src} is empty")
    lines[last] = str(gpu_count)
    stem = Path(input_name).stem
    suffix = Path(input_name).suffix
    out_name = f"{stem}_gpus{gpu_count}{suffix}"
    (case_dir / out_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="perf_1gpu_6shots", choices=CASE_INPUTS)
    parser.add_argument("--gpus", required=True, help="Comma-separated visible GPU ids, e.g. 0,1,2")
    parser.add_argument("--np", type=int, help="MPI rank count; defaults to number of requested GPUs")
    parser.add_argument("--tag", default="true_multigpu_batching")
    parser.add_argument("--execute", action="store_true", help="Actually run tools/run_benchmark.py")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    case_dir_rel, input_name = CASE_INPUTS[args.case]
    case_dir = root / case_dir_rel
    requested_gpus = parse_gpu_list(args.gpus)
    gpu_count = len(requested_gpus)
    mpi_np = args.np or gpu_count

    physical_count = visible_gpu_count_from_nvidia_smi()
    if gpu_count < 2:
        print("Refusing to run: true multi-GPU batching requires at least 2 requested GPUs.")
        return 2
    if physical_count < gpu_count:
        print(
            f"Refusing to run: nvidia-smi reports {physical_count} GPUs, "
            f"but {gpu_count} were requested."
        )
        return 2
    if mpi_np != gpu_count:
        print(
            f"Refusing to run: --np ({mpi_np}) must equal requested GPU count ({gpu_count}) "
            "for one-rank-per-GPU validation."
        )
        return 2

    input_variant = write_input_variant(case_dir, input_name, gpu_count)
    cmd = [
        sys.executable,
        str(root / "tools" / "run_benchmark.py"),
        "--case",
        args.case,
        "--input",
        input_variant,
        "--np",
        str(mpi_np),
        "--gpus",
        ",".join(requested_gpus),
        "--tag",
        args.tag,
    ]

    print(f"Prepared input: {case_dir_rel / input_variant}")
    print("Command:")
    print(" ".join(cmd))

    if not args.execute:
        print("Dry run only. Add --execute to launch the benchmark.")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(requested_gpus)
    result = subprocess.run(cmd, cwd=root, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
