#!/usr/bin/env python3
"""Compile and benchmark V-PML-only CUDA block-size variants."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


WP_RE = re.compile(r"WP computing time\s*=\s*([0-9.]+)s")
GRAD_RE = re.compile(r"Gradient TIME all=\s*([0-9.]+)s")


def parse_variant(text: str) -> tuple[int, int, int]:
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected B1xB2xB3, got {text!r}")
    values = tuple(int(p) for p in parts)
    threads = values[0] * values[1] * values[2]
    if threads <= 0 or threads > 1024:
        raise argparse.ArgumentTypeError(f"Invalid CUDA block thread count {threads} for {text!r}")
    return values


def run_command(command: list[str], cwd: Path, log_path: Path) -> int:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    return result.returncode


def parse_times(run_log: Path) -> dict:
    text = run_log.read_text(encoding="utf-8", errors="replace")
    wp = WP_RE.search(text)
    grad = GRAD_RE.search(text)
    return {
        "wp_seconds": float(wp.group(1)) if wp else None,
        "gradient_seconds": float(grad.group(1)) if grad else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="profile_1gpu")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--baseline-wp", type=float, default=3.491814)
    parser.add_argument("--nv-arch", default="sm_89")
    parser.add_argument("--extra-nvflags", default="--use_fast_math")
    parser.add_argument(
        "--variants",
        nargs="+",
        type=parse_variant,
        default=[
            (128, 2, 1),
            (64, 4, 1),
            (64, 2, 1),
            (64, 1, 2),
            (32, 8, 1),
            (32, 4, 2),
            (32, 2, 4),
            (16, 16, 1),
            (16, 8, 2),
            (8, 16, 2),
        ],
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = root / "benchmarks" / "reports" / f"v_blocksize_sweep_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=False)

    records = []
    for b1, b2, b3 in args.variants:
        tag = f"v_{b1}x{b2}x{b3}"
        nvflags = (
            f"-O3 -arch={args.nv_arch} -D_gpu_par_ -Dradius=4 "
            "-DBlockSize1=128 -DBlockSize2=2 -DBlockSize3=1 "
            f"-DVBlockSize1={b1} -DVBlockSize2={b2} -DVBlockSize3={b3} "
            "-DPBlockSize1=128 -DPBlockSize2=2 -DPBlockSize3=1 "
            "-DPCoreBlockSize1=128 -DPCoreBlockSize2=2 -DPCoreBlockSize3=1 "
            "-DPmlBlockSize1=128 -DPmlBlockSize2=2 -DPmlBlockSize3=1 "
            "-DBlockSize=256 -DMAX_BLOCK_SZ=256"
        )
        if args.extra_nvflags:
            nvflags = f"{nvflags} {args.extra_nvflags}"
        build_cmd = (
            "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; "
            f"make -B -f makefile.server test NVFLAGS={shlex.quote(nvflags)}"
        )
        build_log = report_dir / f"{tag}.build.log"
        build_rc = run_command(["bash", "-lc", build_cmd], src_dir, build_log)
        record = {
            "tag": tag,
            "v_block_size": [b1, b2, b3],
            "threads_per_block": b1 * b2 * b3,
            "nvflags": nvflags,
            "build_returncode": build_rc,
            "build_log": str(build_log.relative_to(root)),
        }
        if build_rc != 0:
            records.append(record)
            continue

        run_log = report_dir / f"{tag}.run_benchmark.log"
        run_cmd = [
            "python3",
            "tools/run_benchmark.py",
            "--case",
            args.case,
            "--tag",
            tag,
            "--gpus",
            args.gpus,
        ]
        run_rc = run_command(run_cmd, root, run_log)
        record["run_returncode"] = run_rc
        record["run_benchmark_log"] = str(run_log.relative_to(root))
        if run_rc == 0:
            run_dir = run_log.read_text(encoding="utf-8").splitlines()[0].strip()
            record["run_dir"] = run_dir
            times = parse_times(Path(run_dir) / "run.log")
            record.update(times)
            if record["wp_seconds"]:
                record["speedup"] = args.baseline_wp / record["wp_seconds"]
        records.append(record)

    (report_dir / "summary.json").write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# V-PML Block Size Sweep", "", "| Variant | Threads | WP seconds | Gradient seconds | Run dir |", "|---|---:|---:|---:|---|"]
    for r in sorted(records, key=lambda x: x.get("wp_seconds") or 1e99):
        wp = r.get("wp_seconds")
        grad = r.get("gradient_seconds")
        lines.append(
            f"| {r['tag']} | {r['threads_per_block']} | "
            f"{wp:.6f} | {grad:.6f} | {r.get('run_dir', '')} |"
            if wp is not None and grad is not None
            else f"| {r['tag']} | {r['threads_per_block']} | nan | nan | {r.get('run_dir', '')} |"
        )
    summary_md = report_dir / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_dir)
    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
