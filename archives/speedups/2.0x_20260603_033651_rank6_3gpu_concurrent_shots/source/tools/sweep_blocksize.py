#!/usr/bin/env python3
"""Compile and benchmark CUDA block-size variants."""

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
MOD_RE = re.compile(r"mod time\s+([0-9.]+)s")


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
        "mod_times": [float(x) for x in MOD_RE.findall(text)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="perf_3gpu")
    parser.add_argument("--baseline-wp", type=float, default=3.491814)
    parser.add_argument("--gpus")
    parser.add_argument("--nv-arch", default="sm_89")
    parser.add_argument("--extra-nvflags", default="")
    parser.add_argument(
        "--variants",
        nargs="+",
        type=parse_variant,
        default=[
            (32, 8, 1),
            (64, 4, 1),
            (16, 16, 1),
            (32, 4, 2),
            (16, 8, 2),
            (16, 4, 4),
        ],
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = root / "benchmarks" / "reports" / f"blocksize_sweep_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=False)

    records = []
    for b1, b2, b3 in args.variants:
        tag = f"bs_{b1}x{b2}x{b3}"
        nvflags = (
            f"-O3 -arch={args.nv_arch} -D_gpu_par_ -Dradius=4 "
            f"-DBlockSize1={b1} -DBlockSize2={b2} -DBlockSize3={b3} "
            f"-DVBlockSize1={b1} -DVBlockSize2={b2} -DVBlockSize3={b3} "
            f"-DPBlockSize1={b1} -DPBlockSize2={b2} -DPBlockSize3={b3} "
            f"-DPCoreBlockSize1={b1} -DPCoreBlockSize2={b2} -DPCoreBlockSize3={b3} "
            f"-DPmlBlockSize1={b1} -DPmlBlockSize2={b2} -DPmlBlockSize3={b3} "
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
            "block_size": [b1, b2, b3],
            "threads_per_block": b1 * b2 * b3,
            "nvflags": nvflags,
            "build_returncode": build_rc,
            "build_log": str(build_log.relative_to(root)),
        }
        if build_rc != 0:
            records.append(record)
            continue

        run_log = report_dir / f"{tag}.run_benchmark.log"
        run_cmd = ["python3", "tools/run_benchmark.py", "--case", args.case, "--tag", tag]
        if args.gpus:
            run_cmd.extend(["--gpus", args.gpus])
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

    summary_json = report_dir / "summary.json"
    summary_json.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# Block Size Sweep", "", "| Variant | Threads | WP seconds | Speedup | Run dir |", "|---|---:|---:|---:|---|"]
    for r in sorted(records, key=lambda x: x.get("wp_seconds") or 1e99):
        lines.append(
            f"| {r['tag']} | {r['threads_per_block']} | "
            f"{r.get('wp_seconds', float('nan')):.6f} | "
            f"{r.get('speedup', float('nan')):.3f} | {r.get('run_dir', '')} |"
        )
    summary_md = report_dir / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_dir)
    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
