#!/usr/bin/env python3
"""Summarize Nsight Systems CUDA API/kernel CSV reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_number(text: str) -> float:
    return float(text.strip().replace(",", ""))


def short_kernel(name: str) -> str:
    match = re.match(r"([A-Za-z0-9_]+)\(", name)
    return match.group(1) if match else name


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text(encoding="utf-8", errors="replace").splitlines()))


def parse_run_log(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, float] = {}
    match = re.search(r"Gradient TIME all=\s*([0-9.]+)s,\s*WP computing time\s*=\s*([0-9.]+)s", text)
    if match:
        result["gradient_time_s"] = float(match.group(1))
        result["wp_time_s"] = float(match.group(2))
    return result


def summarize(api_csv: Path, kernel_csv: Path, run_log: Path | None) -> dict[str, object]:
    api_rows = read_csv(api_csv)
    kernel_rows = read_csv(kernel_csv)
    run = parse_run_log(run_log)

    api = []
    for row in api_rows:
        api.append(
            {
                "name": row["Name"],
                "time_pct": parse_number(row["Time (%)"]),
                "total_time_s": parse_number(row["Total Time (ns)"]) / 1e9,
                "num_calls": int(parse_number(row["Num Calls"])),
                "avg_us": parse_number(row["Avg (ns)"]) / 1e3,
                "median_us": parse_number(row["Med (ns)"]) / 1e3,
            }
        )

    kernels = []
    for row in kernel_rows:
        kernels.append(
            {
                "name": short_kernel(row["Name"]),
                "full_name": row["Name"],
                "time_pct": parse_number(row["Time (%)"]),
                "total_time_s": parse_number(row["Total Time (ns)"]) / 1e9,
                "instances": int(parse_number(row["Instances"])),
                "avg_us": parse_number(row["Avg (ns)"]) / 1e3,
                "median_us": parse_number(row["Med (ns)"]) / 1e3,
            }
        )

    kernel_total_s = sum(float(item["total_time_s"]) for item in kernels)
    launch = next((item for item in api if item["name"] == "cudaLaunchKernel"), None)
    wp_time_s = run.get("wp_time_s")
    gap_s = None
    gap_fraction = None
    ideal_gap_speedup = None
    if wp_time_s is not None and kernel_total_s > 0:
        gap_s = wp_time_s - kernel_total_s
        gap_fraction = gap_s / wp_time_s
        ideal_gap_speedup = wp_time_s / kernel_total_s if kernel_total_s > 0 else None

    return {
        "run": run,
        "api": api,
        "kernels": kernels,
        "totals": {
            "kernel_total_s": kernel_total_s,
            "cuda_launch_total_s": launch["total_time_s"] if launch else None,
            "cuda_launch_calls": launch["num_calls"] if launch else None,
            "cuda_launch_avg_us": launch["avg_us"] if launch else None,
            "wp_minus_kernel_total_s": gap_s,
            "wp_minus_kernel_total_fraction": gap_fraction,
            "ideal_gap_elimination_speedup": ideal_gap_speedup,
        },
    }


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary: dict[str, object]) -> str:
    totals = summary["totals"]
    run = summary["run"]
    lines = [
        "# Nsight Systems CUDA Summary",
        "",
        "## Run",
        "",
        f"- WP computing time: `{fmt(run.get('wp_time_s'))}s`",
        f"- Gradient TIME all: `{fmt(run.get('gradient_time_s'))}s`",
        f"- GPU kernel total: `{fmt(totals.get('kernel_total_s'))}s`",
        f"- WP minus GPU kernel total: `{fmt(totals.get('wp_minus_kernel_total_s'))}s`",
        f"- WP minus GPU kernel total fraction: `{fmt(totals.get('wp_minus_kernel_total_fraction'), 4)}`",
        f"- Ideal speedup if that gap vanished: `{fmt(totals.get('ideal_gap_elimination_speedup'), 6)}x`",
        f"- cudaLaunchKernel CPU API total: `{fmt(totals.get('cuda_launch_total_s'))}s`",
        f"- cudaLaunchKernel calls: `{totals.get('cuda_launch_calls')}`",
        f"- cudaLaunchKernel avg: `{fmt(totals.get('cuda_launch_avg_us'), 3)}us`",
        "",
        "## Kernels",
        "",
        "| kernel | total s | instances | avg us | share |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["kernels"]:
        lines.append(
            f"| `{item['name']}` | {float(item['total_time_s']):.6f} | "
            f"{item['instances']} | {float(item['avg_us']):.3f} | {float(item['time_pct']):.1f}% |"
        )

    lines.extend([
        "",
        "## CUDA API",
        "",
        "| API | total s | calls | avg us | share |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for item in summary["api"][:8]:
        lines.append(
            f"| `{item['name']}` | {float(item['total_time_s']):.6f} | "
            f"{item['num_calls']} | {float(item['avg_us']):.3f} | {float(item['time_pct']):.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-csv", required=True)
    parser.add_argument("--kernel-csv", required=True)
    parser.add_argument("--run-log")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    summary = summarize(
        Path(args.api_csv),
        Path(args.kernel_csv),
        Path(args.run_log) if args.run_log else None,
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = render_markdown(summary)
    if args.md_out:
        Path(args.md_out).write_text(markdown, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(markdown)


if __name__ == "__main__":
    main()
