#!/usr/bin/env python3
"""Summarize CUDA3D host/setup timer probe logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_elapsed(log: str) -> float | None:
    match = re.search(r"Elapsed \(wall clock\) time.*?:\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", log)
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_key_values(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([0-9.eE+-]+)", text):
        values[key] = float(value)
    return values


def parse_log(path: Path) -> dict[str, Any]:
    log = path.read_text(encoding="utf-8", errors="replace")
    timers: dict[str, dict[str, float]] = {}
    for line in log.splitlines():
        match = re.match(r"HOST_SETUP_TIMER\s+(\w+)\s+(.*)", line.strip())
        if not match:
            continue
        section = match.group(1)
        timers.setdefault(section, {}).update(parse_key_values(match.group(2)))

    gradient_match = re.search(
        r"Gradient TIME all=\s*([0-9.]+)s, WP computing time =\s*([0-9.]+)s, read time =\s*([0-9.]+)s",
        log,
    )
    gradient = float(gradient_match.group(1)) if gradient_match else None
    wp = float(gradient_match.group(2)) if gradient_match else None
    read_time = float(gradient_match.group(3)) if gradient_match else None
    elapsed = parse_elapsed(log)

    main = timers.get("main", {})
    cal = timers.get("cal", {})
    measured_pre_gradient = main.get("total_pre_gradient", 0.0) + cal.get("pre_gradient_init", 0.0)
    elapsed_minus_gradient = elapsed - gradient if elapsed is not None and gradient is not None else None
    unaccounted = (
        elapsed_minus_gradient - measured_pre_gradient
        if elapsed_minus_gradient is not None
        else None
    )
    return {
        "run_log": str(path),
        "elapsed_s": elapsed,
        "gradient_s": gradient,
        "wp_s": wp,
        "read_time_s": read_time,
        "timers": timers,
        "measured_pre_gradient_s": measured_pre_gradient,
        "elapsed_minus_gradient_s": elapsed_minus_gradient,
        "unaccounted_elapsed_minus_gradient_s": unaccounted,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    main = summary["timers"].get("main", {})
    cal = summary["timers"].get("cal", {})
    elapsed = summary["elapsed_s"]
    gradient = summary["gradient_s"]
    lines = [
        "# Host / Setup Timer Probe",
        "",
        "## Program Timing",
        "",
        f"- elapsed: `{elapsed:.3f}s`" if elapsed is not None else "- elapsed: `n/a`",
        f"- Gradient TIME all: `{gradient:.6f}s`" if gradient is not None else "- Gradient TIME all: `n/a`",
        f"- WP computing time: `{summary['wp_s']:.6f}s`" if summary["wp_s"] is not None else "- WP computing time: `n/a`",
        f"- elapsed - Gradient: `{summary['elapsed_minus_gradient_s']:.6f}s`" if summary["elapsed_minus_gradient_s"] is not None else "- elapsed - Gradient: `n/a`",
        "",
        "## Main Setup Timers",
        "",
        "| stage | seconds |",
        "| --- | ---: |",
    ]
    for key in [
        "input_scan",
        "gpu_setup",
        "input_bcast",
        "coeff_init",
        "static_alloc",
        "root_model_read",
        "model_bcast",
        "acqui_read",
        "acqui_bcast",
        "lint",
        "shot_list",
        "total_pre_gradient",
        "gradient_call_total",
        "post_gradient_barrier_and_free",
        "total_after_mpi_to_pre_finalize",
    ]:
        if key in main:
            lines.append(f"| `{key}` | `{main[key]:.6f}` |")
    lines += [
        "",
        "## Cal Setup Timers",
        "",
        "| stage | seconds |",
        "| --- | ---: |",
    ]
    for key in ["pre_gradient_init"]:
        if key in cal:
            lines.append(f"| `{key}` | `{cal[key]:.6f}` |")
    lines += [
        "",
        "## Accounting",
        "",
        f"- measured pre-Gradient setup: `{summary['measured_pre_gradient_s']:.6f}s`",
    ]
    if summary["unaccounted_elapsed_minus_gradient_s"] is not None:
        lines.append(
            f"- unaccounted elapsed-minus-Gradient: `{summary['unaccounted_elapsed_minus_gradient_s']:.6f}s`"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    summary = parse_log(Path(args.run_log))
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
