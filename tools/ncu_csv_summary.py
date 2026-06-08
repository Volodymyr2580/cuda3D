#!/usr/bin/env python3
"""Summarize selected Nsight Compute CSV metrics by CUDA kernel."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


METRICS = {
    ("GPU Speed Of Light Throughput", "Duration"): "duration_ns",
    ("GPU Speed Of Light Throughput", "Memory Throughput"): "sol_memory_throughput_pct",
    ("GPU Speed Of Light Throughput", "DRAM Throughput"): "sol_dram_throughput_pct",
    ("GPU Speed Of Light Throughput", "L1/TEX Cache Throughput"): "sol_l1tex_throughput_pct",
    ("GPU Speed Of Light Throughput", "L2 Cache Throughput"): "sol_l2_throughput_pct",
    ("GPU Speed Of Light Throughput", "Compute (SM) Throughput"): "sol_compute_throughput_pct",
    ("Memory Workload Analysis", "Memory Throughput"): "memory_throughput_bytes_per_s",
    ("Memory Workload Analysis", "Mem Busy"): "mem_busy_pct",
    ("Memory Workload Analysis", "L1/TEX Hit Rate"): "l1tex_hit_rate_pct",
    ("Memory Workload Analysis", "L2 Hit Rate"): "l2_hit_rate_pct",
    ("Memory Workload Analysis", "Mem Pipes Busy"): "mem_pipes_busy_pct",
    ("Scheduler Statistics", "One or More Eligible"): "one_or_more_eligible_pct",
    ("Scheduler Statistics", "No Eligible"): "no_eligible_pct",
    ("Scheduler Statistics", "Issued Warp Per Scheduler"): "issued_warp_per_scheduler",
    ("Scheduler Statistics", "Active Warps Per Scheduler"): "active_warps_per_scheduler",
    ("Scheduler Statistics", "Eligible Warps Per Scheduler"): "eligible_warps_per_scheduler",
    ("Warp State Statistics", "Warp Cycles Per Issued Instruction"): "warp_cycles_per_issued_instruction",
    ("Warp State Statistics", "Avg. Active Threads Per Warp"): "avg_active_threads_per_warp",
    ("Warp State Statistics", "Avg. Not Predicated Off Threads Per Warp"): "avg_not_predicated_off_threads_per_warp",
    ("Source Counters", "Branch Efficiency"): "branch_efficiency_pct",
    ("Source Counters", "Branch Instructions"): "branch_instructions",
    ("Source Counters", "Avg. Divergent Branches"): "avg_divergent_branches",
    ("Occupancy", "Theoretical Occupancy"): "theoretical_occupancy_pct",
    ("Occupancy", "Achieved Occupancy"): "achieved_occupancy_pct",
    ("Occupancy", "Achieved Active Warps Per SM"): "achieved_active_warps_per_sm",
    ("Occupancy", "Block Limit Registers"): "block_limit_registers",
    ("Occupancy", "Block Limit Shared Mem"): "block_limit_shared_mem",
}


@dataclass(frozen=True)
class ProfileInput:
    label: str
    path: Path


def parse_number(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text or text == "no data":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def short_kernel(name: str) -> str:
    match = re.match(r"([A-Za-z0-9_]+)\(", name)
    return match.group(1) if match else name


def read_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    csv_lines = [line for line in text if line.startswith('"')]
    return list(csv.DictReader(csv_lines))


def summarize_profile(item: ProfileInput) -> dict[str, object]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rules: dict[str, list[dict[str, str]]] = defaultdict(list)
    launches: dict[str, set[str]] = defaultdict(set)

    for row in read_rows(item.path):
        kernel = short_kernel(row.get("Kernel Name", "unknown"))
        launches[kernel].add(row.get("ID", ""))
        section = row.get("Section Name", "")
        metric = row.get("Metric Name", "")
        alias = METRICS.get((section, metric))
        if alias:
            number = parse_number(row.get("Metric Value", ""))
            if number is not None:
                values[kernel][alias].append(number)
        rule_name = row.get("Rule Name", "")
        if rule_name in {"SOLBottleneck", "CPIStall", "IssueSlotUtilization", "ThreadDivergence"}:
            desc = row.get("Rule Description", "")
            if desc:
                rules[kernel].append(
                    {
                        "section": section,
                        "rule": rule_name,
                        "type": row.get("Rule Type", ""),
                        "estimated_speedup_type": row.get("Estimated Speedup Type", ""),
                        "estimated_speedup": row.get("Estimated Speedup", ""),
                        "description": desc,
                    }
                )

    kernels = {}
    for kernel in sorted(values):
        kernels[kernel] = {
            "launches": len(launches[kernel]),
            "metrics": {name: mean(nums) for name, nums in sorted(values[kernel].items())},
            "rules": rules.get(kernel, [])[:4],
        }
    return {"label": item.label, "path": str(item.path), "kernels": kernels}


def format_float(value: object, suffix: str = "") -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if abs(value) >= 1000:
        text = f"{value:,.0f}"
    else:
        text = f"{value:.3f}"
    return f"{text}{suffix}"


def render_markdown(summary: dict[str, object]) -> str:
    profiles = summary["profiles"]
    lines = [
        "# NCU CSV Summary",
        "",
        "## Profiles",
        "",
    ]
    for profile in profiles:
        lines.append(f"- `{profile['label']}`: `{profile['path']}`")

    metric_rows = [
        ("duration_ns", "Duration", " ns"),
        ("sol_compute_throughput_pct", "SOL compute", "%"),
        ("sol_memory_throughput_pct", "SOL memory", "%"),
        ("sol_dram_throughput_pct", "SOL DRAM", "%"),
        ("mem_pipes_busy_pct", "Mem pipes busy", "%"),
        ("l1tex_hit_rate_pct", "L1/TEX hit", "%"),
        ("l2_hit_rate_pct", "L2 hit", "%"),
        ("no_eligible_pct", "No eligible", "%"),
        ("issued_warp_per_scheduler", "Issued warp/scheduler", ""),
        ("active_warps_per_scheduler", "Active warps/scheduler", ""),
        ("eligible_warps_per_scheduler", "Eligible warps/scheduler", ""),
        ("warp_cycles_per_issued_instruction", "Warp cycles/issued inst", ""),
        ("avg_active_threads_per_warp", "Avg active threads/warp", ""),
        ("avg_not_predicated_off_threads_per_warp", "Avg not-predicated threads/warp", ""),
        ("branch_efficiency_pct", "Branch efficiency", "%"),
        ("branch_instructions", "Branch instructions", ""),
        ("avg_divergent_branches", "Avg divergent branches", ""),
        ("achieved_occupancy_pct", "Achieved occupancy", "%"),
    ]

    all_kernels = sorted({k for profile in profiles for k in profile["kernels"]})
    for kernel in all_kernels:
        lines.extend(["", f"## `{kernel}`", ""])
        header = "| metric | " + " | ".join(profile["label"] for profile in profiles) + " |"
        lines.append(header)
        lines.append("| --- | " + " | ".join("---:" for _ in profiles) + " |")
        for key, label, suffix in metric_rows:
            cells = []
            for profile in profiles:
                metrics = profile["kernels"].get(kernel, {}).get("metrics", {})
                cells.append(format_float(metrics.get(key), suffix))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Rules"])
    for profile in profiles:
        for kernel, data in profile["kernels"].items():
            for rule in data.get("rules", [])[:2]:
                desc = rule["description"].replace("\n", " ")
                if len(desc) > 280:
                    desc = desc[:277] + "..."
                lines.append(f"- `{profile['label']}` `{kernel}` `{rule['rule']}`: {desc}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="append", nargs=2, metavar=("LABEL", "CSV"), required=True)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    profiles = [summarize_profile(ProfileInput(label, Path(path))) for label, path in args.profile]
    summary = {"profiles": profiles}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = render_markdown(summary)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(md)


if __name__ == "__main__":
    main()
