#!/usr/bin/env python3
"""Budget post-len16 pressure-PML compact descriptor candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pml_active_segment_compaction_model as active_model  # noqa: E402


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.2%}"


def ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}x"


def ceil_to(value: int, group: int) -> int:
    return ((value + group - 1) // group) * group


def sampled_speedup(kernel_speedup: float, kernel_share: float) -> float:
    return 1.0 / ((1.0 - kernel_share) + kernel_share / kernel_speedup)


def profile_metrics(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    profiles = {item["label"]: item for item in data["profiles"]}

    def duration_us(label: str, kernel: str) -> float:
        return profiles[label]["kernels"][kernel]["metrics"]["duration_ns"] / 1000.0

    direct_p = duration_us("directfill", "cuda_fd3d_p_pml_tile_ns")
    direct_core = duration_us("directfill", "cuda_fd3d_p_core_ns")
    direct_v = duration_us("directfill", "cuda_fd3d_v_pml_tile_ns")
    len16_residual = duration_us("len16", "cuda_fd3d_p_pml_tile_ns")
    len16_packed = duration_us("len16", "cuda_fd3d_p_pml_len16_halfwarp_ns")
    len16_core = duration_us("len16", "cuda_fd3d_p_core_ns")
    len16_v = duration_us("len16", "cuda_fd3d_v_pml_tile_ns")
    len16_p = len16_residual + len16_packed
    direct_main = direct_p + direct_core + direct_v
    len16_main = len16_p + len16_core + len16_v
    return {
        "directfill": {
            "p_pml_us": direct_p,
            "p_core_us": direct_core,
            "v_pml_us": direct_v,
            "sampled_main_us": direct_main,
        },
        "len16": {
            "p_pml_residual_us": len16_residual,
            "p_pml_packed_us": len16_packed,
            "p_pml_total_us": len16_p,
            "p_core_us": len16_core,
            "v_pml_us": len16_v,
            "sampled_main_us": len16_main,
            "p_pml_share": len16_p / len16_main,
        },
        "observed": {
            "p_pml_speedup_direct_to_len16": direct_p / len16_p,
            "sampled_main_speedup_direct_to_len16": direct_main / len16_main,
        },
    }


def compute_budget(active: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    hist = {int(k): int(v) for k, v in active["length_hist"].items()}
    active_lanes = int(active["totals"]["active_lanes"])
    current_lanes = int(active["totals"]["current_lanes"])
    length16_lines = hist.get(16, 0)
    length23_lines = hist.get(23, 0)
    length32_lines = hist.get(32, 0)
    length23_active = length23_lines * 23
    length23_inactive = length23_lines * (32 - 23)

    accepted_len16_lanes = ceil_to(length16_lines, 2) * 16 + (length23_lines + length32_lines) * 32
    exact_all_lanes = ceil_to(active_lanes, 256)
    exact_len23_only_lanes = (
        ceil_to(length16_lines, 2) * 16
        + ceil_to(length23_active, 256)
        + length32_lines * 32
    )
    line_descriptor_lanes = accepted_len16_lanes

    direct_to_len16_lane_ceiling = current_lanes / accepted_len16_lanes
    observed_direct_to_len16 = profile["observed"]["p_pml_speedup_direct_to_len16"]
    lane_to_time_efficiency = observed_direct_to_len16 / direct_to_len16_lane_ceiling

    candidates: dict[str, dict[str, Any]] = {}

    def add_candidate(name: str, lanes: int, descriptor_bytes: int, description: str) -> None:
        p_speedup = accepted_len16_lanes / lanes
        sampled_ceiling = sampled_speedup(p_speedup, profile["len16"]["p_pml_share"])
        calibrated_p_speedup = 1.0 + max(0.0, (p_speedup - 1.0) * lane_to_time_efficiency)
        calibrated_sampled = sampled_speedup(calibrated_p_speedup, profile["len16"]["p_pml_share"])
        candidates[name] = {
            "description": description,
            "lanes": lanes,
            "lane_reduction_vs_len16": 1.0 - lanes / accepted_len16_lanes,
            "p_pml_lane_speedup_ceiling_vs_len16": p_speedup,
            "sampled_main_speedup_ceiling_vs_len16": sampled_ceiling,
            "calibrated_p_pml_speedup_vs_len16": calibrated_p_speedup,
            "calibrated_sampled_main_speedup_vs_len16": calibrated_sampled,
            "descriptor_mib_per_step_aggregate_shots": descriptor_bytes / (1024.0 * 1024.0),
            "descriptor_bytes_per_saved_lane": (
                descriptor_bytes / (accepted_len16_lanes - lanes)
                if accepted_len16_lanes > lanes and descriptor_bytes > 0
                else None
            ),
        }

    add_candidate(
        "line_descriptor_len16_current",
        line_descriptor_lanes,
        int(active["totals"]["active_line_slots"]) * 8,
        "No further compaction; one compact line descriptor per active z-line.",
    )
    add_candidate(
        "exact_length23_points_only",
        exact_len23_only_lanes,
        length23_active * 4,
        "Keep accepted len16 packing and length32 full warps, compact only length-23 active points.",
    )
    add_candidate(
        "exact_all_active_points",
        exact_all_lanes,
        active_lanes * 4,
        "Compact every active pressure-PML point with one uint32 point descriptor.",
    )

    gate = {
        "decision": "reject_cuda_prototype",
        "reason": (
            "After the accepted len16 packing, exact active-point compaction can only remove the remaining "
            "length-23 inactive lanes.  The optimistic sampled-main ceiling is below the >=5% prototype gate, "
            "and the required descriptor stream risks adding more memory/control overhead than the saved inactive lanes."
        ),
        "next_allowed": [
            "source-level drill-down of cuda_fd3d_p_pml_len16_halfwarp_ns",
            "v-PML memory layout/coalescing design",
        ],
        "reopen_condition": "Only reopen compact descriptors if a new design shows >=5% perf_1gpu_6shots repeat speedup ceiling after descriptor/control overhead.",
    }

    return {
        "inputs": {
            "active_model": active["case"],
            "profile": profile,
        },
        "post_len16_lane_shape": {
            "accepted_len16_lanes": accepted_len16_lanes,
            "active_lanes": active_lanes,
            "length16_lines": length16_lines,
            "length23_lines": length23_lines,
            "length32_lines": length32_lines,
            "length23_active_lanes": length23_active,
            "length23_inactive_lanes": length23_inactive,
            "direct_to_len16_lane_ceiling": direct_to_len16_lane_ceiling,
            "observed_direct_to_len16_p_pml_speedup": observed_direct_to_len16,
            "lane_to_time_efficiency": lane_to_time_efficiency,
            "post_len16_p_pml_share": profile["len16"]["p_pml_share"],
        },
        "candidates": candidates,
        "gate": gate,
    }


def render_markdown(result: dict[str, Any]) -> str:
    shape = result["post_len16_lane_shape"]
    profile = result["inputs"]["profile"]
    lines = [
        "# PML Compact Descriptor Budget",
        "",
        "## Context",
        "",
        f"- accepted len16 lanes: `{shape['accepted_len16_lanes']}`",
        f"- active lanes: `{shape['active_lanes']}`",
        f"- post-len16 pressure-PML sampled-main share: `{pct(shape['post_len16_p_pml_share'])}`",
        f"- observed direct-fill -> len16 pressure-PML speedup: `{ratio(shape['observed_direct_to_len16_p_pml_speedup'])}`",
        f"- direct-fill -> len16 lane ceiling: `{ratio(shape['direct_to_len16_lane_ceiling'])}`",
        f"- observed lane-to-time efficiency factor: `{shape['lane_to_time_efficiency']:.3f}`",
        "",
        "## NCU Anchor",
        "",
        "| metric | direct-fill | len16 |",
        "| --- | ---: | ---: |",
        f"| pressure-PML total | `{profile['directfill']['p_pml_us']:.3f}us` | `{profile['len16']['p_pml_total_us']:.3f}us` |",
        f"| sampled main total | `{profile['directfill']['sampled_main_us']:.3f}us` | `{profile['len16']['sampled_main_us']:.3f}us` |",
        f"| p-core | `{profile['directfill']['p_core_us']:.3f}us` | `{profile['len16']['p_core_us']:.3f}us` |",
        f"| v-PML | `{profile['directfill']['v_pml_us']:.3f}us` | `{profile['len16']['v_pml_us']:.3f}us` |",
        "",
        "## Remaining Lane Opportunity",
        "",
        "| segment | lines | active lanes | inactive lanes after len16 |",
        "| --- | ---: | ---: | ---: |",
        f"| length 16 | {shape['length16_lines']} | {shape['length16_lines'] * 16} | 0 |",
        f"| length 23 | {shape['length23_lines']} | {shape['length23_active_lanes']} | {shape['length23_inactive_lanes']} |",
        f"| length 32 | {shape['length32_lines']} | {shape['length32_lines'] * 32} | 0 |",
        "",
        "## Candidate Ceilings Vs Accepted Len16",
        "",
        "| candidate | lanes | lane reduction | p-PML lane ceiling | sampled-main ceiling | calibrated sampled-main | descriptor MiB/step | bytes/saved lane |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in result["candidates"].items():
        bps = item["descriptor_bytes_per_saved_lane"]
        bps_text = "n/a" if bps is None else f"{bps:.2f}"
        lines.append(
            f"| `{name}` | {item['lanes']} | {pct(item['lane_reduction_vs_len16'])} | "
            f"{ratio(item['p_pml_lane_speedup_ceiling_vs_len16'])} | "
            f"{ratio(item['sampled_main_speedup_ceiling_vs_len16'])} | "
            f"{ratio(item['calibrated_sampled_main_speedup_vs_len16'])} | "
            f"{item['descriptor_mib_per_step_aggregate_shots']:.3f} | {bps_text} |"
        )

    gate = result["gate"]
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- reason: {gate['reason']}",
            f"- reopen condition: {gate['reopen_condition']}",
            "",
            "Allowed next directions:",
            "",
        ]
    )
    for item in gate["next_allowed"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="benchmarks/cases/perf_1gpu_6shots")
    parser.add_argument("--directfill-ncu-summary", default="reports/day_20260608/directfill_combo_ncu_20260608_120449_summary.json")
    parser.add_argument("--len16-ncu-summary", default="reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case)
    if not case_dir.is_absolute():
        case_dir = root / case_dir
    direct_ncu = Path(args.directfill_ncu_summary)
    if not direct_ncu.is_absolute():
        direct_ncu = root / direct_ncu
    len16_ncu = Path(args.len16_ncu_summary)
    if not len16_ncu.is_absolute():
        len16_ncu = root / len16_ncu

    active = active_model.analyze_segments(case_dir, direct_ncu, 32, 4, 2, 4)
    profile = profile_metrics(len16_ncu)
    result = compute_budget(active, profile)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_markdown(result)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(md)


if __name__ == "__main__":
    main()
