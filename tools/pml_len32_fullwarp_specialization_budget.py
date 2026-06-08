#!/usr/bin/env python3
"""Budget pressure-PML length-32 full-warp specialization after len16 packing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def pct(value: float) -> str:
    return f"{value:.2%}"


def ratio(value: float) -> str:
    return f"{value:.4f}x"


def sampled_speedup(sampled_main_us: float, saved_us: float) -> float:
    remaining = sampled_main_us - saved_us
    if remaining <= 0:
        return math.inf
    return sampled_main_us / remaining


def required_local_speedup(local_time_us: float, target_saved_us: float) -> float | None:
    if target_saved_us <= 0:
        return 1.0
    if target_saved_us >= local_time_us:
        return None
    return local_time_us / (local_time_us - target_saved_us)


def load_len16_profile(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for profile in data["profiles"]:
        if profile["label"] != "len16":
            continue
        kernels = profile["kernels"]
        p_core = kernels["cuda_fd3d_p_core_ns"]["metrics"]["duration_ns"] / 1000.0
        v_pml = kernels["cuda_fd3d_v_pml_tile_ns"]["metrics"]["duration_ns"] / 1000.0
        residual = kernels["cuda_fd3d_p_pml_tile_ns"]["metrics"]["duration_ns"] / 1000.0
        packed = kernels["cuda_fd3d_p_pml_len16_halfwarp_ns"]["metrics"]["duration_ns"] / 1000.0
        residual_metrics = kernels["cuda_fd3d_p_pml_tile_ns"]["metrics"]
        return {
            "p_core_us": p_core,
            "v_pml_us": v_pml,
            "p_pml_residual_us": residual,
            "p_pml_packed_us": packed,
            "p_pml_total_us": residual + packed,
            "sampled_main_us": p_core + v_pml + residual + packed,
            "residual_branch_efficiency_pct": residual_metrics["branch_efficiency_pct"],
            "residual_avg_active_threads_per_warp": residual_metrics["avg_active_threads_per_warp"],
            "residual_avg_not_predicated_threads_per_warp": residual_metrics[
                "avg_not_predicated_off_threads_per_warp"
            ],
            "residual_no_eligible_pct": residual_metrics["no_eligible_pct"],
            "residual_eligible_warps_per_scheduler": residual_metrics["eligible_warps_per_scheduler"],
        }
    raise ValueError(f"no len16 profile in {path}")


def analyze(active_budget_path: Path, profile_path: Path, target_speedup: float, control_proxy: float) -> dict[str, Any]:
    active_budget = json.loads(active_budget_path.read_text(encoding="utf-8"))
    shape = active_budget["post_len16_lane_shape"]
    profile = load_len16_profile(profile_path)

    len23_lines = int(shape["length23_lines"])
    len32_lines = int(shape["length32_lines"])
    len23_active = int(shape["length23_active_lanes"])
    len32_active = len32_lines * 32
    len23_launched = len23_lines * 32
    len32_launched = len32_lines * 32
    residual_launched = len23_launched + len32_launched
    residual_active = len23_active + len32_active

    full32_line_share = len32_lines / (len23_lines + len32_lines)
    full32_active_share = len32_active / residual_active
    full32_launched_share = len32_launched / residual_launched

    sampled_main_us = profile["sampled_main_us"]
    residual_us = profile["p_pml_residual_us"]
    target_saved_us = sampled_main_us * (1.0 - 1.0 / target_speedup)

    full32_time_line_us = residual_us * full32_line_share
    full32_time_active_us = residual_us * full32_active_share
    req_line = required_local_speedup(full32_time_line_us, target_saved_us)
    req_active = required_local_speedup(full32_time_active_us, target_saved_us)

    residual_branch_ineff = max(0.0, 1.0 - profile["residual_branch_efficiency_pct"] / 100.0)
    scenarios = {
        "control_proxy_on_full32_line_time": {
            "saved_us": full32_time_line_us * control_proxy,
            "description": "Apply direct-fill source-visible address/control proxy only to estimated length-32 time.",
        },
        "control_proxy_on_entire_residual": {
            "saved_us": residual_us * control_proxy,
            "description": "Apply direct-fill source-visible address/control proxy to the entire residual kernel.",
        },
        "perfect_branch_efficiency_on_full32_active_time": {
            "saved_us": full32_time_active_us * residual_branch_ineff,
            "description": "Generously treat residual branch inefficiency as removable time on length-32 work.",
        },
        "perfect_branch_efficiency_on_entire_residual": {
            "saved_us": residual_us * residual_branch_ineff,
            "description": "Utopian: eliminate the full residual branch-efficiency gap.",
        },
        "twenty_percent_full32_speedup": {
            "saved_us": full32_time_active_us * 0.20,
            "description": "Generous full-line specialization estimate without lane or state-traffic reduction.",
        },
    }
    for scenario in scenarios.values():
        scenario["sampled_main_speedup"] = sampled_speedup(sampled_main_us, scenario["saved_us"])
        scenario["meets_gate"] = scenario["sampled_main_speedup"] >= target_speedup

    gate = {
        "decision": "reject_cuda_prototype",
        "candidate": "CUDA3D_PML_PRESSURE_LEN32_FULL_WARP_SPECIALIZE",
        "reason": (
            "Length-32 residual work has no inactive-lane saving after len16 packing.  "
            "A separate full-warp kernel would need about 1.32x-1.35x local speedup to move "
            "sampled-main by 5%, while existing branch/control proxies and even a generous "
            "perfect-branch-efficiency scenario remain below the gate."
        ),
        "reopen_condition": (
            "Only reopen if a source-level profile separates length-32 residual work and proves "
            "that removable branch/control or memory-ownership overhead is at least the required "
            "local reduction after extra launch and tile-list overhead."
        ),
    }

    return {
        "inputs": {
            "active_budget_json": str(active_budget_path),
            "profile_json": str(profile_path),
            "target_sampled_main_speedup": target_speedup,
            "control_proxy_fraction": control_proxy,
        },
        "profile": profile,
        "residual_shape": {
            "length23_lines": len23_lines,
            "length32_lines": len32_lines,
            "length23_active_lanes": len23_active,
            "length32_active_lanes": len32_active,
            "length23_launched_lanes": len23_launched,
            "length32_launched_lanes": len32_launched,
            "full32_line_share": full32_line_share,
            "full32_active_lane_share": full32_active_share,
            "full32_launched_lane_share": full32_launched_share,
        },
        "target": {
            "sampled_main_us": sampled_main_us,
            "target_saved_us": target_saved_us,
            "estimated_full32_time_us_line_share": full32_time_line_us,
            "estimated_full32_time_us_active_share": full32_time_active_us,
            "required_full32_speedup_line_share": req_line,
            "required_full32_speedup_active_share": req_active,
            "required_full32_time_reduction_line_share": None if req_line is None else 1.0 - 1.0 / req_line,
            "required_full32_time_reduction_active_share": None if req_active is None else 1.0 - 1.0 / req_active,
        },
        "scenarios": scenarios,
        "gate": gate,
    }


def render_markdown(result: dict[str, Any]) -> str:
    profile = result["profile"]
    shape = result["residual_shape"]
    target = result["target"]
    lines = [
        "# Pressure-PML Len32 Full-Warp Specialization Budget",
        "",
        "## Context",
        "",
        f"- sampled main: `{profile['sampled_main_us']:.3f}us`",
        f"- residual pressure-PML: `{profile['p_pml_residual_us']:.3f}us`",
        f"- packed len16 pressure-PML: `{profile['p_pml_packed_us']:.3f}us`",
        f"- target sampled-main speedup gate: `{ratio(result['inputs']['target_sampled_main_speedup'])}`",
        f"- target saved time: `{target['target_saved_us']:.3f}us`",
        "",
        "## Residual Shape After Len16",
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| length-23 lines | `{shape['length23_lines']}` |",
        f"| length-32 lines | `{shape['length32_lines']}` |",
        f"| length-32 line share | `{pct(shape['full32_line_share'])}` |",
        f"| length-32 active-lane share | `{pct(shape['full32_active_lane_share'])}` |",
        f"| length-32 launched-lane share | `{pct(shape['full32_launched_lane_share'])}` |",
        "",
        "## Required Speedup",
        "",
        "| estimate basis | full32 time | required local speedup | required local reduction |",
        "| --- | ---: | ---: | ---: |",
        f"| line share | `{target['estimated_full32_time_us_line_share']:.3f}us` | `{ratio(target['required_full32_speedup_line_share'])}` | `{pct(target['required_full32_time_reduction_line_share'])}` |",
        f"| active-lane share | `{target['estimated_full32_time_us_active_share']:.3f}us` | `{ratio(target['required_full32_speedup_active_share'])}` | `{pct(target['required_full32_time_reduction_active_share'])}` |",
        "",
        "## Scenario Ceilings",
        "",
        "| scenario | saved | sampled-main speedup | gate |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, item in result["scenarios"].items():
        gate = "pass" if item["meets_gate"] else "reject"
        lines.append(f"| `{name}` | `{item['saved_us']:.3f}us` | `{ratio(item['sampled_main_speedup'])}` | `{gate}` |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- decision: `{result['gate']['decision']}`",
            f"- candidate: `{result['gate']['candidate']}`",
            f"- reason: {result['gate']['reason']}",
            f"- reopen condition: {result['gate']['reopen_condition']}",
            "",
            "## Boundary",
            "",
            "This rejects a branch/control-only `len32` pressure-PML split.  It does not reject a future design that removes real memory traffic or proves a different ownership model with profiler evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--active-budget-json",
        default="reports/day_20260608/pml_compact_descriptor_budget.json",
    )
    parser.add_argument(
        "--profile-json",
        default="reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json",
    )
    parser.add_argument("--target-speedup", type=float, default=1.05)
    parser.add_argument("--control-proxy", type=float, default=0.0431)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    active_path = Path(args.active_budget_json)
    profile_path = Path(args.profile_json)
    if not active_path.is_absolute():
        active_path = root / active_path
    if not profile_path.is_absolute():
        profile_path = root / profile_path

    result = analyze(active_path, profile_path, args.target_speedup, args.control_proxy)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_markdown(result)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(md)


if __name__ == "__main__":
    main()
