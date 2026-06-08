#!/usr/bin/env python3
"""Gate residual pressure-PML routes after the formal v-PML len16 table.

The accepted current-best still spends meaningful time in the residual
``cuda_fd3d_p_pml_tile_ns`` path.  This script anchors that residual path to the
latest NCU profile and checks whether branch/control/lane-compaction style
follow-ups are strong enough to justify another CUDA prototype.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}x"


def amdahl(local_speedup: float, share: float) -> float:
    if share <= 0.0:
        return 1.0
    if local_speedup <= 0.0:
        return 0.0
    if math.isinf(local_speedup):
        return 1.0 / (1.0 - share)
    return 1.0 / ((1.0 - share) + share / local_speedup)


def required_local_speedup(target_speedup: float, share: float) -> float | None:
    if share <= 0.0:
        return None
    denom = (1.0 / target_speedup) - (1.0 - share)
    if denom <= 0.0:
        return math.inf
    return share / denom


def load_post_gate(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = data["inputs"]["profile"]
    shares = profile["shares"]
    return {
        "sampled_main_us": profile["sampled_main_us"],
        "p_pml_residual_us": profile["p_pml_residual_us"],
        "p_pml_residual_share": shares["p_pml_residual"],
        "p_pml_total_us": profile["p_pml_total_us"],
        "p_pml_total_share": shares["p_pml_total"],
        "target_speedup": data["inputs"]["target_speedup"],
        "required_residual_local_speedup": data["derived"]["required_local_speedup_by_region"][
            "p_pml_residual"
        ],
    }


def load_residual_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = data["profiles"][0]
    metrics = profile["kernels"]["cuda_fd3d_p_pml_tile_ns"]["metrics"]
    return {
        "profile_label": profile["label"],
        "metrics": metrics,
    }


def load_compact_descriptor(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", {})
    exact23 = candidates.get("exact_length23_points_only", {})
    return {
        "decision": data.get("gate", {}).get("decision"),
        "exact_length23_calibrated_sampled_speedup": exact23.get(
            "calibrated_sampled_main_speedup_vs_len16"
        ),
        "exact_length23_ceiling": exact23.get("sampled_main_speedup_ceiling_vs_len16"),
        "exact_length23_lane_reduction": exact23.get("lane_reduction_vs_len16"),
    }


def compute_model(post: dict[str, Any], profile: dict[str, Any], compact: dict[str, Any]) -> dict[str, Any]:
    share = post["p_pml_residual_share"]
    required_local = post["required_residual_local_speedup"]
    required_reduction = 1.0 - 1.0 / required_local
    required_saved_us = post["p_pml_residual_us"] * required_reduction
    metrics = profile["metrics"]
    branch_eff = float(metrics["branch_efficiency_pct"]) / 100.0
    active_threads = float(metrics["avg_active_threads_per_warp"])
    not_pred_threads = float(metrics["avg_not_predicated_off_threads_per_warp"])

    # These are deliberately upper bounds. They ignore extra tile-list, launch,
    # descriptor, and control overhead.
    perfect_branch_local = 1.0 / branch_eff if branch_eff > 0.0 else math.inf
    recover_predication_local = active_threads / not_pred_threads if not_pred_threads > 0.0 else math.inf
    recover_to_full_warp_local = 32.0 / active_threads if active_threads > 0.0 else math.inf
    recover_to_full_not_pred_local = 32.0 / not_pred_threads if not_pred_threads > 0.0 else math.inf

    scenarios = {
        "perfect_branch_efficiency": {
            "local_speedup": perfect_branch_local,
            "sampled_main_speedup": amdahl(perfect_branch_local, share),
            "interpretation": "upper bound for branch/control specialization without removing memory traffic",
        },
        "recover_predicated_off_threads_only": {
            "local_speedup": recover_predication_local,
            "sampled_main_speedup": amdahl(recover_predication_local, share),
            "interpretation": "upper bound for predicate cleanup while keeping same active-lane shape",
        },
        "recover_all_active_threads_to_full_warp": {
            "local_speedup": recover_to_full_warp_local,
            "sampled_main_speedup": amdahl(recover_to_full_warp_local, share),
            "interpretation": "utopian lane-utilization bound; descriptor/control overhead not counted",
        },
        "recover_all_not_predicated_threads_to_full_warp": {
            "local_speedup": recover_to_full_not_pred_local,
            "sampled_main_speedup": amdahl(recover_to_full_not_pred_local, share),
            "interpretation": "more utopian bound combining active-lane and predication cleanup",
        },
        "twenty_percent_residual_reduction": {
            "local_speedup": 1.0 / 0.8,
            "sampled_main_speedup": amdahl(1.0 / 0.8, share),
            "interpretation": "minimum ballpark residual reduction needed to cross the gate",
        },
        "exact_length23_descriptor_calibrated": {
            "local_speedup": None,
            "sampled_main_speedup": compact["exact_length23_calibrated_sampled_speedup"],
            "interpretation": "existing calibrated descriptor budget after accepted pressure len16",
        },
    }

    gate = {
        "decision": "reject_residual_pressure_micro_cuda_prototype",
        "reason": (
            "Residual pressure-PML would need about "
            f"{pct(required_reduction)} local time reduction ({required_saved_us:.3f}us) "
            "to move sampled-main by >=5%.  The current profile already has branch efficiency "
            f"{metrics['branch_efficiency_pct']:.2f}% and avg active threads/warp "
            f"{metrics['avg_active_threads_per_warp']:.2f}; branch/predicate cleanup stays below the gate, "
            "while lane/descriptor compaction was already calibrated at only about "
            f"{ratio(compact['exact_length23_calibrated_sampled_speedup'])} sampled-main speedup."
        ),
        "next_allowed": [
            "pressure/wave-step ownership model that removes real pressure writeback or CPML state traffic",
            "cross-CTA or cluster-level ownership study only if a concrete synchronization primitive is identified",
            "precision-relaxation study only after an explicit tolerance-policy change",
            "external handoff/report summarizing current hard gates",
        ],
        "prohibited": [
            "residual pressure branch-only split",
            "length-32 branch/control specialization retry",
            "length-23 or exact active-point descriptor retry",
            "residual p0 __ldg/local-new_mem/cache-policy/z-cache micro-tuning",
        ],
    }

    return {
        "inputs": {
            "post_vlen16_gate": post,
            "residual_profile": profile,
            "compact_descriptor": compact,
        },
        "derived": {
            "required_residual_local_speedup": required_local,
            "required_residual_local_reduction": required_reduction,
            "required_saved_us": required_saved_us,
            "scenarios": scenarios,
        },
        "gate": gate,
    }


def render_markdown(result: dict[str, Any]) -> str:
    post = result["inputs"]["post_vlen16_gate"]
    metrics = result["inputs"]["residual_profile"]["metrics"]
    compact = result["inputs"]["compact_descriptor"]
    derived = result["derived"]
    gate = result["gate"]
    lines = [
        "# Residual Pressure-PML Route Gate",
        "",
        "## Context",
        "",
        f"- sampled main: `{post['sampled_main_us']:.3f}us`",
        f"- residual pressure-PML: `{post['p_pml_residual_us']:.3f}us` / `{pct(post['p_pml_residual_share'])}`",
        f"- target sampled-main gate: `{ratio(post['target_speedup'])}`",
        f"- required residual local speedup: `{ratio(derived['required_residual_local_speedup'])}`",
        f"- required residual local reduction: `{pct(derived['required_residual_local_reduction'])}`",
        f"- required saved time: `{derived['required_saved_us']:.3f}us`",
        "",
        "## Residual NCU Anchor",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| No Eligible | `{metrics['no_eligible_pct']:.3f}%` |",
        f"| eligible warps/scheduler | `{metrics['eligible_warps_per_scheduler']:.3f}` |",
        f"| warp cycles/issued inst | `{metrics['warp_cycles_per_issued_instruction']:.3f}` |",
        f"| avg active threads/warp | `{metrics['avg_active_threads_per_warp']:.3f}` |",
        f"| avg not-predicated threads/warp | `{metrics['avg_not_predicated_off_threads_per_warp']:.3f}` |",
        f"| branch efficiency | `{metrics['branch_efficiency_pct']:.3f}%` |",
        f"| L1/TEX hit | `{metrics['l1tex_hit_rate_pct']:.3f}%` |",
        f"| L2 hit | `{metrics['l2_hit_rate_pct']:.3f}%` |",
        f"| achieved occupancy | `{metrics['achieved_occupancy_pct']:.3f}%` |",
        "",
        "## Scenario Ceilings",
        "",
        "| scenario | local speedup | sampled-main speedup | interpretation |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, item in derived["scenarios"].items():
        lines.append(
            f"| `{name}` | `{ratio(item['local_speedup'])}` | "
            f"`{ratio(item['sampled_main_speedup'])}` | {item['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Descriptor Prior",
            "",
            f"- existing compact descriptor decision: `{compact['decision']}`",
            f"- length-23 calibrated sampled-main speedup: `{ratio(compact['exact_length23_calibrated_sampled_speedup'])}`",
            f"- length-23 lane reduction after accepted len16: `{pct(compact['exact_length23_lane_reduction'])}`",
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- reason: {gate['reason']}",
            "",
            "Allowed next:",
            "",
        ]
    )
    for item in gate["next_allowed"]:
        lines.append(f"- {item}")
    lines.extend(["", "Prohibited:", ""])
    for item in gate["prohibited"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--post-gate",
        default="reports/day_20260608/post_vlen16_pressure_next_gate.json",
    )
    parser.add_argument(
        "--residual-profile",
        default="reports/day_20260608/residual_pressure_source_profile_20260609_0012/details_summary.json",
    )
    parser.add_argument(
        "--compact-budget",
        default="reports/day_20260608/pml_compact_descriptor_budget.json",
    )
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    def rooted(path_text: str) -> Path:
        path = Path(path_text)
        return path if path.is_absolute() else root / path

    result = compute_model(
        load_post_gate(rooted(args.post_gate)),
        load_residual_profile(rooted(args.residual_profile)),
        load_compact_descriptor(rooted(args.compact_budget)),
    )

    if args.json_out:
        out = rooted(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if args.md_out:
        out = rooted(args.md_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(result), encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(render_markdown(result))


if __name__ == "__main__":
    main()
