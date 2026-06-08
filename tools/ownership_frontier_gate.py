#!/usr/bin/env python3
"""Consolidate the exact-CUDA ownership frontier after current-best.

This gate is intentionally broader than the previous single-route gates.  It
answers a planning question: after pressure len16 and v-PML len16, is there any
ordinary CUDA, exact-arithmetic structural route left that justifies writing a
new prototype immediately?
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


def load_formal(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = {item["candidate"]: item for item in data["summary"]}
    current = summary["current_best_v_pml_len16"]
    return {
        "current_best_alias": data["current_best_alias"],
        "mean_wp_speedup_vs_zmem": current["mean_wp_speedup_vs_zmem"],
        "mean_gradient_speedup_vs_zmem": current["mean_gradient_speedup_vs_zmem"],
        "mean_elapsed_speedup_vs_zmem": current["mean_elapsed_speedup_vs_zmem"],
        "max_rel_l2": current["max_rel_l2"],
        "directfill_wp": summary["directfill"]["mean_wp_speedup_vs_zmem"],
        "pressure_len16_wp": summary["pressure_len16"]["mean_wp_speedup_vs_zmem"],
    }


def load_post(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = data["inputs"]["profile"]
    derived = data["derived"]
    source_req = derived["len16_source_subgroup_required_speedup"]
    return {
        "sampled_main_us": profile["sampled_main_us"],
        "p_core_us": profile["p_core_us"],
        "p_core_share": profile["shares"]["p_core"],
        "p_pml_total_us": profile["p_pml_total_us"],
        "p_pml_total_share": profile["shares"]["p_pml_total"],
        "p_pml_len16_us": profile["p_pml_len16_us"],
        "p_pml_len16_share": profile["shares"]["p_pml_len16"],
        "p_pml_residual_us": profile["p_pml_residual_us"],
        "p_pml_residual_share": profile["shares"]["p_pml_residual"],
        "v_pml_total_us": profile["v_pml_total_us"],
        "v_pml_total_share": profile["shares"]["v_pml_total"],
        "target_speedup": data["inputs"]["target_speedup"],
        "required_local_by_region": derived["required_local_speedup_by_region"],
        "source_group_required": source_req,
        "pressure_scenarios": derived["pressure_scenario_sampled_speedups"],
        "fusion_scenarios": derived["fusion_scenario_sampled_speedups"],
    }


def load_residual(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "required_residual_local_speedup": data["derived"]["required_residual_local_speedup"],
        "perfect_branch_sampled": data["derived"]["scenarios"]["perfect_branch_efficiency"][
            "sampled_main_speedup"
        ],
        "descriptor_sampled": data["derived"]["scenarios"]["exact_length23_descriptor_calibrated"][
            "sampled_main_speedup"
        ],
        "gate_decision": data["gate"]["decision"],
    }


def compute_frontier(formal: dict[str, Any], post: dict[str, Any], residual: dict[str, Any]) -> dict[str, Any]:
    target_next_milestone = 1.5
    additional_to_15 = target_next_milestone / formal["mean_wp_speedup_vs_zmem"]
    target_sampled = post["target_speedup"]
    sampled = post["sampled_main_us"]
    target_saved_us = sampled - sampled / target_sampled

    routes = {
        "pressure_final_writeback_state_representation": {
            "region_share": post["p_pml_len16_share"],
            "model_ceiling": post["pressure_scenarios"]["len16_final_update_2x"],
            "required_local": post["source_group_required"]["final_p0_p1_cw2_if_alone"],
            "decision": "reject_cuda_prototype",
            "reason": (
                "A 2x final-update subgroup win is mathematically enough in isolation, but exact state "
                "representation variants have already moved or increased traffic into p_core/v_pml or added state writes."
            ),
        },
        "cpml_recursive_state_traffic": {
            "region_share": post["p_pml_len16_share"],
            "model_ceiling": post["pressure_scenarios"]["len16_mem_dzz_2x"],
            "required_local": post["source_group_required"]["cpml_mem_dzz_if_alone"],
            "decision": "reject_cuda_prototype",
            "reason": (
                "mem_dzz is a recursive per-step state with no intra-step reuse target; even a 2x local win is below gate."
            ),
        },
        "combined_final_plus_mem_dzz_state_redesign": {
            "region_share": post["p_pml_len16_share"],
            "model_ceiling": post["pressure_scenarios"]["len16_final_plus_mem_dzz_1_5x"],
            "required_local": post["source_group_required"]["final_plus_mem_dzz"],
            "decision": "design_only",
            "reason": (
                "Only a broader state/ownership redesign can plausibly reach the gate, but no exact ordinary-CUDA design has "
                "yet shown how to remove both traffic groups without larger side effects."
            ),
        },
        "residual_pressure_branch_or_descriptor": {
            "region_share": post["p_pml_residual_share"],
            "model_ceiling": max(residual["perfect_branch_sampled"], residual["descriptor_sampled"]),
            "required_local": residual["required_residual_local_speedup"],
            "decision": "reject_cuda_prototype",
            "reason": (
                "Residual pressure branch/predicate/descriptor cleanups are below the 5% sampled-main gate."
            ),
        },
        "v_pml_descriptor_or_micro_packing": {
            "region_share": post["v_pml_total_share"],
            "model_ceiling": None,
            "required_local": post["required_local_by_region"]["v_pml_total"],
            "decision": "reject_cuda_prototype",
            "reason": (
                "After accepted v-len16, v-PML share is only 18.37%; descriptor expansion has no >=5% overhead model."
            ),
        },
        "ordinary_v_pressure_zface_fusion": {
            "region_share": post["v_pml_total_share"] + post["p_pml_total_share"],
            "model_ceiling": post["fusion_scenarios"]["perfect_remove_v_len16_time"],
            "required_local": None,
            "decision": "reject_previous_failed_family",
            "reason": (
                "Direct z-face VP fusion/shared-VP has already failed; ordinary CUDA variants duplicate halo/control work or "
                "need cross-CTA values not available in shared/register state."
            ),
        },
        "source_aware_temporal_wavefront": {
            "region_share": post["p_core_share"],
            "model_ceiling": 1.1248,
            "required_local": post["required_local_by_region"]["p_core"],
            "decision": "reject_ordinary_cuda",
            "reason": (
                "The ideal K=2 source-aware ceiling is meaningful, but ordinary CUDA schedules either materialize p_mid globally, "
                "duplicate halos heavily, or need a grid-wide/cross-CTA ownership primitive."
            ),
        },
        "host_launch_or_stream_scheduling": {
            "region_share": None,
            "model_ceiling": 1.005183,
            "required_local": None,
            "decision": "reject_cuda_prototype",
            "reason": (
                "The tested async stream prototype was correct but reached only about 0.5% WP speedup."
            ),
        },
    }

    ordinary_cuda_allowed = [
        name
        for name, route in routes.items()
        if route["decision"] in {"allow_cuda_prototype", "accept_cuda_prototype"}
    ]
    design_only = [name for name, route in routes.items() if route["decision"] == "design_only"]
    gate = {
        "decision": "ordinary_exact_cuda_frontier_exhausted_for_micro_routes",
        "ordinary_cuda_allowed_count": len(ordinary_cuda_allowed),
        "design_only_routes": design_only,
        "reason": (
            "All remaining ordinary CUDA exact routes either fail the >=5% modeled repeat-speedup gate, repeat a measured "
            "failed family, or require cross-CTA/global synchronization semantics not present in the current implementation model."
        ),
        "next_allowed": [
            "write a handoff report to Pro/next agent with current-best and prohibited routes",
            "investigate concrete cluster/cooperative persistent-kernel primitives before any cross-CTA ownership prototype",
            "precision-relaxation plan only if the user explicitly changes the tolerance policy",
            "application-level batching/multi-shot scheduling outside the CUDA-core exactness track",
        ],
        "stop_for_now": [
            "do not start another exact ordinary-CUDA micro prototype",
            "do not reopen residual pressure, v-PML descriptor, z-face fusion, current p-core shared-plane, or K=2 temporal routes",
            "do not claim 1.5x archive; current formal WP speedup is 1.222023x",
        ],
    }
    return {
        "inputs": {
            "formal": formal,
            "post_vlen16": post,
            "residual": residual,
        },
        "derived": {
            "target_next_milestone": target_next_milestone,
            "additional_wp_speedup_needed_to_1_5x": additional_to_15,
            "sampled_main_target_speedup": target_sampled,
            "target_saved_us_for_5pct_sampled_main": target_saved_us,
        },
        "routes": routes,
        "gate": gate,
    }


def render_markdown(result: dict[str, Any]) -> str:
    formal = result["inputs"]["formal"]
    post = result["inputs"]["post_vlen16"]
    derived = result["derived"]
    gate = result["gate"]
    lines = [
        "# Ownership Frontier Gate",
        "",
        "## Context",
        "",
        f"- current best: `{formal['current_best_alias']}`",
        f"- formal WP speedup vs zmem: `{ratio(formal['mean_wp_speedup_vs_zmem'])}`",
        f"- formal Gradient speedup vs zmem: `{ratio(formal['mean_gradient_speedup_vs_zmem'])}`",
        f"- formal elapsed speedup vs zmem: `{ratio(formal['mean_elapsed_speedup_vs_zmem'])}`",
        f"- max rel L2: `{formal['max_rel_l2']:.6e}`",
        f"- additional WP speedup needed to reach `1.5x`: `{ratio(derived['additional_wp_speedup_needed_to_1_5x'])}`",
        f"- sampled main profile anchor: `{post['sampled_main_us']:.3f}us`",
        f"- saved time required for `>=5%` sampled-main: `{derived['target_saved_us_for_5pct_sampled_main']:.3f}us`",
        "",
        "## Region Shares",
        "",
        "| region | duration | share | local speedup required for 5% sampled-main |",
        "| --- | ---: | ---: | ---: |",
        f"| `p_core` | `{post['p_core_us']:.3f}us` | `{pct(post['p_core_share'])}` | `{ratio(post['required_local_by_region']['p_core'])}` |",
        f"| `pressure-PML total` | `{post['p_pml_total_us']:.3f}us` | `{pct(post['p_pml_total_share'])}` | `{ratio(post['required_local_by_region']['p_pml_total'])}` |",
        f"| `pressure len16` | `{post['p_pml_len16_us']:.3f}us` | `{pct(post['p_pml_len16_share'])}` | `{ratio(post['required_local_by_region']['p_pml_len16'])}` |",
        f"| `pressure residual` | `{post['p_pml_residual_us']:.3f}us` | `{pct(post['p_pml_residual_share'])}` | `{ratio(post['required_local_by_region']['p_pml_residual'])}` |",
        f"| `v-PML total` | `{post['v_pml_total_us']:.3f}us` | `{pct(post['v_pml_total_share'])}` | `{ratio(post['required_local_by_region']['v_pml_total'])}` |",
        "",
        "## Route Matrix",
        "",
        "| route | model ceiling | required local | decision | reason |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for name, route in result["routes"].items():
        lines.append(
            f"| `{name}` | `{ratio(route['model_ceiling'])}` | `{ratio(route['required_local'])}` | "
            f"`{route['decision']}` | {route['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- ordinary CUDA allowed count: `{gate['ordinary_cuda_allowed_count']}`",
            f"- reason: {gate['reason']}",
            "",
            "Next allowed:",
            "",
        ]
    )
    for item in gate["next_allowed"]:
        lines.append(f"- {item}")
    lines.extend(["", "Stop for now:", ""])
    for item in gate["stop_for_now"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal",
        default="reports/day_20260608/formal_vpmlen16_table_20260608_2359/summary.json",
    )
    parser.add_argument(
        "--post-gate",
        default="reports/day_20260608/post_vlen16_pressure_next_gate.json",
    )
    parser.add_argument(
        "--residual-gate",
        default="reports/day_20260608/residual_pressure_route_gate.json",
    )
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    def rooted(path_text: str) -> Path:
        path = Path(path_text)
        return path if path.is_absolute() else root / path

    result = compute_frontier(
        load_formal(rooted(args.formal)),
        load_post(rooted(args.post_gate)),
        load_residual(rooted(args.residual_gate)),
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
