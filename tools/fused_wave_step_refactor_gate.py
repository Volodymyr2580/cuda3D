#!/usr/bin/env python3
"""Gate the proposed fused wave-step / z-tiling refactor route.

The user-provided refactor proposal suggests fusing velocity, pressure, and
CPML updates, adding z-direction tiling/prefetch, unifying CPML formulas, and
revisiting data layout / precision.  This tool converts that proposal into a
model-first gate using the already-audited current-best measurements.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PML_TILE_Z = 32
PML_TILE_X = 4
PML_TILE_Y = 2
PML_DERIVATIVE_LEFT_HALO = 4
PML_DERIVATIVE_RIGHT_HALO = 3


def ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}x"


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def amdahl(local_speedup: float, share: float) -> float:
    if share <= 0.0:
        return 1.0
    if local_speedup <= 0.0:
        return 0.0
    if math.isinf(local_speedup):
        return 1.0 / (1.0 - share)
    return 1.0 / ((1.0 - share) + share / local_speedup)


def load_json(root: Path, path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    return json.loads(path.read_text(encoding="utf-8"))


def formal_current_best(data: dict[str, Any]) -> dict[str, Any]:
    summary = {item["candidate"]: item for item in data["summary"]}
    item = summary["current_best_v_pml_len16"]
    return {
        "candidate": "current_best_v_pml_len16",
        "wp_speedup_vs_zmem": item["mean_wp_speedup_vs_zmem"],
        "gradient_speedup_vs_zmem": item["mean_gradient_speedup_vs_zmem"],
        "elapsed_speedup_vs_zmem": item["mean_elapsed_speedup_vs_zmem"],
        "max_rel_l2": item["max_rel_l2"],
        "mean_wp_s": item["mean_wp"],
        "mean_gradient_s": item["mean_gradient"],
    }


def pml_halo_duplication_model() -> dict[str, Any]:
    outputs = PML_TILE_Z * PML_TILE_X * PML_TILE_Y
    x_span = PML_TILE_X + PML_DERIVATIVE_LEFT_HALO + PML_DERIVATIVE_RIGHT_HALO
    y_span = PML_TILE_Y + PML_DERIVATIVE_LEFT_HALO + PML_DERIVATIVE_RIGHT_HALO
    vx_points_for_pressure = PML_TILE_Z * x_span * PML_TILE_Y
    vy_points_for_pressure = PML_TILE_Z * PML_TILE_X * y_span
    xy_baseline_component_points = 2 * outputs
    xy_fused_component_points = vx_points_for_pressure + vy_points_for_pressure
    return {
        "tile_shape": [PML_TILE_Z, PML_TILE_X, PML_TILE_Y],
        "pressure_outputs_per_tile": outputs,
        "derivative_halo": {
            "left": PML_DERIVATIVE_LEFT_HALO,
            "right": PML_DERIVATIVE_RIGHT_HALO,
        },
        "vx_points_needed_for_pressure_tile": vx_points_for_pressure,
        "vy_points_needed_for_pressure_tile": vy_points_for_pressure,
        "xy_baseline_component_points": xy_baseline_component_points,
        "xy_fused_component_points": xy_fused_component_points,
        "vx_duplication_factor": vx_points_for_pressure / outputs,
        "vy_duplication_factor": vy_points_for_pressure / outputs,
        "xy_component_duplication_factor": xy_fused_component_points / xy_baseline_component_points,
    }


def compute(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    formal = formal_current_best(load_json(root, args.formal))
    ownership = load_json(root, args.ownership)
    nsys = load_json(root, args.nsys)
    async_summary = load_json(root, args.async_summary)
    cluster = load_json(root, args.cluster_local)
    cluster_probe = load_json(root, args.cluster_probe)

    post = ownership["inputs"]["post_vlen16"]
    sampled_main_us = post["sampled_main_us"]
    p_core_us = post["p_core_us"]
    p_pml_us = post["p_pml_total_us"]
    v_pml_us = post["v_pml_total_us"]
    p_core_share = post["p_core_share"]
    p_pml_share = post["p_pml_total_share"]
    v_pml_share = post["v_pml_total_share"]
    target_speedup = args.target_speedup
    target_saved_us = sampled_main_us - sampled_main_us / target_speedup

    halo = pml_halo_duplication_model()
    xy_dup = halo["xy_component_duplication_factor"]
    optimistic_fused_xy_velocity_equiv_us = v_pml_us * (2.0 / 3.0) * xy_dup
    conservative_fused_velocity_equiv_us = v_pml_us * ((2.0 / 3.0) * xy_dup + (1.0 / 3.0))
    no_dup_perfect_velocity_removed_speedup = amdahl(math.inf, v_pml_share)

    optimistic_new_sampled = p_core_us + p_pml_us + optimistic_fused_xy_velocity_equiv_us
    conservative_new_sampled = p_core_us + p_pml_us + conservative_fused_velocity_equiv_us

    routes = {
        "single_kernel_fuse_v_p_pml": {
            "decision": "reject_cuda_prototype",
            "modeled_speedup": sampled_main_us / conservative_new_sampled,
            "optimistic_speedup": sampled_main_us / optimistic_new_sampled,
            "utopian_no_dup_speedup": no_dup_perfect_velocity_removed_speedup,
            "reason": (
                "Exact pressure update needs updated vx/vy from neighboring x/y positions.  Without a grid-wide "
                "barrier, a single kernel cannot safely consume neighbor-block velocity values; computing halo "
                "velocities locally duplicates about "
                f"{ratio(xy_dup)} x/y component work before register/shared-memory overhead."
            ),
        },
        "reduce_kernel_launch_count": {
            "decision": "reject_as_primary_route",
            "modeled_speedup": nsys["totals"]["ideal_gap_elimination_speedup"],
            "optimistic_speedup": async_summary["mean_wp_speedup"],
            "reason": (
                "Nsight Systems shows WP time is already almost fully GPU-kernel time; the visible WP-minus-kernel "
                "gap is only "
                f"{pct(nsys['totals']['wp_minus_kernel_total_fraction'])}, and an async scheduling prototype "
                f"measured only {ratio(async_summary['mean_wp_speedup'])} WP speedup."
            ),
        },
        "z_direction_loop_tiling_prefetch": {
            "decision": "reject_current_exact_cuda_prototype",
            "modeled_speedup": ownership["routes"]["source_aware_temporal_wavefront"]["model_ceiling"],
            "optimistic_speedup": ownership["routes"]["source_aware_temporal_wavefront"]["model_ceiling"],
            "reason": (
                "Spatial z reuse is already present in p_core shared z tiles and accepted pressure/v-PML len16 "
                "z-line ownership.  True multi-step temporal z-tiling reopens the K=2 p_mid dependency problem, "
                "which ordinary CUDA and cluster DSM gates already rejected."
            ),
        },
        "unify_cpml_formula_template": {
            "decision": "documentation_or_cleanup_only",
            "modeled_speedup": 1.0,
            "optimistic_speedup": 1.0,
            "reason": (
                "A formula template can reduce source duplication, but current hot spots are final pressure "
                "writeback and recursive CPML state traffic.  A template alone does not remove bytes or "
                "synchronization."
            ),
        },
        "aos_or_soa_layout_rewrite": {
            "decision": "reject_aos_for_exact_current_best",
            "modeled_speedup": 1.0,
            "optimistic_speedup": 1.0,
            "reason": (
                "The code already uses SoA arrays, which match p_core and PML streaming access.  AoS would make "
                "single-field stencil loads stride across unrelated variables unless a new all-field fused "
                "ownership model passes first."
            ),
        },
        "mixed_precision_tensor_core": {
            "decision": "out_of_scope_for_exact_fp32_branch",
            "modeled_speedup": None,
            "optimistic_speedup": None,
            "reason": (
                "This may be a separate relaxed-precision branch, but it changes the tolerance policy and cannot "
                "be mixed into the current exact-FP32 line."
            ),
        },
        "cluster_or_persistent_fusion": {
            "decision": "design_only_rejected_by_current_cluster_model",
            "modeled_speedup": cluster["gate"]["best_sampled_main_speedup_estimate"],
            "optimistic_speedup": cluster["byte_model"]["ideal_no_dup_sampled_main_speedup"],
            "reason": (
                "RTX 5090 supports clusters, but the current best DSM tile is slower than baseline "
                f"({ratio(cluster['gate']['best_sampled_main_speedup_estimate'])}); full cooperative K=2 needs "
                f"{cluster_probe['probe']['cooperative_required_blocks_for_previous_k2']} blocks versus a "
                f"{cluster_probe['probe']['cooperative_grid_block_ceiling']} block resident ceiling."
            ),
        },
    }

    allowed = [
        name
        for name, route in routes.items()
        if route["decision"] in {"allow_cuda_prototype", "accept_cuda_prototype"}
    ]
    design_only = [
        name
        for name, route in routes.items()
        if "design_only" in route["decision"] or route["decision"].endswith("_only")
    ]

    gate = {
        "decision": "reject_immediate_fused_wave_step_cuda_prototype",
        "cuda_prototype_allowed": False,
        "allowed_cuda_prototypes": allowed,
        "design_only_routes": design_only,
        "target_speedup": target_speedup,
        "target_saved_us": target_saved_us,
        "reason": (
            "The proposal is directionally ambitious, but the exact-FP32 current-best path already removed the "
            "cheap z/global round trips.  The remaining one-kernel fusion needs cross-CTA velocity availability "
            "or heavy halo recomputation, while launch reduction, simple z-tiling, CPML templating, and AoS layout "
            "do not pass the >=5% modeled repeat-speedup gate."
        ),
        "next_allowed": [
            "If staying exact-FP32 single-GPU, only study a new persistent/cluster ownership representation that first beats the halo/DSM byte gate.",
            "If the goal is total throughput, move to multi-GPU batching using the current-best kernel stack.",
            "If the scientific tolerance can change, open a separate relaxed-precision branch with its own correctness policy.",
        ],
    }

    return {
        "inputs": {
            "formal_current_best": formal,
            "sampled_main": {
                "total_us": sampled_main_us,
                "p_core_us": p_core_us,
                "p_pml_total_us": p_pml_us,
                "v_pml_total_us": v_pml_us,
                "p_core_share": p_core_share,
                "p_pml_total_share": p_pml_share,
                "v_pml_total_share": v_pml_share,
            },
            "nsys_gap": nsys["totals"],
            "async_streams": {
                "mean_wp_speedup": async_summary["mean_wp_speedup"],
                "mean_gradient_speedup": async_summary["mean_gradient_speedup"],
                "all_compare_pass": async_summary["all_compare_pass"],
            },
            "cluster": {
                "best_sampled_main_speedup_estimate": cluster["gate"]["best_sampled_main_speedup_estimate"],
                "ideal_no_dup_sampled_main_speedup": cluster["byte_model"]["ideal_no_dup_sampled_main_speedup"],
                "cooperative_grid_block_ceiling": cluster_probe["probe"]["cooperative_grid_block_ceiling"],
                "cooperative_required_blocks_for_previous_k2": cluster_probe["probe"]["cooperative_required_blocks_for_previous_k2"],
            },
        },
        "derived": {
            "target_saved_us_for_5pct_sampled_main": target_saved_us,
            "pml_halo_duplication_model": halo,
            "optimistic_fused_xy_velocity_equiv_us": optimistic_fused_xy_velocity_equiv_us,
            "conservative_fused_velocity_equiv_us": conservative_fused_velocity_equiv_us,
            "optimistic_fused_sampled_main_us": optimistic_new_sampled,
            "conservative_fused_sampled_main_us": conservative_new_sampled,
        },
        "routes": routes,
        "gate": gate,
    }


def render_markdown(result: dict[str, Any]) -> str:
    formal = result["inputs"]["formal_current_best"]
    sampled = result["inputs"]["sampled_main"]
    halo = result["derived"]["pml_halo_duplication_model"]
    gate = result["gate"]
    lines = [
        "# Fused Wave-Step Refactor Gate",
        "",
        "## Summary",
        "",
        "This report evaluates the proposed fused-kernel / z-tiling / CPML-unification refactor against the current exact-FP32 CUDA3D frontier.",
        "",
        f"- current best: `{formal['candidate']}`",
        f"- formal WP speedup vs zmem: `{ratio(formal['wp_speedup_vs_zmem'])}`",
        f"- formal Gradient speedup vs zmem: `{ratio(formal['gradient_speedup_vs_zmem'])}`",
        f"- max rel L2: `{formal['max_rel_l2']:.6e}`",
        f"- target gate for a new CUDA prototype: `{ratio(gate['target_speedup'])}` sampled-main / repeat speedup ceiling",
        f"- decision: `{gate['decision']}`",
        "",
        "The proposal is not ignored.  It is rejected as an immediate CUDA prototype because the current data show no safe `>=5%` exact-FP32 single-GPU ceiling after synchronization and halo costs.",
        "",
        "## Current Profile Anchor",
        "",
        "| region | duration | share |",
        "| --- | ---: | ---: |",
        f"| `p_core` | `{sampled['p_core_us']:.3f}us` | `{pct(sampled['p_core_share'])}` |",
        f"| `pressure-PML total` | `{sampled['p_pml_total_us']:.3f}us` | `{pct(sampled['p_pml_total_share'])}` |",
        f"| `v-PML total` | `{sampled['v_pml_total_us']:.3f}us` | `{pct(sampled['v_pml_total_share'])}` |",
        f"| sampled main total | `{sampled['total_us']:.3f}us` | `100.00%` |",
        "",
        f"To pass a `>=5%` gate, a candidate needs to save about `{gate['target_saved_us']:.3f}us` from sampled-main work.",
        "",
        "## Why Full Fusion Is Not a Free Win",
        "",
        "The tempting goal is to compute updated velocity and immediately consume it in pressure-PML.  The catch is that pressure uses neighboring `vx/vy` values.  With the accepted `32x4x2` PML tile, an exact fused tile must either read globally written neighbor velocities or recompute halo velocities locally.",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        f"| pressure outputs per tile | `{halo['pressure_outputs_per_tile']}` |",
        f"| `vx` points needed by pressure tile | `{halo['vx_points_needed_for_pressure_tile']}` |",
        f"| `vy` points needed by pressure tile | `{halo['vy_points_needed_for_pressure_tile']}` |",
        f"| x/y component duplication factor | `{ratio(halo['xy_component_duplication_factor'])}` |",
        "",
        "That duplication is before accounting for extra registers, shared memory, branch control, and CPML state halos.",
        "",
        "## Route Matrix",
        "",
        "| route from proposal | modeled speedup | optimistic speedup | decision | reason |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for name, route in result["routes"].items():
        lines.append(
            f"| `{name}` | `{ratio(route['modeled_speedup'])}` | `{ratio(route['optimistic_speedup'])}` | "
            f"`{route['decision']}` | {route['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- CUDA prototype allowed: `{str(gate['cuda_prototype_allowed']).lower()}`",
            f"- reason: {gate['reason']}",
            "",
            "Next allowed work:",
            "",
        ]
    )
    for item in gate["next_allowed"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", default="reports/day_20260608/formal_vpmlen16_table_20260608_2359/summary.json")
    parser.add_argument("--ownership", default="reports/day_20260608/ownership_frontier_gate.json")
    parser.add_argument("--nsys", default="reports/day_20260608/directfill_scheduling_nsys_summary.json")
    parser.add_argument("--async-summary", default="reports/day_20260608/wavestep_async_perf6_repeat_20260608_175407/summary.json")
    parser.add_argument("--cluster-local", default="reports/day_20260609/cluster_local_ownership_model.json")
    parser.add_argument("--cluster-probe", default="reports/day_20260609/cluster_cooperative_frontier_gate.json")
    parser.add_argument("--target-speedup", type=float, default=1.05)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    result = compute(args, root)

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if args.md_out:
        out = Path(args.md_out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(result), encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(render_markdown(result))


if __name__ == "__main__":
    main()
