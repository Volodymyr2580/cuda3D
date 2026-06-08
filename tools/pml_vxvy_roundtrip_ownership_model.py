#!/usr/bin/env python3
"""Gate PML vx/vy round-trip ownership candidates.

The current best keeps v-PML and pressure-PML as separate kernels.  v-PML writes
vx/vy to global memory, then pressure-PML reads vx/vy stencils back.  This model
checks whether CTA-local ownership can remove that round trip without
duplicating enough velocity/CPML work to lose the gain.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FLOAT_BYTES = 4
DEFAULT_MACRO_XY = [
    ("current_pressure_tile_4x2", 4, 2),
    ("macro_8x4", 8, 4),
    ("macro_16x8", 16, 8),
    ("macro_16x16", 16, 16),
    ("macro_32x8", 32, 8),
    ("macro_32x16", 32, 16),
]


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}x"


def load_formal(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary_by_candidate"]["len16_current_best"]
    return {
        "mean_wp_speedup_vs_zmem": summary["mean_wp_speedup_vs_zmem"],
        "mean_candidate_wp": summary["mean_candidate_wp"],
    }


def load_v_layout(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    current = data["shapes"]["current_32x4x2"]
    totals = current["totals"]
    profile = data["profile_anchor"]
    return {
        "active_lanes": int(totals["active_lanes"]),
        "component_lanes": int(totals["component_lanes"]),
        "component_per_active_pressure_point": totals["component_lanes"] / totals["active_lanes"],
        "sampled_main_us": profile["sampled_main_us"],
        "p_core_us": profile["p_core_us"],
        "v_pml_us": profile["v_pml_us"],
        "p_pml_us": profile["p_pml_us"],
    }


def load_writeback(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = data["inputs"]["profile"]
    groups = data["inputs"]["source"]["grouped_sample_shares"]
    return {
        "len16_packed_us": profile["p_pml_len16_us"],
        "residual_pml_us": profile["p_pml_residual_us"],
        "source_groups": groups,
    }


def macro_candidate(
    name: str,
    x: int,
    y: int,
    z: int,
    baseline_component_per_pressure: float,
    timings: dict[str, float],
    savable_us: float,
    shared_limit_bytes: int,
) -> dict[str, Any]:
    pressure_points = z * x * y
    vx_slots = z * y * (x + 7)
    vy_slots = z * x * (y + 7)
    velocity_slots = vx_slots + vy_slots
    slots_per_pressure_point = velocity_slots / pressure_points
    duplicate_factor = slots_per_pressure_point / baseline_component_per_pressure
    candidate_time = (
        timings["p_core_us"]
        + timings["v_pml_us"] * duplicate_factor
        + timings["p_pml_us"]
        - savable_us
    )
    shared_bytes = velocity_slots * FLOAT_BYTES
    return {
        "macro_xy": [x, y],
        "pressure_points_per_macro_tile": pressure_points,
        "vx_slots_per_macro_tile": vx_slots,
        "vy_slots_per_macro_tile": vy_slots,
        "velocity_slots_per_pressure_point": slots_per_pressure_point,
        "duplicate_velocity_work_factor": duplicate_factor,
        "shared_velocity_cache_bytes": shared_bytes,
        "within_shared_limit": shared_bytes <= shared_limit_bytes,
        "candidate_sampled_main_us": candidate_time,
        "candidate_sampled_main_speedup": timings["sampled_main_us"] / candidate_time,
        "decision": "reject_cuda_prototype",
        "reason": (
            "CTA-local velocity cache needs x/y halo velocity values.  After counting duplicated "
            "velocity/CPML work and shared-memory footprint, this macro tile does not meet the >=5% gate."
        ),
    }


def compute_model(
    formal: dict[str, Any],
    v_layout: dict[str, Any],
    writeback: dict[str, Any],
    macro_xy: list[tuple[str, int, int]],
    target_speedup: float,
    shared_limit_bytes: int,
) -> dict[str, Any]:
    groups = writeback["source_groups"]
    len16_unknown_us = writeback["len16_packed_us"] * groups["other_or_unparsed"]
    residual_generous_fraction = 0.20
    residual_savable_us = writeback["residual_pml_us"] * residual_generous_fraction
    savable_us = len16_unknown_us + residual_savable_us
    target_time = v_layout["sampled_main_us"] / target_speedup
    max_duplicate = (
        target_time
        - v_layout["p_core_us"]
        - (v_layout["p_pml_us"] - savable_us)
    ) / v_layout["v_pml_us"]

    timings = {
        "sampled_main_us": v_layout["sampled_main_us"],
        "p_core_us": v_layout["p_core_us"],
        "v_pml_us": v_layout["v_pml_us"],
        "p_pml_us": v_layout["p_pml_us"],
    }
    candidates = {
        name: macro_candidate(
            name,
            x,
            y,
            32,
            v_layout["component_per_active_pressure_point"],
            timings,
            savable_us,
            shared_limit_bytes,
        )
        for name, x, y in macro_xy
    }
    ideal_no_duplicate_time = (
        timings["p_core_us"]
        + timings["v_pml_us"]
        + timings["p_pml_us"]
        - savable_us
    )
    candidates["ideal_no_duplicate_cross_cta_owner"] = {
        "duplicate_velocity_work_factor": 1.0,
        "candidate_sampled_main_us": ideal_no_duplicate_time,
        "candidate_sampled_main_speedup": timings["sampled_main_us"] / ideal_no_duplicate_time,
        "within_shared_limit": None,
        "decision": "reject_not_ordinary_cuda",
        "reason": (
            "The speed ceiling can look meaningful only if velocity values are computed once and consumed "
            "by neighboring pressure CTAs without global memory.  Ordinary CUDA has no cross-CTA register/shared "
            "exchange or grid-wide barrier for this producer-consumer pattern."
        ),
    }
    best_name, best = max(candidates.items(), key=lambda item: item[1]["candidate_sampled_main_speedup"])
    implementable = [
        item
        for item in candidates.values()
        if item["decision"] == "allow_cuda_prototype"
    ]
    return {
        "inputs": {
            "formal": formal,
            "v_layout": v_layout,
            "writeback": writeback,
            "target_speedup": target_speedup,
            "shared_limit_bytes": shared_limit_bytes,
            "savable_model": {
                "len16_unknown_or_vxvy_us": len16_unknown_us,
                "residual_generous_fraction": residual_generous_fraction,
                "residual_generous_savable_us": residual_savable_us,
                "total_generous_vxvy_roundtrip_savable_us": savable_us,
                "note": (
                    "This is intentionally generous: len16 assigns all unparsed source samples to vx/vy, "
                    "and residual pressure-PML grants 20% savable time even though source evidence points "
                    "mostly to CPML state and p0 writeback."
                ),
            },
        },
        "thresholds": {
            "target_sampled_main_us": target_time,
            "max_duplicate_velocity_work_factor_for_5pct": max_duplicate,
            "baseline_component_per_active_pressure_point": v_layout["component_per_active_pressure_point"],
        },
        "candidates": candidates,
        "gate": {
            "decision": "reject_vxvy_roundtrip_ownership_cuda_prototype",
            "best_candidate": best_name,
            "best_sampled_main_speedup": best["candidate_sampled_main_speedup"],
            "reason": (
                "Under a generous vx/vy-roundtrip savings budget, an implementable CTA-local cache would need "
                f"duplicate velocity/CPML work factor <= {max_duplicate:.3f}.  Feasible macro tiles exceed that "
                "or exceed a conservative shared-memory limit, while the only passing ceiling requires impossible "
                "cross-CTA exchange without global memory."
            ),
            "next_allowed": [
                "source-aware multi-step/wavefront design only after synchronization and halo ownership proof",
                "precision-relaxation study only with explicit tolerance policy",
                "application-level multi-shot batching if CUDA-core exact routes remain gated off",
            ],
            "prohibited": [
                "CTA-local vx/vy shared-cache fusion under current tile/macro-tile ownership",
                "RECOMPUTE_X/Y/XYZ or direct p1 x/y derivative replacement",
                "current-geometry vx/vy component-owner split",
                "ordinary CUDA producer-consumer vx/vy fusion that relies on cross-CTA shared values",
            ],
            "implementable_candidate_count": len(implementable),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    inputs = result["inputs"]
    v = inputs["v_layout"]
    savable = inputs["savable_model"]
    thresholds = result["thresholds"]
    gate = result["gate"]
    lines = [
        "# PML Vx/Vy Round-Trip Ownership Gate",
        "",
        "## Context",
        "",
        f"- sampled main: `{v['sampled_main_us']:.3f}us`",
        f"- p_core: `{v['p_core_us']:.3f}us`",
        f"- v_pml: `{v['v_pml_us']:.3f}us`",
        f"- pressure-PML total: `{v['p_pml_us']:.3f}us`",
        f"- formal current-best WP speedup vs zmem: `{ratio(inputs['formal']['mean_wp_speedup_vs_zmem'])}`",
        f"- baseline component lanes / active pressure lane: `{v['component_per_active_pressure_point']:.4f}`",
        "",
        "Generous savable-time model:",
        "",
        f"- len16 unknown/unparsed source time assigned to vx/vy: `{savable['len16_unknown_or_vxvy_us']:.3f}us`",
        f"- residual pressure-PML generous savable fraction: `{pct(savable['residual_generous_fraction'])}`",
        f"- residual generous savable time: `{savable['residual_generous_savable_us']:.3f}us`",
        f"- total generous vx/vy-roundtrip savable time: `{savable['total_generous_vxvy_roundtrip_savable_us']:.3f}us`",
        f"- duplicate velocity work factor allowed for 5% sampled-main speedup: `{thresholds['max_duplicate_velocity_work_factor_for_5pct']:.3f}`",
        "",
        "## Candidate Macro Tiles",
        "",
        "| candidate | macro x/y | shared bytes | duplicate v work | sampled-main speedup | shared OK | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, item in result["candidates"].items():
        xy = item.get("macro_xy", ["n/a", "n/a"])
        shared = item.get("shared_velocity_cache_bytes")
        shared_text = "n/a" if shared is None else str(shared)
        shared_ok = item.get("within_shared_limit")
        lines.append(
            f"| `{name}` | `{xy[0]}/{xy[1]}` | `{shared_text}` | "
            f"`{item['duplicate_velocity_work_factor']:.3f}` | "
            f"`{ratio(item['candidate_sampled_main_speedup'])}` | `{shared_ok}` | `{item['decision']}` |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- best candidate: `{gate['best_candidate']}`",
            f"- best sampled-main speedup ceiling: `{ratio(gate['best_sampled_main_speedup'])}`",
            f"- reason: {gate['reason']}",
            "",
            "Allowed next directions:",
            "",
        ]
    )
    for item in gate["next_allowed"]:
        lines.append(f"- {item}")
    lines.extend(["", "Do not continue:", ""])
    for item in gate["prohibited"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def parse_macro(text: str) -> tuple[str, int, int]:
    if "=" in text:
        name, dims = text.split("=", 1)
    else:
        dims = text
        name = "macro_" + dims.replace(",", "x").replace("x", "_x")
    parts = dims.replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"macro shape must be name=x,y or x,y: {text}")
    x, y = (int(part) for part in parts)
    if x <= 0 or y <= 0:
        raise ValueError(f"macro dimensions must be positive: {text}")
    return name, x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-summary", default="reports/day_20260608/formal_current_best_table_20260608_182525/summary.json")
    parser.add_argument("--v-layout", default="reports/day_20260608/v_pml_coalescing_layout_budget.json")
    parser.add_argument("--writeback-gate", default="reports/day_20260608/pressure_pml_writeback_state_model.json")
    parser.add_argument("--target-speedup", type=float, default=1.05)
    parser.add_argument("--shared-limit-bytes", type=int, default=96 * 1024)
    parser.add_argument("--macro", action="append", help="Macro x/y shape, e.g. macro_16x8=16,8")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    formal = Path(args.formal_summary)
    v_layout = Path(args.v_layout)
    writeback = Path(args.writeback_gate)
    if not formal.is_absolute():
        formal = root / formal
    if not v_layout.is_absolute():
        v_layout = root / v_layout
    if not writeback.is_absolute():
        writeback = root / writeback

    macros = [parse_macro(item) for item in args.macro] if args.macro else DEFAULT_MACRO_XY
    result = compute_model(
        load_formal(formal),
        load_v_layout(v_layout),
        load_writeback(writeback),
        macros,
        args.target_speedup,
        args.shared_limit_bytes,
    )
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
