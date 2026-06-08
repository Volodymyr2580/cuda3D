#!/usr/bin/env python3
"""Budget v-PML tile-layout and coalescing candidates.

The accepted pressure-PML len16 path keeps the global PML tile shape at
32x4x2.  A v-PML-only layout would require separate velocity tile-list
plumbing, so this script is a gate before writing CUDA code.  It mirrors the
current velocity tile-list skip rule and estimates whether a small set of
reasoned v-only tile shapes can meet the >=5% prototype gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pml_pressure_dataflow_audit as audit  # noqa: E402


CORE_PML_MARGIN = 4
DEFAULT_SHAPES = [
    ("current_32x4x2", 32, 4, 2),
    ("z64_x2_y2", 64, 2, 2),
    ("z32_x8_y1", 32, 8, 1),
    ("z32_x2_y4", 32, 2, 4),
    ("z16_x8_y2", 16, 8, 2),
    ("z16_x4_y4", 16, 4, 4),
    ("z16_x16_y1", 16, 16, 1),
    ("z8_x8_y4", 8, 8, 4),
    ("z8_x16_y2", 8, 16, 2),
]


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.2%}"


def ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}x"


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def sampled_speedup(kernel_speedup: float, kernel_share: float | None) -> float | None:
    if kernel_share is None or kernel_speedup <= 0:
        return None
    return 1.0 / ((1.0 - kernel_share) + kernel_share / kernel_speedup)


def parse_shape(text: str) -> tuple[str, int, int, int]:
    if "=" in text:
        name, dims = text.split("=", 1)
    else:
        dims = text
        name = "z" + dims.replace(",", "x").replace("x", "_x")
    parts = dims.replace("x", ",").split(",")
    if len(parts) != 3:
        raise ValueError(f"shape must be name=z,x,y or z,x,y: {text}")
    z, x, y = (int(item) for item in parts)
    if min(z, x, y) <= 0:
        raise ValueError(f"shape dimensions must be positive: {text}")
    return name, z, x, y


def load_profile_anchor(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = {item["label"]: item for item in data["profiles"]}
    profile = profiles.get("len16") or profiles.get("directfill") or next(iter(profiles.values()))
    kernels = profile["kernels"]

    def duration_us(name: str) -> float:
        item = kernels.get(name)
        if item is None:
            return 0.0
        return float(item["metrics"]["duration_ns"]) / 1000.0

    p_core = duration_us("cuda_fd3d_p_core_ns")
    p_pml = duration_us("cuda_fd3d_p_pml_tile_ns") + duration_us("cuda_fd3d_p_pml_len16_halfwarp_ns")
    v_pml = duration_us("cuda_fd3d_v_pml_tile_ns")
    total = p_core + p_pml + v_pml
    return {
        "profile_label": profile["label"],
        "p_core_us": p_core,
        "p_pml_us": p_pml,
        "v_pml_us": v_pml,
        "sampled_main_us": total,
        "v_pml_sampled_main_share": v_pml / total if total > 0 else None,
    }


def load_v_source_metrics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = data["profiles"][0]
    kernel = profile["kernels"].get("cuda_fd3d_v_pml_tile_ns", {})
    return {
        "profile_label": profile["label"],
        "metrics": kernel.get("metrics", {}),
        "rules": kernel.get("rules", []),
    }


def shape_static_warp_segments(b1: int, b2: int, b3: int) -> dict[str, Any]:
    volume = b1 * b2 * b3
    warps = ceil_div(volume, 32)
    segments = []
    for warp in range(warps):
        keys = set()
        lanes = 0
        for lane in range(32):
            linear = warp * 32 + lane
            if linear >= volume:
                continue
            local_z = linear % b1
            local_x = (linear // b1) % b2
            local_y = linear // (b1 * b2)
            keys.add((local_x, local_y))
            lanes += 1
            _ = local_z
        segments.append({"warp": warp, "lanes": lanes, "z_segments": len(keys)})
    return {
        "warps_per_block": warps,
        "avg_z_segments_per_warp": sum(item["z_segments"] for item in segments) / len(segments),
        "max_z_segments_per_warp": max(item["z_segments"] for item in segments),
    }


def analyze_shape(domains: list[audit.ShotDomain], npml: int, b1: int, b2: int, b3: int) -> dict[str, Any]:
    totals = {
        "tiles": 0,
        "launched_lanes": 0,
        "in_domain_lanes": 0,
        "active_lanes": 0,
        "vx_active_lanes": 0,
        "vy_active_lanes": 0,
        "component_lanes": 0,
        "warps": 0,
        "component_warps": 0,
        "component_warp_active_lanes": 0,
        "p1_component_sector_units": 0.0,
        "p1_component_ideal_sector_units": 0.0,
    }
    shot_rows = []
    static_warp = shape_static_warp_segments(b1, b2, b3)
    volume = b1 * b2 * b3
    warps_per_block = ceil_div(volume, 32)

    for domain in domains:
        n3, n2, n1 = domain.n3n2n1
        grid1 = ceil_div(n1, b1)
        grid2 = ceil_div(n2, b2)
        grid3 = ceil_div(n3, b3)
        core1_lo = npml + CORE_PML_MARGIN
        core2_lo = npml + CORE_PML_MARGIN
        core3_lo = npml + CORE_PML_MARGIN
        core1_hi = n1 - npml - CORE_PML_MARGIN
        core2_hi = n2 - npml - CORE_PML_MARGIN
        core3_hi = n3 - npml - CORE_PML_MARGIN
        skip1_lo = core1_lo + 3
        skip2_lo = core2_lo + 3
        skip3_lo = core3_lo + 3
        skip1_hi = core1_hi - 4
        skip2_hi = core2_hi - 4
        skip3_hi = core3_hi - 4
        shot = {key: 0 for key in totals}

        for by in range(grid3):
            y0 = by * b3
            for bx in range(grid2):
                x0 = bx * b2
                for bz in range(grid1):
                    z0 = bz * b1
                    if (
                        skip1_hi > skip1_lo
                        and skip2_hi > skip2_lo
                        and skip3_hi > skip3_lo
                        and audit.tile_fully_inside_box(
                            z0,
                            x0,
                            y0,
                            b1,
                            b2,
                            b3,
                            n1,
                            n2,
                            n3,
                            skip1_lo,
                            skip1_hi,
                            skip2_lo,
                            skip2_hi,
                            skip3_lo,
                            skip3_hi,
                        )
                    ):
                        continue

                    shot["tiles"] += 1
                    shot["launched_lanes"] += volume
                    shot["warps"] += warps_per_block
                    z1 = min(z0 + b1, n1)
                    x1 = min(x0 + b2, n2)
                    y1 = min(y0 + b3, n3)
                    domain_volume = max(0, z1 - z0) * max(0, x1 - x0) * max(0, y1 - y0)
                    if domain_volume == 0:
                        continue

                    vx_inactive = audit.box_intersection_volume(
                        z0,
                        z1,
                        x0,
                        x1,
                        y0,
                        y1,
                        core1_lo,
                        core1_hi,
                        core2_lo + 3,
                        core2_hi - 4,
                        core3_lo,
                        core3_hi,
                    )
                    vy_inactive = audit.box_intersection_volume(
                        z0,
                        z1,
                        x0,
                        x1,
                        y0,
                        y1,
                        core1_lo,
                        core1_hi,
                        core2_lo,
                        core2_hi,
                        core3_lo + 3,
                        core3_hi - 4,
                    )
                    any_inactive = audit.box_intersection_volume(
                        z0,
                        z1,
                        x0,
                        x1,
                        y0,
                        y1,
                        skip1_lo,
                        skip1_hi,
                        skip2_lo,
                        skip2_hi,
                        skip3_lo,
                        skip3_hi,
                    )
                    vx_active = domain_volume - vx_inactive
                    vy_active = domain_volume - vy_inactive
                    active_any = domain_volume - any_inactive
                    component_lanes = vx_active + vy_active

                    shot["in_domain_lanes"] += domain_volume
                    shot["active_lanes"] += active_any
                    shot["vx_active_lanes"] += vx_active
                    shot["vy_active_lanes"] += vy_active
                    shot["component_lanes"] += component_lanes
                    shot["component_warps"] += ceil_div(component_lanes, 32)
                    shot["component_warp_active_lanes"] += component_lanes

                    # This is a static sector-unit model, not a hardware
                    # counter.  It preserves two gate facts: component work is
                    # invariant across valid shapes, while z<32 shapes split a
                    # warp across multiple discontiguous x/y rows.
                    split_factor = static_warp["avg_z_segments_per_warp"]
                    shot["p1_component_sector_units"] += component_lanes * split_factor
                    shot["p1_component_ideal_sector_units"] += component_lanes

        for key, value in shot.items():
            totals[key] += value
        shot_rows.append({"shot": domain.shot, **shot})

    active_eff = totals["active_lanes"] / totals["launched_lanes"] if totals["launched_lanes"] else 0.0
    component_eff = totals["component_lanes"] / totals["launched_lanes"] if totals["launched_lanes"] else 0.0
    avg_comp_lanes_per_warp = (
        totals["component_warp_active_lanes"] / totals["component_warps"] if totals["component_warps"] else 0.0
    )
    p1_sector_excess = (
        totals["p1_component_sector_units"] / totals["p1_component_ideal_sector_units"] - 1.0
        if totals["p1_component_ideal_sector_units"]
        else 0.0
    )
    return {
        "tile_block_zxy": [b1, b2, b3],
        "static_warp_shape": static_warp,
        "totals": totals,
        "derived": {
            "active_lane_efficiency": active_eff,
            "component_lane_efficiency": component_eff,
            "avg_active_lanes_per_component_warp": avg_comp_lanes_per_warp,
            "p1_sector_excess_over_active_contiguous": p1_sector_excess,
            "p1_sector_unit_note": "sector units are component_lanes weighted by warp z-segment split factor",
        },
        "shots": shot_rows,
    }


def compute_budget(case_dir: Path, profile_path: Path, source_path: Path | None, shapes: list[tuple[str, int, int, int]]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    case_meta, domains = audit.shot_domains(case_dir)
    npml = int(case_meta["npml"])
    profile = load_profile_anchor(profile_path)
    source = load_v_source_metrics(source_path)
    results: dict[str, Any] = {}
    for name, b1, b2, b3 in shapes:
        results[name] = analyze_shape(domains, npml, b1, b2, b3)

    current = results.get("current_32x4x2") or next(iter(results.values()))
    cur_totals = current["totals"]
    cur_derived = current["derived"]
    v_share = profile["v_pml_sampled_main_share"]

    for name, item in results.items():
        totals = item["totals"]
        derived = item["derived"]
        lane_ceiling = cur_totals["launched_lanes"] / totals["launched_lanes"]
        sector_ceiling = (
            cur_totals["p1_component_sector_units"] / totals["p1_component_sector_units"]
            if totals["p1_component_sector_units"]
            else 0.0
        )
        component_ceiling = (
            cur_totals["component_lanes"] / totals["component_lanes"] if totals["component_lanes"] else 0.0
        )
        optimistic_v_speedup = max(lane_ceiling, sector_ceiling, component_ceiling)
        item["vs_current"] = {
            "launched_lane_ratio": totals["launched_lanes"] / cur_totals["launched_lanes"],
            "active_lane_ratio": totals["active_lanes"] / cur_totals["active_lanes"],
            "component_lane_ratio": totals["component_lanes"] / cur_totals["component_lanes"],
            "p1_sector_ratio": totals["p1_component_sector_units"] / cur_totals["p1_component_sector_units"],
            "lane_speedup_ceiling": lane_ceiling,
            "p1_sector_speedup_ceiling": sector_ceiling,
            "component_work_speedup_ceiling": component_ceiling,
            "optimistic_v_kernel_speedup_ceiling": optimistic_v_speedup,
            "optimistic_sampled_main_speedup_ceiling": sampled_speedup(optimistic_v_speedup, v_share),
            "p1_sector_excess_delta": derived["p1_sector_excess_over_active_contiguous"]
            - cur_derived["p1_sector_excess_over_active_contiguous"],
        }

    best_name, best_item = max(
        results.items(),
        key=lambda pair: pair[1]["vs_current"]["optimistic_sampled_main_speedup_ceiling"] or 0.0,
    )
    best_speedup = best_item["vs_current"]["optimistic_sampled_main_speedup_ceiling"]
    best_v_speedup = best_item["vs_current"]["optimistic_v_kernel_speedup_ceiling"]
    required_v_speedup = None
    if v_share:
        target = 1.05
        required_v_speedup = 1.0 / ((1.0 / target - (1.0 - v_share)) / v_share)

    if best_speedup is not None and best_speedup >= 1.05:
        decision = "allow_profile_or_prototype"
        reason = (
            "At least one reasoned v-PML-only tile layout has an optimistic sampled-main ceiling above the "
            ">=5% gate.  Before CUDA implementation, verify this candidate with a dedicated NCU profile plan."
        )
    else:
        decision = "reject_tile_layout_cuda_prototype"
        reason = (
            "The current 32x4x2 layout already maps each warp to one contiguous z-line segment. "
            "The best reasoned v-only shape does not reach the >=5% sampled-main ceiling, and the real "
            "implementation would also need separate velocity tile-list plumbing plus pressure-path compatibility."
        )

    return {
        "case": {
            "case_dir": (
                str(case_dir.relative_to(root)).replace("\\", "/")
                if case_dir.is_relative_to(root)
                else str(case_dir)
            ),
            "npml": npml,
            "core_pml_margin": CORE_PML_MARGIN,
            "shape_count": len(shapes),
        },
        "profile_anchor": profile,
        "v_source_anchor": source,
        "shapes": results,
        "gate": {
            "decision": decision,
            "best_shape": best_name,
            "best_optimistic_v_kernel_speedup_ceiling": best_v_speedup,
            "best_optimistic_sampled_main_speedup_ceiling": best_speedup,
            "required_v_kernel_speedup_for_5pct_sampled_main": required_v_speedup,
            "reason": reason,
            "next_allowed": [
                "profile current-best v_pml source hotlines if source-level evidence is stale",
                "design a memory-ownership change that reduces global vx/vy round trip without doubling component work",
                "revisit v-PML only if a new model shows >=5% perf_1gpu_6shots repeat ceiling after tile-list overhead",
            ],
            "prohibited": [
                "random PmlTileBlockSize sweep",
                "current-geometry vx/vy component split",
                "v-only tile-layout CUDA prototype below the 5% gate",
            ],
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    profile = result["profile_anchor"]
    gate = result["gate"]
    lines = [
        "# V-PML Coalescing/Layout Budget",
        "",
        "## Context",
        "",
        f"- case: `{result['case']['case_dir']}`",
        f"- npml/core margin: `{result['case']['npml']}/{result['case']['core_pml_margin']}`",
        f"- NCU anchor profile: `{profile['profile_label']}`",
        f"- sampled main: `{profile['sampled_main_us']:.3f}us`",
        f"- v-PML duration/share: `{profile['v_pml_us']:.3f}us` / `{pct(profile['v_pml_sampled_main_share'])}`",
        f"- v-kernel speedup required for 5% sampled-main gain: `{ratio(gate['required_v_kernel_speedup_for_5pct_sampled_main'])}`",
        "",
        "The current CUDA mapping uses `threadIdx.x` as the z index.  With the accepted `32x4x2` tile, each warp is one contiguous z-line at fixed x/y.  That is already the favorable coalescing shape for the `p1`, `mem_dx`, and `mem_dy` paths.",
        "",
        "## Candidate Shapes",
        "",
        "| shape | block z/x/y | tiles | launched lanes ratio | active lane eff | component density | warp split factor | p1 sector-unit excess | p1 sector-unit ratio | optimistic v speedup | sampled-main ceiling |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in result["shapes"].items():
        totals = item["totals"]
        derived = item["derived"]
        static = item["static_warp_shape"]
        vs = item["vs_current"]
        lines.append(
            f"| `{name}` | `{item['tile_block_zxy'][0]}/{item['tile_block_zxy'][1]}/{item['tile_block_zxy'][2]}` | "
            f"{totals['tiles']} | {vs['launched_lane_ratio']:.4f} | "
            f"{pct(derived['active_lane_efficiency'])} | {pct(derived['component_lane_efficiency'])} | "
            f"{static['avg_z_segments_per_warp']:.2f} avg / {static['max_z_segments_per_warp']} max | "
            f"{pct(derived['p1_sector_excess_over_active_contiguous'])} | "
            f"{vs['p1_sector_ratio']:.4f} | {ratio(vs['optimistic_v_kernel_speedup_ceiling'])} | "
            f"{ratio(vs['optimistic_sampled_main_speedup_ceiling'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `32x4x2` is not an arbitrary local optimum: it gives one contiguous z segment per warp.",
        "- `component density` counts `vx + vy` work per launched lane, so it can be above 100%.",
        "- Shapes with `z < 32` split a warp across multiple x/y lines.  Some remove launched lanes, but the most optimistic sampled-main ceiling is still only about 2.6% before implementation overhead.",
        "- Larger-z or x/y-rebalanced shapes do not help this perf case; they pack more inactive interior/margin lanes into each velocity tile or keep the same component work.",
            "- The earlier rejected vx/vy split remains rejected because it nearly doubles component work under this geometry.",
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- best shape: `{gate['best_shape']}`",
            f"- best optimistic v-kernel speedup ceiling: `{ratio(gate['best_optimistic_v_kernel_speedup_ceiling'])}`",
            f"- best optimistic sampled-main ceiling: `{ratio(gate['best_optimistic_sampled_main_speedup_ceiling'])}`",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="benchmarks/cases/perf_1gpu_6shots")
    parser.add_argument("--profile-summary", default="reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json")
    parser.add_argument("--v-source-summary", default="reports/day_20260608/directfill_v_pml_source_summary.json")
    parser.add_argument("--shape", action="append", help="Reasoned shape, e.g. current_32x4x2=32,4,2")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case)
    if not case_dir.is_absolute():
        case_dir = root / case_dir
    profile = Path(args.profile_summary)
    if not profile.is_absolute():
        profile = root / profile
    source = Path(args.v_source_summary) if args.v_source_summary else None
    if source is not None and not source.is_absolute():
        source = root / source
    shapes = [parse_shape(item) for item in args.shape] if args.shape else DEFAULT_SHAPES

    result = compute_budget(case_dir, profile, source, shapes)

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
