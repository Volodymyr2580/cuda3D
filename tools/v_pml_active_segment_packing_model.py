#!/usr/bin/env python3
"""Model v-PML active z-segment packing candidates.

This is a gate before opening another CUDA prototype.  The accepted pressure
PML path gained speed by packing whole length-16 z-lines into half warps.  This
tool checks whether the velocity PML path has the same exploitable structure
under the current 32x4x2 tile ownership.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pml_pressure_dataflow_audit as audit  # noqa: E402
import v_pml_coalescing_layout_budget as v_layout  # noqa: E402


CORE_PML_MARGIN = 4
DEFAULT_B1 = 32
DEFAULT_B2 = 4
DEFAULT_B3 = 2


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


def need_vx(z: int, x: int, y: int, core: dict[str, int]) -> bool:
    return not (
        core["z_lo"] <= z < core["z_hi"]
        and core["x_lo"] + 3 <= x < core["x_hi"] - 4
        and core["y_lo"] <= y < core["y_hi"]
    )


def need_vy(z: int, x: int, y: int, core: dict[str, int]) -> bool:
    return not (
        core["z_lo"] <= z < core["z_hi"]
        and core["x_lo"] <= x < core["x_hi"]
        and core["y_lo"] + 3 <= y < core["y_hi"] - 4
    )


def old_velocity_tile_skip(
    z0: int,
    x0: int,
    y0: int,
    b1: int,
    b2: int,
    b3: int,
    n1: int,
    n2: int,
    n3: int,
    core: dict[str, int],
) -> bool:
    """Mirror build_pml_tile_list(..., for_velocity=1) without v-tile prune.

    The current-best flags do not enable CUDA3D_PML_ZMEM_V_TILE_PRUNE, and that
    route is prohibited by the architecture log.  The launched lane model
    therefore follows the existing host tile list, while active lanes below use
    the true vx/vy needs when CUDA3D_PML_ZMEM_IN_P makes vz unnecessary.
    """

    return (
        core["z_hi"] - 4 > core["z_lo"] + 3
        and core["x_hi"] - 4 > core["x_lo"] + 3
        and core["y_hi"] - 4 > core["y_lo"] + 3
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
            core["z_lo"] + 3,
            core["z_hi"] - 4,
            core["x_lo"] + 3,
            core["x_hi"] - 4,
            core["y_lo"] + 3,
            core["y_hi"] - 4,
        )
    )


def analyze(case_dir: Path, profile_summary: Path, b1: int, b2: int, b3: int) -> dict[str, Any]:
    case_meta, domains = audit.shot_domains(case_dir)
    npml = int(case_meta["npml"])
    profile = v_layout.load_profile_anchor(profile_summary)
    v_share = profile["v_pml_sampled_main_share"]
    target = 1.05
    required_v_speedup = (
        1.0 / ((1.0 / target - (1.0 - v_share)) / v_share)
        if v_share
        else None
    )

    length_hist: collections.Counter[int] = collections.Counter()
    component_hist: collections.Counter[int] = collections.Counter()
    whole_tile_hist: collections.Counter[str] = collections.Counter()
    shot_rows: list[dict[str, Any]] = []
    totals = {
        "tiles": 0,
        "current_lanes": 0,
        "in_domain_lanes": 0,
        "active_any_lanes": 0,
        "vx_lanes": 0,
        "vy_lanes": 0,
        "component_lanes": 0,
        "line_slots": 0,
        "empty_line_slots": 0,
        "len16_line_slots": 0,
        "whole_len16_tiles": 0,
        "whole_len16_current_lanes": 0,
        "whole_len16_packed_lanes": 0,
    }

    for domain in domains:
        n3, n2, n1 = domain.n3n2n1
        grid1 = ceil_div(n1, b1)
        grid2 = ceil_div(n2, b2)
        grid3 = ceil_div(n3, b3)
        core = {
            "z_lo": npml + CORE_PML_MARGIN,
            "x_lo": npml + CORE_PML_MARGIN,
            "y_lo": npml + CORE_PML_MARGIN,
            "z_hi": n1 - npml - CORE_PML_MARGIN,
            "x_hi": n2 - npml - CORE_PML_MARGIN,
            "y_hi": n3 - npml - CORE_PML_MARGIN,
        }
        shot = {key: 0 for key in totals}
        shot_length_hist: collections.Counter[int] = collections.Counter()

        for by in range(grid3):
            y0 = by * b3
            for bx in range(grid2):
                x0 = bx * b2
                for bz in range(grid1):
                    z0 = bz * b1
                    if old_velocity_tile_skip(z0, x0, y0, b1, b2, b3, n1, n2, n3, core):
                        continue
                    z1 = min(z0 + b1, n1)
                    x1 = min(x0 + b2, n2)
                    y1 = min(y0 + b3, n3)
                    shot["tiles"] += 1
                    shot["current_lanes"] += b1 * b2 * b3

                    tile_lengths: list[int] = []
                    tile_full_xy = (x1 - x0 == b2) and (y1 - y0 == b3)
                    for y in range(y0, y1):
                        for x in range(x0, x1):
                            any_len = 0
                            vx_len = 0
                            vy_len = 0
                            for z in range(z0, z1):
                                vx = need_vx(z, x, y, core)
                                vy = need_vy(z, x, y, core)
                                vx_len += int(vx)
                                vy_len += int(vy)
                                any_len += int(vx or vy)
                            comp_len = vx_len + vy_len
                            shot["line_slots"] += 1
                            shot["in_domain_lanes"] += z1 - z0
                            shot["active_any_lanes"] += any_len
                            shot["vx_lanes"] += vx_len
                            shot["vy_lanes"] += vy_len
                            shot["component_lanes"] += comp_len
                            if any_len == 0:
                                shot["empty_line_slots"] += 1
                            if any_len == 16:
                                shot["len16_line_slots"] += 1
                            length_hist[any_len] += 1
                            component_hist[comp_len] += 1
                            shot_length_hist[any_len] += 1
                            tile_lengths.append(any_len)

                    full_tile_lines = b2 * b3
                    whole_len16 = (
                        tile_full_xy
                        and len(tile_lengths) == full_tile_lines
                        and all(length == 16 for length in tile_lengths)
                    )
                    if whole_len16:
                        shot["whole_len16_tiles"] += 1
                        shot["whole_len16_current_lanes"] += full_tile_lines * 32
                        shot["whole_len16_packed_lanes"] += (full_tile_lines // 2) * 32
                        mask = audit.mask_label(audit.make_pml_tile_mask(z0, x0, y0, b1, b2, b3, n1, n2, n3, npml))
                        whole_tile_hist[mask] += 1

        for key, value in shot.items():
            totals[key] += value
        shot_rows.append(
            {
                "shot": domain.shot,
                "domain_size_yx": domain.domain_size_yx,
                **shot,
                "length_hist": dict(sorted(shot_length_hist.items())),
            }
        )

    current_lanes = totals["current_lanes"]
    nonempty_lines = totals["line_slots"] - totals["empty_line_slots"]
    len16_lines = totals["len16_line_slots"]
    whole16_lanes = current_lanes - totals["whole_len16_current_lanes"] + totals["whole_len16_packed_lanes"]
    line_packed_lanes = (
        ceil_div(len16_lines, 2) * 32
        + (nonempty_lines - len16_lines) * 32
    )
    exact_point_lanes = ceil_div(totals["active_any_lanes"], 256) * 256

    candidates: dict[str, dict[str, float | int | None | str]] = {
        "whole_len16_tile_halfwarp_pack": {
            "lanes": whole16_lanes,
            "v_lane_speedup_ceiling": current_lanes / whole16_lanes if whole16_lanes else None,
            "implementation_class": "tile_list_only",
        },
        "line_descriptor_len16_pack_remove_empty": {
            "lanes": line_packed_lanes,
            "v_lane_speedup_ceiling": current_lanes / line_packed_lanes if line_packed_lanes else None,
            "implementation_class": "line_descriptor",
        },
        "exact_active_point_list": {
            "lanes": exact_point_lanes,
            "v_lane_speedup_ceiling": current_lanes / exact_point_lanes if exact_point_lanes else None,
            "implementation_class": "point_descriptor",
        },
    }
    for item in candidates.values():
        lane_speed = item["v_lane_speedup_ceiling"]
        item["sampled_main_speedup_ceiling"] = (
            sampled_speedup(float(lane_speed), v_share) if lane_speed else None
        )
        item["lane_reduction_vs_current"] = (
            1.0 - float(item["lanes"]) / current_lanes if item["lanes"] else None
        )

    best_name, best_item = max(
        candidates.items(),
        key=lambda pair: pair[1]["sampled_main_speedup_ceiling"] or 0.0,
    )

    whole_speed = candidates["whole_len16_tile_halfwarp_pack"]["sampled_main_speedup_ceiling"]
    if whole_speed is not None and whole_speed >= 1.05:
        decision = "allow_whole_len16_v_pml_cuda_prototype"
        reason = (
            "Whole-tile length-16 velocity lines clear the >=5% sampled-main gate without a line descriptor. "
            "Open a macro-default-off CUDA prototype that mirrors the pressure len16 split."
        )
    else:
        decision = "reject_v_pml_len16_tile_pack_prototype"
        reason = (
            "The tile-list-only length-16 packing path does not clear the >=5% sampled-main gate. "
            "More aggressive line/point descriptor variants are treated as design-only unless their "
            "descriptor and control costs are explicitly modeled and still clear the gate."
        )

    return {
        "case": {
            "case_dir": str(case_dir),
            "npml": npml,
            "core_pml_margin": CORE_PML_MARGIN,
            "tile_block_zxy": [b1, b2, b3],
        },
        "profile_anchor": {
            **profile,
            "required_v_kernel_speedup_for_5pct_sampled_main": required_v_speedup,
        },
        "totals": totals,
        "length_hist": dict(sorted(length_hist.items())),
        "component_hist": dict(sorted(component_hist.items())),
        "whole_len16_tile_mask_hist": dict(sorted(whole_tile_hist.items())),
        "candidates": candidates,
        "descriptor_bytes": {
            "line_descriptor_uint64_mib_per_step_aggregate_shots": totals["line_slots"] * 8 / (1024.0 * 1024.0),
            "point_descriptor_uint32_mib_per_step_aggregate_shots": totals["active_any_lanes"] * 4 / (1024.0 * 1024.0),
        },
        "gate": {
            "decision": decision,
            "best_candidate": best_name,
            "best_sampled_main_speedup_ceiling": best_item["sampled_main_speedup_ceiling"],
            "best_v_lane_speedup_ceiling": best_item["v_lane_speedup_ceiling"],
            "reason": reason,
            "prohibited_if_rejected": [
                "Do not write a v-PML whole-len16 half-warp CUDA prototype below the >=5% gate.",
                "Do not re-open random v-PML tile-shape sweep or current-geometry vx/vy split.",
                "Do not use line/point descriptors without a descriptor/control overhead model.",
            ],
        },
        "shots": shot_rows,
    }


def render_markdown(result: dict[str, Any]) -> str:
    totals = result["totals"]
    profile = result["profile_anchor"]
    gate = result["gate"]
    lines = [
        "# V-PML Active Segment Packing Model",
        "",
        "## Context",
        "",
        f"- case: `{result['case']['case_dir']}`",
        f"- tile block z/x/y: `{result['case']['tile_block_zxy'][0]}/{result['case']['tile_block_zxy'][1]}/{result['case']['tile_block_zxy'][2]}`",
        f"- NCU anchor profile: `{profile['profile_label']}`",
        f"- sampled main: `{profile['sampled_main_us']:.3f}us`",
        f"- v-PML duration/share: `{profile['v_pml_us']:.3f}us` / `{pct(profile['v_pml_sampled_main_share'])}`",
        f"- v-kernel speedup required for 5% sampled-main: `{ratio(profile['required_v_kernel_speedup_for_5pct_sampled_main'])}`",
        "",
        "## Current Lane Shape",
        "",
        f"- tiles: `{totals['tiles']}`",
        f"- current launched lanes: `{totals['current_lanes']}`",
        f"- in-domain line lanes: `{totals['in_domain_lanes']}`",
        f"- true vx/vy active-any lanes: `{totals['active_any_lanes']}`",
        f"- vx lanes: `{totals['vx_lanes']}`",
        f"- vy lanes: `{totals['vy_lanes']}`",
        f"- component lanes vx+vy: `{totals['component_lanes']}`",
        f"- active-any lane efficiency vs launched: `{pct(totals['active_any_lanes'] / totals['current_lanes'])}`",
        f"- component density vs launched: `{pct(totals['component_lanes'] / totals['current_lanes'])}`",
        f"- z-line slots: `{totals['line_slots']}`",
        f"- empty z-line slots: `{totals['empty_line_slots']}`",
        f"- length-16 z-line slots: `{totals['len16_line_slots']}`",
        f"- whole length-16 tiles: `{totals['whole_len16_tiles']}`",
        "",
        "## Active-Any Z-Line Histogram",
        "",
        "| active z length | line slots |",
        "| ---: | ---: |",
    ]
    for length, count in result["length_hist"].items():
        lines.append(f"| {length} | {count} |")
    lines.extend(
        [
            "",
            "## Component-Lane Histogram Per Z-Line",
            "",
            "| vx+vy component lanes | line slots |",
            "| ---: | ---: |",
        ]
    )
    for length, count in result["component_hist"].items():
        lines.append(f"| {length} | {count} |")
    lines.extend(
        [
            "",
            "## Candidate Ceilings",
            "",
            "| candidate | implementation class | lanes | lane reduction | v lane speedup ceiling | sampled-main ceiling |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, item in result["candidates"].items():
        lines.append(
            f"| `{name}` | `{item['implementation_class']}` | {item['lanes']} | "
            f"{pct(item['lane_reduction_vs_current'])} | {ratio(item['v_lane_speedup_ceiling'])} | "
            f"{ratio(item['sampled_main_speedup_ceiling'])} |"
        )
    lines.extend(
        [
            "",
            "## Descriptor Traffic",
            "",
            f"- line descriptor uint64: `{result['descriptor_bytes']['line_descriptor_uint64_mib_per_step_aggregate_shots']:.3f} MiB/step aggregate-shots`",
            f"- point descriptor uint32: `{result['descriptor_bytes']['point_descriptor_uint32_mib_per_step_aggregate_shots']:.3f} MiB/step aggregate-shots`",
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- best candidate: `{gate['best_candidate']}`",
            f"- best v lane speedup ceiling: `{ratio(gate['best_v_lane_speedup_ceiling'])}`",
            f"- best sampled-main speedup ceiling: `{ratio(gate['best_sampled_main_speedup_ceiling'])}`",
            f"- reason: {gate['reason']}",
            "",
            "Do not continue if rejected:",
            "",
        ]
    )
    for item in gate["prohibited_if_rejected"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="benchmarks/cases/perf_1gpu_6shots")
    parser.add_argument("--profile-summary", default="reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json")
    parser.add_argument("--tile-b1", type=int, default=DEFAULT_B1)
    parser.add_argument("--tile-b2", type=int, default=DEFAULT_B2)
    parser.add_argument("--tile-b3", type=int, default=DEFAULT_B3)
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

    result = analyze(case_dir, profile, args.tile_b1, args.tile_b2, args.tile_b3)

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
