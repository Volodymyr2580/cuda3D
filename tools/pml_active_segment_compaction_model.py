#!/usr/bin/env python3
"""Model pressure-PML active z-segment compaction candidates."""

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


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.2%}"


def ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}x"


def sampled_speedup(p_pml_speedup: float, p_pml_share: float | None) -> float | None:
    if p_pml_share is None or p_pml_speedup <= 0:
        return None
    return 1.0 / ((1.0 - p_pml_share) + p_pml_share / p_pml_speedup)


def ceil_to(value: int, group: int) -> int:
    return ((value + group - 1) // group) * group


def analyze_segments(case_dir: Path, ncu_summary: Path | None, b1: int, b2: int, b3: int, margin: int) -> dict[str, Any]:
    case_meta, domains = audit.shot_domains(case_dir)
    npml = int(case_meta["npml"])
    ncu = audit.read_ncu_summary(ncu_summary) if ncu_summary else None
    p_pml_duration = audit.ncu_kernel_duration_ns(ncu, "cuda_fd3d_p_pml") if ncu else None
    p_core_duration = audit.ncu_kernel_duration_ns(ncu, "cuda_fd3d_p_core") if ncu else None
    v_pml_duration = audit.ncu_kernel_duration_ns(ncu, "cuda_fd3d_v_pml") if ncu else None
    sampled_main = sum(v for v in (p_pml_duration, p_core_duration, v_pml_duration) if v is not None)
    p_pml_share = p_pml_duration / sampled_main if p_pml_duration and sampled_main else None

    length_hist: collections.Counter[int] = collections.Counter()
    mask_hist: collections.Counter[str] = collections.Counter()
    length_by_mask: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    shot_rows = []
    totals = {
        "kept_tiles": 0,
        "current_lanes": 0,
        "active_lanes": 0,
        "active_line_slots": 0,
    }

    for domain in domains:
        n3, n2, n1 = domain.n3n2n1
        grid1 = audit.ceil_div(n1, b1)
        grid2 = audit.ceil_div(n2, b2)
        grid3 = audit.ceil_div(n3, b3)
        core1_lo = npml + margin
        core2_lo = npml + margin
        core3_lo = npml + margin
        core1_hi = n1 - npml - margin
        core2_hi = n2 - npml - margin
        core3_hi = n3 - npml - margin
        shot_hist: collections.Counter[int] = collections.Counter()
        shot_active = 0
        shot_lines = 0
        shot_tiles = 0

        for by in range(grid3):
            y0 = by * b3
            y1 = min(y0 + b3, n3)
            for bx in range(grid2):
                x0 = bx * b2
                x1 = min(x0 + b2, n2)
                for bz in range(grid1):
                    z0 = bz * b1
                    z1 = min(z0 + b1, n1)
                    if audit.tile_fully_inside_box(
                        z0,
                        x0,
                        y0,
                        b1,
                        b2,
                        b3,
                        n1,
                        n2,
                        n3,
                        core1_lo,
                        core1_hi,
                        core2_lo,
                        core2_hi,
                        core3_lo,
                        core3_hi,
                    ):
                        continue
                    shot_tiles += 1
                    mask = audit.mask_label(audit.make_pml_tile_mask(z0, x0, y0, b1, b2, b3, n1, n2, n3, npml))
                    for y in range(y0, y1):
                        for x in range(x0, x1):
                            active_len = audit.active_line_points(
                                z0,
                                z1,
                                x,
                                y,
                                core1_lo,
                                core1_hi,
                                core2_lo,
                                core2_hi,
                                core3_lo,
                                core3_hi,
                            )
                            if active_len == 0:
                                continue
                            length_hist[active_len] += 1
                            length_by_mask[mask][active_len] += 1
                            mask_hist[mask] += 1
                            shot_hist[active_len] += 1
                            shot_lines += 1
                            shot_active += active_len

        totals["kept_tiles"] += shot_tiles
        totals["active_line_slots"] += shot_lines
        totals["active_lanes"] += shot_active
        totals["current_lanes"] += shot_tiles * b1 * b2 * b3
        shot_rows.append(
            {
                "shot": domain.shot,
                "domain_size_yx": domain.domain_size_yx,
                "kept_tiles": shot_tiles,
                "active_line_slots": shot_lines,
                "active_lanes": shot_active,
                "length_hist": dict(sorted(shot_hist.items())),
            }
        )

    current_lanes = totals["current_lanes"]
    active_lanes = totals["active_lanes"]
    line_slots = totals["active_line_slots"]
    line_list_lanes = ceil_to(line_slots, 8) * 32
    exact_point_lanes = ceil_to(active_lanes, 256)
    length16 = length_hist.get(16, 0)
    half16_lanes = ceil_to(length16, 2) * 16
    other_line_lanes = sum(count * 32 for length, count in length_hist.items() if length != 16)
    half16_segment_lanes = half16_lanes + other_line_lanes

    candidates = {
        "active_line_list_8warp": {
            "lanes": line_list_lanes,
            "lane_efficiency": active_lanes / line_list_lanes,
            "lane_reduction_vs_current": 1.0 - line_list_lanes / current_lanes,
            "p_pml_lane_speedup_ceiling": current_lanes / line_list_lanes,
        },
        "exact_active_point_list": {
            "lanes": exact_point_lanes,
            "lane_efficiency": active_lanes / exact_point_lanes,
            "lane_reduction_vs_current": 1.0 - exact_point_lanes / current_lanes,
            "p_pml_lane_speedup_ceiling": current_lanes / exact_point_lanes,
        },
        "pack_len16_halfwarp_plus_fullwarps": {
            "lanes": half16_segment_lanes,
            "lane_efficiency": active_lanes / half16_segment_lanes,
            "lane_reduction_vs_current": 1.0 - half16_segment_lanes / current_lanes,
            "p_pml_lane_speedup_ceiling": current_lanes / half16_segment_lanes,
        },
    }
    for item in candidates.values():
        item["sampled_main_speedup_ceiling"] = sampled_speedup(item["p_pml_lane_speedup_ceiling"], p_pml_share)

    descriptor_bytes = {
        "uint32_point_index_per_active_point_mib_per_step_aggregate_shots": active_lanes * 4 / (1024.0 * 1024.0),
        "uint64_line_descriptor_per_active_line_mib_per_step_aggregate_shots": line_slots * 8 / (1024.0 * 1024.0),
    }
    gate = {
        "active_line_list_8warp": "reject",
        "pack_len16_halfwarp_plus_fullwarps": "design_only",
        "exact_active_point_list": "design_only",
        "reason": (
            "Simple active-line compaction only removes empty lines and has about a 1.0196x p_pml lane ceiling. "
            "The meaningful ceiling comes from packing length-16 z-face/margin lines or exact active points, "
            "but those designs must preserve the accepted z-cache dataflow before CUDA implementation."
        ),
    }
    return {
        "case": {
            "case_dir": str(case_dir),
            "tile_block_zxy": [b1, b2, b3],
            "npml": npml,
            "core_pml_margin": margin,
        },
        "ncu_link": {
            "p_pml_duration_ns": p_pml_duration,
            "p_core_duration_ns": p_core_duration,
            "v_pml_duration_ns": v_pml_duration,
            "p_pml_sampled_main_share": p_pml_share,
        },
        "totals": totals,
        "length_hist": dict(sorted(length_hist.items())),
        "mask_line_slots": dict(sorted(mask_hist.items())),
        "length_hist_by_mask": {key: dict(sorted(value.items())) for key, value in sorted(length_by_mask.items())},
        "candidates": candidates,
        "descriptor_bytes": descriptor_bytes,
        "gate": gate,
        "shots": shot_rows,
    }


def render_markdown(result: dict[str, Any]) -> str:
    totals = result["totals"]
    lines = [
        "# PML Active Segment Compaction Model",
        "",
        "## Context",
        "",
        f"- case_dir: `{result['case']['case_dir']}`",
        f"- tile block z/x/y: `{result['case']['tile_block_zxy'][0]}/{result['case']['tile_block_zxy'][1]}/{result['case']['tile_block_zxy'][2]}`",
        f"- npml/core_pml_margin: `{result['case']['npml']}/{result['case']['core_pml_margin']}`",
        f"- p_pml sampled-main share: `{pct(result['ncu_link']['p_pml_sampled_main_share'])}`",
        "",
        "## Current Lane Shape",
        "",
        f"- kept tiles: `{totals['kept_tiles']}`",
        f"- current launched lanes: `{totals['current_lanes']}`",
        f"- active lanes after core return: `{totals['active_lanes']}`",
        f"- current lane efficiency: `{pct(totals['active_lanes'] / totals['current_lanes'])}`",
        f"- active line slots: `{totals['active_line_slots']}`",
        f"- average active lanes per active line: `{totals['active_lanes'] / totals['active_line_slots']:.3f}`",
        "",
        "## Active Z-Line Length Histogram",
        "",
        "| active z length | line slots | active lanes |",
        "| ---: | ---: | ---: |",
    ]
    for length, count in result["length_hist"].items():
        lines.append(f"| {length} | {count} | {int(length) * int(count)} |")

    lines.extend(
        [
            "",
            "## Candidate Lane Ceilings",
            "",
            "| candidate | lanes | lane efficiency | lane reduction | p_pml lane speedup ceiling | sampled-main ceiling | gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    gates = result["gate"]
    for name, item in result["candidates"].items():
        lines.append(
            f"| `{name}` | {item['lanes']} | {pct(item['lane_efficiency'])} | "
            f"{pct(item['lane_reduction_vs_current'])} | {ratio(item['p_pml_lane_speedup_ceiling'])} | "
            f"{ratio(item['sampled_main_speedup_ceiling'])} | `{gates.get(name, 'n/a')}` |"
        )

    lines.extend(
        [
            "",
            "## Descriptor Traffic",
            "",
            f"- point list with one uint32 per active point: `{result['descriptor_bytes']['uint32_point_index_per_active_point_mib_per_step_aggregate_shots']:.3f} MiB/step aggregate-shots`",
            f"- line list with one uint64 per active line: `{result['descriptor_bytes']['uint64_line_descriptor_per_active_line_mib_per_step_aggregate_shots']:.3f} MiB/step aggregate-shots`",
            "",
            "## Gate",
            "",
            f"- active-line list: `{gates['active_line_list_8warp']}`",
            f"- length-16 half-warp packing: `{gates['pack_len16_halfwarp_plus_fullwarps']}`",
            f"- exact active-point list: `{gates['exact_active_point_list']}`",
            f"- reason: {gates['reason']}",
            "",
            "## Design Boundary",
            "",
            "A CUDA prototype is not opened by the simple active-line list, because it only removes about `1.92%` of launched lanes.",
            "",
            "A future prototype may be opened only if it preserves the accepted pressure z-cache dataflow while packing the length-16 z-face/margin lines or exact active points. This is a lane-utilization design, not a repeat of the rejected z-face direct-derivative/fusion route.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="benchmarks/cases/perf_1gpu_6shots")
    parser.add_argument("--ncu-summary-json", default="reports/day_20260608/directfill_combo_ncu_20260608_120449_summary.json")
    parser.add_argument("--tile-b1", type=int, default=32)
    parser.add_argument("--tile-b2", type=int, default=4)
    parser.add_argument("--tile-b3", type=int, default=2)
    parser.add_argument("--core-pml-margin", type=int, default=4)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case)
    if not case_dir.is_absolute():
        case_dir = root / case_dir
    ncu_path = Path(args.ncu_summary_json) if args.ncu_summary_json else None
    if ncu_path and not ncu_path.is_absolute():
        ncu_path = root / ncu_path

    result = analyze_segments(case_dir, ncu_path, args.tile_b1, args.tile_b2, args.tile_b3, args.core_pml_margin)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_markdown(result)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(md)


if __name__ == "__main__":
    main()
