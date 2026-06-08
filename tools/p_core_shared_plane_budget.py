#!/usr/bin/env python3
"""Budget p_core shared-plane stencil candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FLOAT_BYTES = 4
RADIUS = 7
CURRENT_Z = 128
CURRENT_X = 2
CURRENT_Y = 1
CURRENT_THREADS = CURRENT_Z * CURRENT_X * CURRENT_Y


def ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}x"


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def load_len16_profile(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for profile in data["profiles"]:
        if profile["label"] != "len16":
            continue
        kernels = profile["kernels"]
        p_core = kernels["cuda_fd3d_p_core_ns"]["metrics"]
        v_pml = kernels["cuda_fd3d_v_pml_tile_ns"]["metrics"]
        p_res = kernels["cuda_fd3d_p_pml_tile_ns"]["metrics"]
        p_len16 = kernels["cuda_fd3d_p_pml_len16_halfwarp_ns"]["metrics"]
        return {
            "p_core_us": p_core["duration_ns"] / 1000.0,
            "v_pml_us": v_pml["duration_ns"] / 1000.0,
            "p_pml_residual_us": p_res["duration_ns"] / 1000.0,
            "p_pml_len16_us": p_len16["duration_ns"] / 1000.0,
            "p_core_l2_hit_rate_pct": p_core["l2_hit_rate_pct"],
            "p_core_sol_l2_pct": p_core["sol_l2_throughput_pct"],
            "p_core_sol_memory_pct": p_core["sol_memory_throughput_pct"],
            "p_core_avg_active_threads": p_core["avg_active_threads_per_warp"],
            "p_core_eligible_warps_per_scheduler": p_core["eligible_warps_per_scheduler"],
        }
    raise ValueError(f"no len16 profile found in {path}")


def sampled_speedup_from_local(local_share: float, local_speedup: float) -> float:
    return 1.0 / ((1.0 - local_share) + local_share / local_speedup)


def current_p1_loads_per_output() -> float:
    # Current p_core loads the z-line into shared memory and reads x/y neighbors
    # directly from global memory.
    z_shared = (CURRENT_Z + 2 * RADIUS) * CURRENT_X * CURRENT_Y / CURRENT_THREADS
    xy_global = 4 * RADIUS
    return z_shared + xy_global


def candidate(shape: tuple[int, int, int], mode: str, current_p1: float, p_core_share: float) -> dict[str, Any]:
    z, x, y = shape
    outputs = z * x * y
    if outputs != CURRENT_THREADS:
        raise ValueError(f"candidate {shape} has {outputs} outputs, expected {CURRENT_THREADS}")

    if mode == "zx_shared_y_global":
        p1_loads = ((z + 2 * RADIUS) * (x + 2 * RADIUS) * y / outputs) + 2 * RADIUS
        smem_floats = (z + 2 * RADIUS) * (x + 2 * RADIUS) * y
        description = "share z+x plane, keep y-neighbor loads global"
    elif mode == "zy_shared_x_global":
        p1_loads = ((z + 2 * RADIUS) * x * (y + 2 * RADIUS) / outputs) + 2 * RADIUS
        smem_floats = (z + 2 * RADIUS) * x * (y + 2 * RADIUS)
        description = "share z+y plane, keep x-neighbor loads global"
    elif mode == "zxy_shared":
        p1_loads = (z + 2 * RADIUS) * (x + 2 * RADIUS) * (y + 2 * RADIUS) / outputs
        smem_floats = (z + 2 * RADIUS) * (x + 2 * RADIUS) * (y + 2 * RADIUS)
        description = "share full z+x+y tile"
    else:
        raise ValueError(mode)

    current_bytes = (current_p1 + 3.0) * FLOAT_BYTES
    candidate_bytes = (p1_loads + 3.0) * FLOAT_BYTES
    local_byte_speedup = current_bytes / candidate_bytes
    sampled_ceiling = sampled_speedup_from_local(p_core_share, local_byte_speedup)
    return {
        "shape_zxy": [z, x, y],
        "mode": mode,
        "description": description,
        "outputs_per_cta": outputs,
        "p1_global_floats_per_output": p1_loads,
        "bytes_per_output": candidate_bytes,
        "p1_load_reduction": 1.0 - p1_loads / current_p1,
        "p_core_byte_speedup_ceiling": local_byte_speedup,
        "sampled_main_speedup_ceiling": sampled_ceiling,
        "shared_bytes": smem_floats * FLOAT_BYTES,
        "shared_floats": smem_floats,
        "z_threads": z,
        "x_threads": x,
        "y_threads": y,
    }


def analyze(profile_path: Path, target_speedup: float) -> dict[str, Any]:
    profile = load_len16_profile(profile_path)
    sampled_main = (
        profile["p_core_us"]
        + profile["v_pml_us"]
        + profile["p_pml_residual_us"]
        + profile["p_pml_len16_us"]
    )
    p_core_share = profile["p_core_us"] / sampled_main
    current_p1 = current_p1_loads_per_output()
    current_bytes = (current_p1 + 3.0) * FLOAT_BYTES

    shapes = [
        (16, 16, 1),
        (16, 8, 2),
        (8, 16, 2),
        (8, 8, 4),
        (32, 8, 1),
        (32, 4, 2),
        (64, 4, 1),
        (64, 2, 2),
    ]
    modes = ["zx_shared_y_global", "zy_shared_x_global", "zxy_shared"]
    candidates = []
    for shape in shapes:
        for mode in modes:
            candidates.append(candidate(shape, mode, current_p1, p_core_share))
    candidates.sort(key=lambda item: item["sampled_main_speedup_ceiling"], reverse=True)

    best = candidates[0]
    gate_decision = "allow_cuda_prototype" if best["sampled_main_speedup_ceiling"] >= target_speedup else "reject_cuda_prototype"
    gate = {
        "decision": gate_decision,
        "candidate": "CUDA3D_P_CORE_SHARED_ZX_PLANE" if gate_decision == "allow_cuda_prototype" else "p_core_shared_plane",
        "reason": (
            "A z+x shared-plane p_core kernel can reduce estimated p1 global loads enough to pass the "
            ">=5% sampled-main modeling gate.  This is a dataflow change, not another simple block/register sweep."
            if gate_decision == "allow_cuda_prototype"
            else "No shared-plane candidate reaches the >=5% sampled-main modeling gate after p_core share is applied."
        ),
        "prototype_guardrails": [
            "macro default-off",
            "preserve current second-order p0/p1/cw2 math",
            "profile/debug/correctness/perf_1gpu_6shots repeat before acceptance",
            "stop immediately if repeat speedup <5% or correctness exceeds tolerance",
        ],
    }

    return {
        "inputs": {
            "profile_json": str(profile_path),
            "target_sampled_main_speedup": target_speedup,
            "radius": RADIUS,
            "current_shape_zxy": [CURRENT_Z, CURRENT_X, CURRENT_Y],
        },
        "profile": {
            **profile,
            "sampled_main_us": sampled_main,
            "p_core_share": p_core_share,
        },
        "current": {
            "p1_global_floats_per_output": current_p1,
            "bytes_per_output": current_bytes,
            "z_shared_floats_per_output": (CURRENT_Z + 2 * RADIUS) * CURRENT_X * CURRENT_Y / CURRENT_THREADS,
            "xy_global_floats_per_output": 4 * RADIUS,
        },
        "candidates": candidates,
        "gate": gate,
    }


def render_markdown(result: dict[str, Any]) -> str:
    profile = result["profile"]
    current = result["current"]
    lines = [
        "# P-Core Shared-Plane Stencil Budget",
        "",
        "## Context",
        "",
        f"- sampled main: `{profile['sampled_main_us']:.3f}us`",
        f"- p_core: `{profile['p_core_us']:.3f}us` / `{pct(profile['p_core_share'])}`",
        f"- p_core L2 SOL: `{profile['p_core_sol_l2_pct']:.2f}%`",
        f"- p_core L2 hit rate: `{profile['p_core_l2_hit_rate_pct']:.2f}%`",
        f"- current p_core block z/x/y: `{result['inputs']['current_shape_zxy']}`",
        "",
        "## Current Byte Model",
        "",
        f"- current p1 global floats/output: `{current['p1_global_floats_per_output']:.6f}`",
        f"- current bytes/output including p0/cw2/store: `{current['bytes_per_output']:.3f}`",
        f"- z shared load contribution: `{current['z_shared_floats_per_output']:.6f}` floats/output",
        f"- x/y global neighbor contribution: `{current['xy_global_floats_per_output']:.6f}` floats/output",
        "",
        "## Top Candidates",
        "",
        "| candidate | mode | p1 floats/out | bytes/out | shared KiB | p_core byte ceiling | sampled-main ceiling |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["candidates"][:10]:
        lines.append(
            f"| `{item['shape_zxy']}` | `{item['mode']}` | `{item['p1_global_floats_per_output']:.3f}` | "
            f"`{item['bytes_per_output']:.3f}` | `{item['shared_bytes'] / 1024.0:.2f}` | "
            f"`{ratio(item['p_core_byte_speedup_ceiling'])}` | `{ratio(item['sampled_main_speedup_ceiling'])}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- decision: `{result['gate']['decision']}`",
            f"- candidate: `{result['gate']['candidate']}`",
            f"- reason: {result['gate']['reason']}",
            "",
            "## Guardrails",
            "",
        ]
    )
    for item in result["gate"]["prototype_guardrails"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-json",
        default="reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json",
    )
    parser.add_argument("--target-speedup", type=float, default=1.05)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    profile_path = Path(args.profile_json)
    if not profile_path.is_absolute():
        profile_path = root / profile_path
    result = analyze(profile_path, args.target_speedup)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_markdown(result)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(md)


if __name__ == "__main__":
    main()
