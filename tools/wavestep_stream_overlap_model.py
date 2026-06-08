#!/usr/bin/env python3
"""Model wave-step stream overlap candidates.

This is a scheduling/ownership gate before writing CUDA stream code.  It uses
the accepted len16 Nsight Compute summary to estimate whether `p_core` can be
profitably overlapped with the independent PML path in one wave step.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}x"


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def duration_us(kernels: dict[str, Any], name: str) -> float:
    item = kernels.get(name)
    if item is None:
        return 0.0
    return float(item["metrics"]["duration_ns"]) / 1000.0


def load_len16_timings(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    profiles = {item["label"]: item for item in data["profiles"]}
    profile = profiles["len16"]
    kernels = profile["kernels"]
    core = duration_us(kernels, "cuda_fd3d_p_core_ns")
    v_pml = duration_us(kernels, "cuda_fd3d_v_pml_tile_ns")
    p_residual = duration_us(kernels, "cuda_fd3d_p_pml_tile_ns")
    p_len16 = duration_us(kernels, "cuda_fd3d_p_pml_len16_halfwarp_ns")
    p_total = p_residual + p_len16
    sequential = core + v_pml + p_total
    return {
        "profile_label": "len16",
        "p_core_us": core,
        "v_pml_us": v_pml,
        "p_pml_residual_us": p_residual,
        "p_pml_len16_us": p_len16,
        "p_pml_total_us": p_total,
        "sequential_sampled_main_us": sequential,
    }


def speedup(base: float, candidate: float) -> float:
    if candidate <= 0:
        return 0.0
    return base / candidate


def required_overlap_efficiency(base: float, overlap_window: float, target_speedup: float) -> float | None:
    if overlap_window <= 0:
        return None
    target_time = base / target_speedup
    required_saved = base - target_time
    return required_saved / overlap_window


def compute_model(t: dict[str, Any], target_speedup: float) -> dict[str, Any]:
    core = t["p_core_us"]
    v_pml = t["v_pml_us"]
    p_residual = t["p_pml_residual_us"]
    p_len16 = t["p_pml_len16_us"]
    p_total = t["p_pml_total_us"]
    base = t["sequential_sampled_main_us"]
    pml_serial = v_pml + p_total
    pml_parallel_pressure = v_pml + max(p_residual, p_len16)

    candidates = {
        "overlap_core_with_v_only": {
            "description": "Run p_core concurrently with v_pml only; pressure kernels remain after both.",
            "critical_path_us": max(core, v_pml) + p_total,
            "overlap_window_us": min(core, v_pml),
            "requires": [
                "two non-blocking streams",
                "event before pressure PML",
                "default stream waits before injection/extract",
            ],
        },
        "overlap_core_with_serial_pml_path": {
            "description": "Run p_core concurrently with the whole v_pml -> pressure-PML path.",
            "critical_path_us": max(core, pml_serial),
            "overlap_window_us": min(core, pml_serial),
            "requires": [
                "p_core stream independent from PML stream",
                "default stream waits for both streams before injection/extract",
                "next iteration streams wait for post-injection event",
            ],
        },
        "overlap_core_with_parallel_pressure_pml": {
            "description": "Also run residual and len16 pressure-PML kernels in separate streams after v_pml.",
            "critical_path_us": max(core, pml_parallel_pressure),
            "overlap_window_us": min(core, pml_parallel_pressure),
            "requires": [
                "three non-blocking streams",
                "v_pml done event fanout to residual and len16 pressure streams",
                "proof residual and len16 tile lists write disjoint p0/mem_dzz/mem_dz_next regions",
            ],
        },
    }
    for item in candidates.values():
        item["sampled_main_speedup_ceiling"] = speedup(base, item["critical_path_us"])
        item["required_overlap_efficiency_for_target"] = required_overlap_efficiency(
            base, item["overlap_window_us"], target_speedup
        )

    best_name, best_item = max(
        candidates.items(),
        key=lambda pair: pair[1]["sampled_main_speedup_ceiling"],
    )
    serial_item = candidates["overlap_core_with_serial_pml_path"]
    gate_decision = (
        "allow_cuda_prototype"
        if serial_item["required_overlap_efficiency_for_target"] is not None
        and serial_item["required_overlap_efficiency_for_target"] <= 0.35
        else "reject_cuda_prototype"
    )
    if gate_decision == "allow_cuda_prototype":
        reason = (
            "The conservative two-stream schedule only needs about "
            f"{pct(serial_item['required_overlap_efficiency_for_target'])} realized overlap to reach "
            f"{ratio(target_speedup)} sampled-main speedup, while preserving kernel math and ownership."
        )
    else:
        reason = (
            "The conservative stream schedule needs too much realized overlap for a first prototype."
        )

    return {
        "inputs": {
            "timings": t,
            "target_speedup": target_speedup,
            "notes": [
                "p_core reads p1/cw2 and writes p0 only in the core region.",
                "v_pml reads p1 and writes vx/vy plus CPML velocity state.",
                "pressure PML reads vx/vy and writes p0 only in PML regions.",
                "source injection/extraction must wait for both core and PML streams before pointer swap reuse.",
            ],
        },
        "candidates": candidates,
        "gate": {
            "decision": gate_decision,
            "best_candidate": best_name,
            "best_sampled_main_speedup_ceiling": best_item["sampled_main_speedup_ceiling"],
            "recommended_first_prototype": "overlap_core_with_serial_pml_path",
            "reason": reason,
            "prototype_macro": "CUDA3D_WAVESTEP_ASYNC_STREAMS",
            "prototype_boundaries": [
                "macro default off",
                "do not change CUDA math kernels",
                "do not overlap next time step before injection/extract completes",
                "first prototype uses two streams only; pressure residual/len16 parallel fanout requires a later gate",
            ],
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    t = result["inputs"]["timings"]
    gate = result["gate"]
    lines = [
        "# Wave-Step Stream Overlap Model",
        "",
        "## Context",
        "",
        f"- profile: `{t['profile_label']}`",
        f"- sequential sampled main: `{t['sequential_sampled_main_us']:.3f}us`",
        f"- p_core: `{t['p_core_us']:.3f}us`",
        f"- v_pml: `{t['v_pml_us']:.3f}us`",
        f"- pressure residual: `{t['p_pml_residual_us']:.3f}us`",
        f"- pressure len16: `{t['p_pml_len16_us']:.3f}us`",
        f"- pressure total: `{t['p_pml_total_us']:.3f}us`",
        f"- target sampled-main speedup: `{ratio(result['inputs']['target_speedup'])}`",
        "",
        "Dependency facts:",
        "",
    ]
    for item in result["inputs"]["notes"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| candidate | critical path | ceiling | overlap window | required overlap for target |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, item in result["candidates"].items():
        lines.append(
            f"| `{name}` | `{item['critical_path_us']:.3f}us` | "
            f"{ratio(item['sampled_main_speedup_ceiling'])} | "
            f"`{item['overlap_window_us']:.3f}us` | "
            f"{pct(item['required_overlap_efficiency_for_target'])} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- recommended first prototype: `{gate['recommended_first_prototype']}`",
            f"- prototype macro: `{gate['prototype_macro']}`",
            f"- best sampled-main ceiling: `{ratio(gate['best_sampled_main_speedup_ceiling'])}`",
            f"- reason: {gate['reason']}",
            "",
            "Prototype boundaries:",
            "",
        ]
    )
    for item in gate["prototype_boundaries"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json")
    parser.add_argument("--target-speedup", type=float, default=1.05)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    summary = Path(args.summary)
    if not summary.is_absolute():
        summary = root / summary
    timings = load_len16_timings(summary)
    result = compute_model(timings, args.target_speedup)

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
