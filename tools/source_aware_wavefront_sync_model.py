#!/usr/bin/env python3
"""Gate source-aware wavefront / synchronization temporal candidates.

This is the last exact CUDA-core temporal route after the current best.  Earlier
models showed a meaningful K=2 upper bound but no implementable ownership.  This
tool restates the gate against the formal current-best timing and checks whether
ordinary CUDA wavefront schedules solve synchronization and halo ownership.
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


def load_temporal(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cta = data["cta_local_candidates"]
    ratios = [float(item["local_pair_byte_ratio_vs_baseline"]) for item in cta]
    return {
        "geometry": data["geometry"],
        "core_byte_model": data["core_byte_model"],
        "cooperative_grid_gate": data["cooperative_grid_gate"],
        "cta_local_candidates": cta,
        "cta_local_ratio_min": min(ratios),
        "cta_local_ratio_max": max(ratios),
        "safe_global_middle": data["safe_global_middle"],
    }


def load_source_aware(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "aggregate_k2_deep_share": data["aggregate_k2_deep_share"],
        "source_overlap_shots": data["source_overlap_shots"],
        "receiver_overlap_shots": data["receiver_overlap_shots"],
        "shots": data["shots"],
        "receivers_per_shot": data["receivers_per_shot"],
    }


def load_len16_ncu(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = {item["label"]: item for item in data["profiles"]}
    kernels = profiles["len16"]["kernels"]

    def duration(name: str) -> float:
        item = kernels.get(name)
        return float(item["metrics"]["duration_ns"]) / 1000.0 if item else 0.0

    p_core = duration("cuda_fd3d_p_core_ns")
    p_residual = duration("cuda_fd3d_p_pml_tile_ns")
    p_len16 = duration("cuda_fd3d_p_pml_len16_halfwarp_ns")
    v_pml = duration("cuda_fd3d_v_pml_tile_ns")
    total = p_core + p_residual + p_len16 + v_pml
    return {
        "sampled_main_us": total,
        "p_core_us": p_core,
        "p_core_share": p_core / total,
        "pressure_pml_us": p_residual + p_len16,
        "v_pml_us": v_pml,
    }


def load_formal(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    best = data["summary_by_candidate"]["len16_current_best"]
    return {
        "mean_wp_speedup_vs_zmem": best["mean_wp_speedup_vs_zmem"],
        "mean_candidate_wp": best["mean_candidate_wp"],
    }


def sampled_speedup(p_core_share: float, p_core_reduction: float) -> float:
    return 1.0 / (1.0 - p_core_share * p_core_reduction)


def compute_model(
    temporal: dict[str, Any],
    source: dict[str, Any],
    ncu: dict[str, Any],
    formal: dict[str, Any],
    target_speedup: float,
) -> dict[str, Any]:
    p_core_reduction_ideal = temporal["core_byte_model"]["ideal_k2_p_core_pair_reduction"]
    ideal_current_best = sampled_speedup(ncu["p_core_share"], p_core_reduction_ideal)
    required_reduction = (1.0 - 1.0 / target_speedup) / ncu["p_core_share"]
    required_fraction_of_ideal = required_reduction / p_core_reduction_ideal
    coop = temporal["cooperative_grid_gate"]
    cta_min = temporal["cta_local_ratio_min"]
    cta_max = temporal["cta_local_ratio_max"]

    candidates: dict[str, dict[str, Any]] = {
        "safe_global_middle_two_kernel": {
            "ordinary_cuda": True,
            "correctness": "safe after a normal kernel boundary",
            "sampled_main_speedup_ceiling": 1.0,
            "reason": "Writes p(t+1) globally and reloads its stencil for step 2, so it keeps the global p_mid traffic that temporal blocking was meant to remove.",
            "decision": "reject_cuda_prototype",
        },
        "cooperative_grid_full_core_k2": {
            "ordinary_cuda": False,
            "correctness": "would provide an in-kernel grid barrier if all blocks were resident",
            "sampled_main_speedup_ceiling": ideal_current_best,
            "grid_blocks": coop["grid_blocks"],
            "resident_block_capacity": coop["resident_block_capacity"],
            "over_capacity_factor": coop["over_capacity_factor"],
            "reason": "The full p_core grid exceeds conservative resident capacity by about 52x, so a cooperative launch cannot cover the current grid.",
            "decision": "reject_cuda_prototype",
        },
        "cta_local_diamond_k2": {
            "ordinary_cuda": True,
            "correctness": "safe only by duplicating p_mid halo ownership inside each CTA",
            "sampled_main_speedup_ceiling": 1.0 / cta_min,
            "local_pair_byte_ratio_min": cta_min,
            "local_pair_byte_ratio_max": cta_max,
            "reason": "Concrete CTA-local candidates require 11.29x to 21.30x baseline pair bytes after p_mid halo duplication.",
            "decision": "reject_cuda_prototype",
        },
        "multi_kernel_global_wavefront": {
            "ordinary_cuda": True,
            "correctness": "can avoid half-updated reads by launching dependency layers as separate kernels",
            "sampled_main_speedup_ceiling": 1.0,
            "reason": "Layered global wavefronts still need global p_mid materialization between layers and add many small wavefront launches; no p_mid global stencil traffic is removed.",
            "decision": "reject_cuda_prototype",
        },
        "persistent_wavefront_without_global_barrier": {
            "ordinary_cuda": False,
            "correctness": "unsafe for neighboring CTA dependencies without a grid-wide barrier or no-duplication ownership",
            "sampled_main_speedup_ceiling": ideal_current_best,
            "reason": "A persistent kernel cannot let one CTA safely read another CTA's p_mid from shared/register state in ordinary CUDA.",
            "decision": "reject_cuda_prototype",
        },
        "ideal_no_dup_source_aware_wavefront": {
            "ordinary_cuda": False,
            "correctness": "source and receiver placement are compatible, but ownership primitive is missing",
            "sampled_main_speedup_ceiling": ideal_current_best,
            "reason": "This is the only meaningful ceiling, but it requires non-duplicating p_mid ownership across CTA boundaries without global reloads.",
            "decision": "reject_not_ordinary_cuda",
        },
    }
    return {
        "inputs": {
            "temporal": temporal,
            "source_aware": source,
            "ncu_current_best": ncu,
            "formal": formal,
            "target_speedup": target_speedup,
        },
        "current_best_rebased_temporal_ceiling": {
            "p_core_share": ncu["p_core_share"],
            "ideal_k2_p_core_pair_reduction": p_core_reduction_ideal,
            "ideal_sampled_main_speedup_current_best": ideal_current_best,
            "required_p_core_reduction_for_5pct": required_reduction,
            "required_fraction_of_ideal_saving": required_fraction_of_ideal,
        },
        "source_receiver_gate": {
            "aggregate_k2_deep_share": source["aggregate_k2_deep_share"],
            "source_overlap_shots": source["source_overlap_shots"],
            "receiver_overlap_shots": source["receiver_overlap_shots"],
            "verdict": "compatible_for_this_case",
        },
        "candidates": candidates,
        "gate": {
            "decision": "reject_source_aware_wavefront_cuda_prototype",
            "reason": (
                "Source and receiver placement are compatible, and the ideal current-best K=2 ceiling is meaningful. "
                "However, every ordinary CUDA schedule either materializes p_mid globally, duplicates p_mid halos by "
                "11x or more, or lacks the grid-wide/cross-CTA ownership primitive required to read p_mid safely."
            ),
            "next_allowed": [
                "application-level multi-shot batching or scheduling",
                "precision-relaxation study only with explicit tolerance policy",
                "future hardware/runtime-specific cross-CTA ownership only after a concrete primitive is identified",
            ],
            "prohibited": [
                "ordinary CUDA K=2 source-aware wavefront prototype",
                "multi-kernel global-middle wavefront prototype",
                "CTA-local diamond temporal prototype",
                "persistent-kernel wavefront relying on cross-CTA shared/register values",
            ],
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    ncu = result["inputs"]["ncu_current_best"]
    ceiling = result["current_best_rebased_temporal_ceiling"]
    source = result["source_receiver_gate"]
    coop = result["inputs"]["temporal"]["cooperative_grid_gate"]
    gate = result["gate"]
    lines = [
        "# Source-Aware Wavefront Synchronization Gate",
        "",
        "## Current-Best Rebase",
        "",
        f"- sampled main: `{ncu['sampled_main_us']:.3f}us`",
        f"- p_core: `{ncu['p_core_us']:.3f}us` / `{pct(ceiling['p_core_share'])}`",
        f"- formal current-best WP speedup vs zmem: `{ratio(result['inputs']['formal']['mean_wp_speedup_vs_zmem'])}`",
        f"- ideal K=2 p_core pair reduction: `{pct(ceiling['ideal_k2_p_core_pair_reduction'])}`",
        f"- ideal K=2 sampled-main speedup on current best: `{ratio(ceiling['ideal_sampled_main_speedup_current_best'])}`",
        f"- p_core reduction required for 1.05x sampled-main: `{pct(ceiling['required_p_core_reduction_for_5pct'])}`",
        f"- fraction of ideal saving required: `{pct(ceiling['required_fraction_of_ideal_saving'])}`",
        "",
        "## Source / Receiver Compatibility",
        "",
        f"- aggregate K=2 deep-core share: `{pct(source['aggregate_k2_deep_share'])}`",
        f"- source overlap shots: `{source['source_overlap_shots']}`",
        f"- receiver overlap shots: `{source['receiver_overlap_shots']}`",
        f"- verdict: `{source['verdict']}`",
        "",
        "## Synchronization Facts",
        "",
        f"- p_core grid blocks: `{coop['grid_blocks']}`",
        f"- conservative resident block capacity: `{coop['resident_block_capacity']}`",
        f"- cooperative-grid over-capacity factor: `{coop['over_capacity_factor']:.2f}x`",
        "",
        "## Candidate Schedules",
        "",
        "| candidate | ordinary CUDA | speedup ceiling | decision | reason |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for name, item in result["candidates"].items():
        lines.append(
            f"| `{name}` | `{item['ordinary_cuda']}` | `{ratio(item['sampled_main_speedup_ceiling'])}` | "
            f"`{item['decision']}` | {item['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
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
    parser.add_argument("--temporal", default="reports/day_20260608/temporal_pipeline_model.json")
    parser.add_argument("--source-aware", default="reports/day_20260608/source_aware_temporal_model.json")
    parser.add_argument("--len16-ncu", default="reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json")
    parser.add_argument("--formal-summary", default="reports/day_20260608/formal_current_best_table_20260608_182525/summary.json")
    parser.add_argument("--target-speedup", type=float, default=1.05)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = {}
    for key, value in {
        "temporal": args.temporal,
        "source": args.source_aware,
        "ncu": args.len16_ncu,
        "formal": args.formal_summary,
    }.items():
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        paths[key] = path

    result = compute_model(
        load_temporal(paths["temporal"]),
        load_source_aware(paths["source"]),
        load_len16_ncu(paths["ncu"]),
        load_formal(paths["formal"]),
        args.target_speedup,
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
