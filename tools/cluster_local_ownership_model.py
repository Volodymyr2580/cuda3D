#!/usr/bin/env python3
"""Cluster-local temporal ownership budget for CUDA3D.

This model asks whether CUDA thread-block clusters can reopen the K=2 temporal
route after the cooperative/cluster primitive probe.  It is intentionally
optimistic: it lets a cluster spend all per-block shared memory as one DSM tile
and ignores distributed-shared-memory latency.  If this upper-bound tile search
does not beat the current two-pass p_core byte model, a CUDA prototype is not
worth writing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FLOAT_BYTES = 4


def speedup_after_region_reduction(region_share: float, reduction: float) -> float:
    return 1.0 / (1.0 - region_share * reduction)


def ratio(value: float) -> str:
    return f"{value:.4f}x"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def enumerate_dsm_tiles(
    cluster_size: int,
    max_smem_per_block: int,
    radius: int,
    baseline_pair_bytes_per_output: float,
    p_mid_compute_bytes_per_output: float,
    final_step_bytes_per_output: float,
) -> list[dict[str, Any]]:
    max_dsm_bytes = cluster_size * max_smem_per_block
    candidates: list[dict[str, Any]] = []

    # Optimistic regular-grid search.  The z/x/y limits are intentionally wider
    # than the current p_core CTA shape and constrained only by DSM capacity.
    for z in range(8, 129, 8):
        for x in range(4, 129, 4):
            for y in range(4, 129, 4):
                final_outputs = z * x * y
                p_mid_elements = (z + 2 * radius) * (x + 2 * radius) * (y + 2 * radius)
                p_mid_bytes = p_mid_elements * FLOAT_BYTES
                if p_mid_bytes > max_dsm_bytes:
                    continue
                p_mid_per_output = p_mid_elements / final_outputs
                local_pair_bytes = (
                    p_mid_per_output * p_mid_compute_bytes_per_output
                    + final_step_bytes_per_output
                )
                byte_ratio = local_pair_bytes / baseline_pair_bytes_per_output
                candidates.append(
                    {
                        "cluster_size": cluster_size,
                        "output_z": z,
                        "output_x": x,
                        "output_y": y,
                        "final_outputs": final_outputs,
                        "p_mid_z": z + 2 * radius,
                        "p_mid_x": x + 2 * radius,
                        "p_mid_y": y + 2 * radius,
                        "p_mid_elements": p_mid_elements,
                        "p_mid_bytes": p_mid_bytes,
                        "max_dsm_bytes": max_dsm_bytes,
                        "p_mid_elements_per_output": p_mid_per_output,
                        "local_pair_bytes_per_output": local_pair_bytes,
                        "baseline_pair_bytes_per_output": baseline_pair_bytes_per_output,
                        "local_pair_byte_ratio_vs_baseline": byte_ratio,
                        "p_core_pair_reduction": 1.0 - byte_ratio,
                    }
                )
    candidates.sort(key=lambda item: item["local_pair_byte_ratio_vs_baseline"])
    return candidates


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    temporal = load_json(args.temporal_json)
    cluster = load_json(args.cluster_json)
    post = load_json(args.post_vlen16_json)

    core_model = temporal["core_byte_model"]
    profile = post["inputs"]["profile"]
    sampled_total_us = float(profile["sampled_main_us"])
    p_core_us = float(profile["p_core_us"])
    p_core_share = p_core_us / sampled_total_us

    p1_global_floats = float(core_model["p1_global_floats_per_output"])
    current_bytes_per_output = float(core_model["bytes_per_output_current"])
    baseline_pair_bytes = 2.0 * current_bytes_per_output
    p_mid_compute_bytes = (p1_global_floats + 2.0) * FLOAT_BYTES
    final_step_bytes = float(core_model["p0_cw2_store_floats_per_output"]) * FLOAT_BYTES

    search: dict[str, Any] = {}
    best_all: dict[str, Any] | None = None
    for cluster_size in (1, 2, 4, 8):
        candidates = enumerate_dsm_tiles(
            cluster_size=cluster_size,
            max_smem_per_block=args.max_smem_per_block,
            radius=args.radius,
            baseline_pair_bytes_per_output=baseline_pair_bytes,
            p_mid_compute_bytes_per_output=p_mid_compute_bytes,
            final_step_bytes_per_output=final_step_bytes,
        )
        best = candidates[0]
        best["sampled_main_speedup_estimate"] = speedup_after_region_reduction(
            p_core_share, best["p_core_pair_reduction"]
        )
        search[str(cluster_size)] = {
            "best": best,
            "top5": candidates[:5],
        }
        if best_all is None or best["local_pair_byte_ratio_vs_baseline"] < best_all["local_pair_byte_ratio_vs_baseline"]:
            best_all = best

    assert best_all is not None
    best_reduction = float(best_all["p_core_pair_reduction"])
    best_sampled_speedup = speedup_after_region_reduction(p_core_share, best_reduction)

    required_p_core_reduction = (1.0 - 1.0 / args.meaningful_speedup) / p_core_share
    required_pair_ratio = 1.0 - required_p_core_reduction

    ideal_reduction = float(core_model["ideal_k2_p_core_pair_reduction"])
    ideal_sampled_speedup = speedup_after_region_reduction(p_core_share, ideal_reduction)

    return {
        "inputs": {
            "temporal_json": args.temporal_json,
            "cluster_json": args.cluster_json,
            "post_vlen16_json": args.post_vlen16_json,
            "radius": args.radius,
            "max_smem_per_block": args.max_smem_per_block,
            "meaningful_speedup": args.meaningful_speedup,
        },
        "profile": {
            "sampled_main_us": sampled_total_us,
            "p_core_us": p_core_us,
            "p_core_share": p_core_share,
            "formal_current_best_wp_speedup": 1.222023,
        },
        "byte_model": {
            "current_bytes_per_output": current_bytes_per_output,
            "baseline_pair_bytes_per_output": baseline_pair_bytes,
            "p1_global_floats_per_output": p1_global_floats,
            "p_mid_compute_bytes_per_output": p_mid_compute_bytes,
            "final_step_bytes_per_output": final_step_bytes,
            "required_p_core_pair_reduction_for_5pct": required_p_core_reduction,
            "required_local_pair_byte_ratio_for_5pct": required_pair_ratio,
            "ideal_no_dup_p_core_pair_reduction": ideal_reduction,
            "ideal_no_dup_sampled_main_speedup": ideal_sampled_speedup,
        },
        "cooperative_probe": {
            "cooperative_grid_block_ceiling": cluster["probe"]["cooperative_grid_block_ceiling"],
            "previous_k2_required_blocks": cluster["probe"]["cooperative_required_blocks_for_previous_k2"],
            "cooperative_over_capacity_factor": cluster["probe"]["cooperative_over_capacity_factor"],
            "max_passing_cluster_size": 8,
        },
        "dsm_tile_search": search,
        "best_dsm_tile": best_all,
        "gate": {
            "decision": "reject_cluster_local_temporal_cuda_prototype",
            "ordinary_cuda_prototype_allowed": False,
            "cluster_cuda_prototype_allowed": False,
            "reason": (
                "Even an optimistic 8-block cluster DSM tile search is slower than the current two-pass p_core byte model. "
                "It gives local pair byte ratio > 1.0 instead of the <= threshold needed for a 5% sampled-main win."
            ),
            "prohibited": [
                "direct cooperative-grid K=2 temporal prototype",
                "cluster-local K=2 temporal CUDA prototype with DSM p_mid tile",
                "cluster producer-consumer fusion without a new ownership model that beats the DSM byte gate",
            ],
            "next_allowed": [
                "precision-relaxation study only after explicit tolerance policy change",
                "application-level multi-shot scheduling / batching",
                "a fundamentally different ownership representation that proves p_mid/state traffic removal without DSM halo blow-up",
            ],
            "best_sampled_main_speedup_estimate": best_sampled_speedup,
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    profile = summary["profile"]
    byte_model = summary["byte_model"]
    coop = summary["cooperative_probe"]
    best = summary["best_dsm_tile"]
    gate = summary["gate"]

    lines = [
        "# Cluster-Local Ownership Model",
        "",
        "## Summary",
        "",
        "Thread-block clusters are available on the RTX 5090, but the",
        "cluster-local K=2 temporal route does not pass the byte model gate.",
        "",
        "Decision:",
        "",
        "```text",
        gate["decision"],
        "ordinary CUDA prototype allowed = false",
        "cluster CUDA prototype allowed = false",
        "```",
        "",
        "## Current-Best Anchor",
        "",
        "```text",
        f"formal current-best WP speedup   {ratio(profile['formal_current_best_wp_speedup'])}",
        f"sampled main                     {profile['sampled_main_us']:.3f}us",
        f"p_core                           {profile['p_core_us']:.3f}us",
        f"p_core share                     {pct(profile['p_core_share'])}",
        "```",
        "",
        "## Cooperative / Cluster Capacity",
        "",
        "```text",
        f"cooperative grid ceiling         {coop['cooperative_grid_block_ceiling']} blocks",
        f"previous K=2 required blocks     {coop['previous_k2_required_blocks']} blocks",
        f"cooperative over-capacity        {ratio(coop['cooperative_over_capacity_factor'])}",
        f"max passing cluster size         {coop['max_passing_cluster_size']}",
        "```",
        "",
        "## Byte Gate",
        "",
        "```text",
        f"baseline pair bytes/output       {byte_model['baseline_pair_bytes_per_output']:.3f}",
        f"p_mid compute bytes/output       {byte_model['p_mid_compute_bytes_per_output']:.3f}",
        f"final step bytes/output          {byte_model['final_step_bytes_per_output']:.3f}",
        f"required p_core reduction        {pct(byte_model['required_p_core_pair_reduction_for_5pct'])}",
        f"required local pair byte ratio   <= {byte_model['required_local_pair_byte_ratio_for_5pct']:.4f}",
        f"ideal no-dup sampled speedup     {ratio(byte_model['ideal_no_dup_sampled_main_speedup'])}",
        "```",
        "",
        "The DSM tile search is optimistic: it lets a cluster spend all per-block",
        "shared memory as one distributed tile and ignores DSM latency.  A real",
        "implementation would be harder, so this is an upper-bound filter.",
        "",
        "Best DSM tile found:",
        "",
        "```text",
        f"cluster size                     {best['cluster_size']}",
        f"output z/x/y                     {best['output_z']} / {best['output_x']} / {best['output_y']}",
        f"p_mid z/x/y                      {best['p_mid_z']} / {best['p_mid_x']} / {best['p_mid_y']}",
        f"p_mid bytes                      {best['p_mid_bytes']}",
        f"p_mid elements/output            {best['p_mid_elements_per_output']:.4f}",
        f"local pair byte ratio            {best['local_pair_byte_ratio_vs_baseline']:.4f}",
        f"estimated sampled-main speedup   {ratio(gate['best_sampled_main_speedup_estimate'])}",
        "```",
        "",
        "Best tile by cluster size:",
        "",
        "| cluster size | output z/x/y | p_mid bytes | local pair byte ratio | sampled-main estimate |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]

    for size in ("1", "2", "4", "8"):
        item = summary["dsm_tile_search"][size]["best"]
        lines.append(
            f"| {size} | {item['output_z']}x{item['output_x']}x{item['output_y']} | "
            f"{item['p_mid_bytes']} | {item['local_pair_byte_ratio_vs_baseline']:.4f}x | "
            f"{ratio(item['sampled_main_speedup_estimate'])} |"
        )

    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- reason: {gate['reason']}",
            "",
            "Do not continue:",
        ]
    )
    for item in gate["prohibited"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Allowed next directions:")
    for item in gate["next_allowed"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-json", default="reports/day_20260608/temporal_pipeline_model.json")
    parser.add_argument("--cluster-json", default="reports/day_20260609/cluster_cooperative_frontier_gate.json")
    parser.add_argument("--post-vlen16-json", default="reports/day_20260608/post_vlen16_pressure_next_gate.json")
    parser.add_argument("--radius", type=int, default=7)
    parser.add_argument("--max-smem-per-block", type=int, default=99_000)
    parser.add_argument("--meaningful-speedup", type=float, default=1.05)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    summary = build_summary(args)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    Path(args.md_out).write_text(render_markdown(summary), encoding="utf-8")
    print(summary["gate"]["decision"])
    print(f"best_local_pair_byte_ratio={summary['best_dsm_tile']['local_pair_byte_ratio_vs_baseline']:.4f}")
    print(f"best_sampled_main_speedup={summary['gate']['best_sampled_main_speedup_estimate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
