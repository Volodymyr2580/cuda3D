#!/usr/bin/env python3
"""Record the CUDA cluster/cooperative primitive gate for CUDA3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_summary() -> dict:
    cooperative_required_blocks = 70688
    cooperative_ceiling_blocks = 2040
    return {
        "device": {
            "name": "NVIDIA GeForce RTX 5090",
            "compute_capability": "12.0",
            "multi_processor_count": 170,
            "cuda": "13.0",
        },
        "probe": {
            "source": "tools/cuda_cluster_capability_probe.cu",
            "remote_worktree": "/work/wenzhe/cuda3D/.codex_worktrees/cluster_probe_20260609_0132",
            "stdout": "reports/day_20260609/cluster_probe_stdout_20260609_0132.txt",
            "block_size": 128,
            "active_blocks_per_sm": 12,
            "cooperative_grid_block_ceiling": cooperative_ceiling_blocks,
            "cooperative_required_blocks_for_previous_k2": cooperative_required_blocks,
            "cooperative_over_capacity_factor": cooperative_required_blocks
            / cooperative_ceiling_blocks,
            "cluster_launch": {
                "supported": True,
                "tested_cluster_sizes": {
                    "1": {"active_clusters": 340, "launch": "pass"},
                    "2": {"active_clusters": 170, "launch": "pass"},
                    "4": {"active_clusters": 85, "launch": "pass"},
                    "8": {"active_clusters": 41, "launch": "pass"},
                    "16": {
                        "active_clusters": -1,
                        "launch": "cluster misconfiguration",
                    },
                },
            },
        },
        "gate": {
            "decision": "reject_direct_cooperative_grid_k2_temporal_reopen",
            "cluster_decision": "design_only_until_cluster_local_ownership_model_passes",
            "ordinary_cuda_prototype_allowed": False,
            "cluster_cuda_prototype_allowed": False,
        },
        "interpretation": [
            "RTX 5090 / CUDA 13 supports cooperative launch and cluster launch.",
            "A full-grid cooperative K=2 temporal kernel remains infeasible because the prior K=2 geometry needs 70688 simultaneously resident blocks, while the measured 128-thread cooperative launch ceiling is 2040 blocks.",
            "Thread-block clusters can synchronize only blocks within the launched cluster; the probe validates up to 8-block clusters, not a grid-wide barrier.",
            "Cluster support may be useful only after a new cluster-local ownership model proves that all producer/consumer dependencies, source injection, receiver extraction, and PML/shell reconciliation stay correct across cluster boundaries.",
        ],
        "next_allowed_work": [
            "Write a cluster-local ownership and byte/synchronization model before any CUDA prototype.",
            "Reject attempts to revive the old full-core K=2 cooperative-grid prototype.",
            "Keep ordinary exact-CUDA micro-prototype frontier closed.",
        ],
    }


def write_markdown(summary: dict, path: Path) -> None:
    over = summary["probe"]["cooperative_over_capacity_factor"]
    lines = [
        "# Cluster / Cooperative Primitive Gate",
        "",
        "## Summary",
        "",
        "RTX 5090 / CUDA 13 exposes both cooperative launch and thread-block",
        "cluster launch, but this does **not** reopen the previous ordinary",
        "K=2 temporal CUDA prototype.",
        "",
        "Decision:",
        "",
        "```text",
        summary["gate"]["decision"],
        summary["gate"]["cluster_decision"],
        "ordinary CUDA prototype allowed = false",
        "cluster CUDA prototype allowed = false",
        "```",
        "",
        "## Probe Evidence",
        "",
        "Probe source:",
        "",
        "```text",
        summary["probe"]["source"],
        "```",
        "",
        "Remote worktree:",
        "",
        "```text",
        summary["probe"]["remote_worktree"],
        "```",
        "",
        "Raw stdout:",
        "",
        "```text",
        summary["probe"]["stdout"],
        "```",
        "",
        "Device:",
        "",
        "```text",
        f"name                         {summary['device']['name']}",
        f"compute capability           {summary['device']['compute_capability']}",
        f"SM count                     {summary['device']['multi_processor_count']}",
        f"CUDA                         {summary['device']['cuda']}",
        "cooperative_launch           1",
        "cluster_launch               1",
        "```",
        "",
        "Cooperative launch capacity:",
        "",
        "```text",
        f"block size                    {summary['probe']['block_size']}",
        f"active blocks / SM            {summary['probe']['active_blocks_per_sm']}",
        f"cooperative grid ceiling      {summary['probe']['cooperative_grid_block_ceiling']} blocks",
        f"previous K=2 required blocks  {summary['probe']['cooperative_required_blocks_for_previous_k2']} blocks",
        f"over-capacity factor          {over:.4f}x",
        "```",
        "",
        "Cluster launch probe:",
        "",
        "| cluster size | active clusters | launch |",
        "| ---: | ---: | --- |",
    ]
    for size, item in summary["probe"]["cluster_launch"]["tested_cluster_sizes"].items():
        lines.append(f"| {size} | {item['active_clusters']} | {item['launch']} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Full-grid cooperative temporal blocking remains infeasible: the grid",
            "  that needs a global barrier is still far larger than the resident",
            "  cooperative launch capacity.",
            "- Thread-block clusters are real on this platform and can synchronize",
            "  inside one cluster, but they are not a grid-wide barrier.",
            "- A cluster route can only reopen after a separate model proves",
            "  cluster-local ownership for `p_mid`/velocity/CPML state and handles",
            "  source injection, receiver extraction, shell/PML reconciliation, and",
            "  cross-cluster boundary dependencies.",
            "",
            "## Next Gate",
            "",
            "Allowed next work is design-only:",
            "",
            "```text",
            "cluster-local ownership byte/synchronization model",
            "```",
            "",
            "Do not write a cluster CUDA prototype until that model shows a",
            "`>=5%` repeat-speedup ceiling after boundary and synchronization",
            "costs are included.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    summary = build_summary()
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, md_path)
    print(summary["gate"]["decision"])
    print(f"cooperative_over_capacity_factor={summary['probe']['cooperative_over_capacity_factor']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
