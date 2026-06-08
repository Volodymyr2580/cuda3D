#!/usr/bin/env python3
"""Generate the current-best package summary for CUDA3D."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_text(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def ratio(value: float) -> str:
    return f"{value:.6f}x"


def current_best_from_formal(path: str) -> dict[str, Any]:
    data = load_json(path)
    alias = data["current_best_alias"]
    row = next(item for item in data["summary"] if item["candidate"] == alias)
    return {
        "alias": alias,
        "mean_elapsed": float(row["mean_elapsed"]),
        "mean_gradient": float(row["mean_gradient"]),
        "mean_wp": float(row["mean_wp"]),
        "elapsed_speedup_vs_zmem": float(row["mean_elapsed_speedup_vs_zmem"]),
        "gradient_speedup_vs_zmem": float(row["mean_gradient_speedup_vs_zmem"]),
        "wp_speedup_vs_zmem": float(row["mean_wp_speedup_vs_zmem"]),
        "max_rel_l2": float(row["max_rel_l2"]),
        "max_abs": float(row["max_abs"]),
        "all_compare_pass": bool(row["all_compare_pass"]),
        "flags": data["flags"][alias],
        "formal_summary": path,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    best = current_best_from_formal(args.formal_json)
    ownership = load_json(args.ownership_json)
    cluster_probe = load_json(args.cluster_probe_json)
    cluster_local = load_json(args.cluster_local_json)
    app = load_json(args.application_json)

    head = git_text(["rev-parse", "HEAD"])
    branch = git_text(["branch", "--show-current"])
    recent_commits = git_text(["log", "--oneline", "--decorate", "-8"]).splitlines()

    return {
        "package": {
            "name": "current_best_v_pml_len16_package",
            "status": "current_best_not_speed_threshold_archive",
            "branch": branch,
            "head_commit": head,
        },
        "current_best": best,
        "milestone": {
            "is_1_5x_archive": False,
            "additional_wp_speedup_needed_for_1_5x": 1.5 / best["wp_speedup_vs_zmem"],
            "reason": "Speed-threshold archives start at 1.5x; current formal WP speedup is below that threshold.",
        },
        "accepted_stack": [
            "CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL",
            "CUDA3D_CPML_VMEM_DISABLE_MPI",
            "CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE",
            "CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK",
            "CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK",
        ],
        "frontier_gates": {
            "ordinary_exact_cuda": {
                "decision": ownership["gate"]["decision"],
                "ordinary_cuda_allowed_count": ownership["gate"]["ordinary_cuda_allowed_count"],
                "report": args.ownership_json.replace("reports/", "docs/").replace(".json", ".md"),
            },
            "cluster_cooperative": {
                "decision": cluster_probe["gate"]["decision"],
                "cluster_decision": cluster_probe["gate"]["cluster_decision"],
                "cooperative_over_capacity_factor": cluster_probe["probe"]["cooperative_over_capacity_factor"],
                "report": args.cluster_probe_md,
            },
            "cluster_local_ownership": {
                "decision": cluster_local["gate"]["decision"],
                "best_local_pair_byte_ratio": cluster_local["best_dsm_tile"]["local_pair_byte_ratio_vs_baseline"],
                "best_sampled_main_speedup_estimate": cluster_local["gate"]["best_sampled_main_speedup_estimate"],
                "report": args.cluster_local_md,
            },
            "application_level": {
                "decision": app["gate"]["decision"],
                "available_gpus": app["platform"]["available_gpus"],
                "same_gpu_best_elapsed_speedup": app["same_gpu_multirank"]["best_elapsed_speedup"],
                "report": args.application_md,
            },
        },
        "allowed_next": [
            "Run true multi-GPU batching when a >=2 GPU platform is available.",
            "Open precision-relaxation only after an explicit tolerance-policy change.",
            "Propose a fundamentally different ownership representation and pass a new byte/synchronization model before CUDA code.",
            "Stop the CUDA-core sprint at current-best and package results.",
        ],
        "do_not_repeat": [
            "ordinary exact-CUDA micro prototypes under the closed route matrix",
            "direct cooperative-grid K=2 temporal reopen",
            "cluster-local K=2 temporal DSM prototype",
            "same-GPU multi-rank oversubscription",
            "host/setup micro-prototypes without a new >=5% measured hotspot",
        ],
        "recent_commits": recent_commits,
        "primary_reports": [
            args.formal_json,
            args.ownership_md,
            args.cluster_probe_md,
            args.cluster_local_md,
            args.application_md,
            "docs/day_20260609/pro_handoff_current_best_frontier.md",
        ],
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    best = summary["current_best"]
    milestone = summary["milestone"]
    gates = summary["frontier_gates"]
    lines = [
        "# CUDA3D Current-Best Package",
        "",
        "## Status",
        "",
        "This is a current-best package, not a speed-threshold archive.",
        "",
        "```text",
        f"branch                         {summary['package']['branch']}",
        f"head commit                    {summary['package']['head_commit']}",
        f"candidate                      {best['alias']}",
        f"mean elapsed                   {best['mean_elapsed']:.6f}s",
        f"mean Gradient                  {best['mean_gradient']:.6f}s",
        f"mean WP                        {best['mean_wp']:.6f}s",
        f"elapsed speedup vs zmem         {ratio(best['elapsed_speedup_vs_zmem'])}",
        f"Gradient speedup vs zmem        {ratio(best['gradient_speedup_vs_zmem'])}",
        f"WP speedup vs zmem              {ratio(best['wp_speedup_vs_zmem'])}",
        f"max rel L2                      {best['max_rel_l2']:.6e}",
        f"max abs                         {best['max_abs']:.6e}",
        f"all compare pass                {best['all_compare_pass']}",
        "```",
        "",
        "Milestone:",
        "",
        "```text",
        f"1.5x archive                    {milestone['is_1_5x_archive']}",
        f"additional WP speedup to 1.5x   {ratio(milestone['additional_wp_speedup_needed_for_1_5x'])}",
        "```",
        "",
        "## Current-Best Flags",
        "",
        "```text",
        best["flags"],
        "```",
        "",
        "## Accepted Stack",
        "",
    ]
    for item in summary["accepted_stack"]:
        lines.append(f"- `{item}`")

    lines.extend(
        [
            "",
            "## Closed Frontiers",
            "",
            "| frontier | decision | key number | report |",
            "| --- | --- | ---: | --- |",
            f"| ordinary exact CUDA | `{gates['ordinary_exact_cuda']['decision']}` | `{gates['ordinary_exact_cuda']['ordinary_cuda_allowed_count']}` allowed | `{gates['ordinary_exact_cuda']['report']}` |",
            f"| cluster/cooperative | `{gates['cluster_cooperative']['decision']}` | `{ratio(gates['cluster_cooperative']['cooperative_over_capacity_factor'])}` over capacity | `{gates['cluster_cooperative']['report']}` |",
            f"| cluster-local ownership | `{gates['cluster_local_ownership']['decision']}` | `{gates['cluster_local_ownership']['best_local_pair_byte_ratio']:.4f}x` byte ratio | `{gates['cluster_local_ownership']['report']}` |",
            f"| application-level local | `{gates['application_level']['decision']}` | `{gates['application_level']['available_gpus']}` GPU | `{gates['application_level']['report']}` |",
            "",
            "## Allowed Next",
            "",
        ]
    )
    for item in summary["allowed_next"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Do Not Repeat")
    lines.append("")
    for item in summary["do_not_repeat"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Primary Reports")
    lines.append("")
    for item in summary["primary_reports"]:
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("## Recent Commits")
    lines.append("")
    lines.append("```text")
    lines.extend(summary["recent_commits"])
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-json", default="reports/day_20260608/formal_vpmlen16_table_20260608_2359/summary.json")
    parser.add_argument("--ownership-json", default="reports/day_20260608/ownership_frontier_gate.json")
    parser.add_argument("--ownership-md", default="docs/day_20260608/ownership_frontier_gate.md")
    parser.add_argument("--cluster-probe-json", default="reports/day_20260609/cluster_cooperative_frontier_gate.json")
    parser.add_argument("--cluster-probe-md", default="docs/day_20260609/cluster_cooperative_frontier_gate.md")
    parser.add_argument("--cluster-local-json", default="reports/day_20260609/cluster_local_ownership_model.json")
    parser.add_argument("--cluster-local-md", default="docs/day_20260609/cluster_local_ownership_model.md")
    parser.add_argument("--application-json", default="reports/day_20260609/application_level_frontier_gate.json")
    parser.add_argument("--application-md", default="docs/day_20260609/application_level_frontier_gate.md")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    summary = build_summary(args)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, Path(args.md_out))
    print(summary["package"]["status"])
    print(f"wp_speedup={summary['current_best']['wp_speedup_vs_zmem']:.6f}")
    print(f"head_commit={summary['package']['head_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
