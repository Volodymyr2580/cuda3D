#!/usr/bin/env python3
"""Current application-level scheduling frontier for CUDA3D.

This consolidates the non-kernel routes after the exact CUDA-core frontier was
closed: same-GPU multi-rank, true multi-GPU batching, and host/setup overhead.
It does not run benchmarks; it records whether there is a local experiment still
worth doing on the current single-RTX-5090 platform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_ratio(value: float) -> str:
    return f"{value:.4f}x"


def current_best(formal: dict[str, Any]) -> dict[str, Any]:
    alias = formal["current_best_alias"]
    rows = {row["candidate"]: row for row in formal["summary"]}
    row = rows[alias]
    records = [item for item in formal["records"] if item["label"] == alias]
    return {
        "alias": alias,
        "mean_elapsed": float(row["mean_elapsed"]),
        "mean_gradient": float(row["mean_gradient"]),
        "mean_wp": float(row["mean_wp"]),
        "mean_elapsed_speedup_vs_zmem": float(row["mean_elapsed_speedup_vs_zmem"]),
        "mean_gradient_speedup_vs_zmem": float(row["mean_gradient_speedup_vs_zmem"]),
        "mean_wp_speedup_vs_zmem": float(row["mean_wp_speedup_vs_zmem"]),
        "max_rel_l2": float(row["max_rel_l2"]),
        "rounds": len(records),
    }


def shot_distribution(shots: int, ranks: int) -> dict[str, Any]:
    ns_pad = shots if shots % ranks == 0 else (shots // ranks + 1) * ranks
    myns = ns_pad // ranks
    per_rank = []
    for rank in range(ranks):
        assigned = []
        for ishot in range(myns):
            snum = ishot * ranks + rank
            if snum < shots:
                assigned.append(snum)
        per_rank.append(assigned)
    active_counts = [len(items) for items in per_rank]
    max_count = max(active_counts)
    ideal_speedup = shots / max_count if max_count else 0.0
    return {
        "ranks": ranks,
        "active_counts": active_counts,
        "ideal_shot_balance_speedup": ideal_speedup,
        "ideal_parallel_efficiency": ideal_speedup / ranks if ranks else 0.0,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    formal = load_json(args.formal_json)
    same_gpu = load_json(args.same_gpu_json)
    process = load_json(args.process_timer_json)
    cal_loop = load_json(args.cal_loop_json)
    best = current_best(formal)

    distributions = []
    for ranks in (1, 2, 3, 4, 6):
        item = shot_distribution(args.shots, ranks)
        item["runnable_on_current_platform"] = ranks <= args.available_gpus
        item["ideal_elapsed"] = best["mean_elapsed"] / item["ideal_shot_balance_speedup"]
        item["ideal_gradient"] = best["mean_gradient"] / item["ideal_shot_balance_speedup"]
        item["ideal_wp"] = best["mean_wp"] / item["ideal_shot_balance_speedup"]
        distributions.append(item)

    wavefield_prep = float(cal_loop["timers"]["cal_loop"]["wavefield_prep"])
    gradient = float(cal_loop["gradient_s"])
    wavefield_prep_speedup = 1.0 / (1.0 - wavefield_prep / gradient)

    return {
        "inputs": {
            "formal_json": args.formal_json,
            "same_gpu_json": args.same_gpu_json,
            "process_timer_json": args.process_timer_json,
            "cal_loop_json": args.cal_loop_json,
            "available_gpus": args.available_gpus,
            "shots": args.shots,
        },
        "platform": {
            "available_gpus": args.available_gpus,
            "latest_remote_probe": "nvidia-smi -L shows one NVIDIA GeForce RTX 5090",
        },
        "formal_current_best": best,
        "same_gpu_multirank": {
            "decision": same_gpu["decision"],
            "best_elapsed_speedup": float(same_gpu["best_elapsed_speedup"]),
            "best_gradient_speedup": float(same_gpu["best_gradient_speedup"]),
            "allowed": False,
        },
        "true_multigpu_batching": {
            "decision": "defer_current_platform_single_gpu",
            "runnable_now": args.available_gpus >= 2,
            "distributions": distributions,
            "gate": "requires >=2 visible GPUs, input gpus_p_node=N, mpirun -np N, CUDA_VISIBLE_DEVICES with N devices, 3-round repeat, and output comparisons pass",
        },
        "host_setup": {
            "decision": "reject_more_local_host_setup_micro_prototypes",
            "process_timer_source": args.process_timer_json,
            "cal_loop_timer_source": args.cal_loop_json,
            "process_timer_elapsed": float(process["elapsed_s"]),
            "process_timer_gradient": float(process["gradient_s"]),
            "outside_process_s": float(process["outside_process_s"]),
            "mpi_init_s": float(process["timers"]["process"]["mpi_init"]),
            "gpu_setup_s": float(process["timers"]["main"]["gpu_setup"]),
            "cal_loop_wavefield_prep_s": wavefield_prep,
            "cal_loop_wavefield_prep_gradient_speedup_ceiling": wavefield_prep_speedup,
            "note": "Timer probes predate the final v-PML len16 formal table, but their route decisions are insensitive to that small compute-kernel improvement.",
        },
        "gate": {
            "decision": "no_local_application_level_experiment_available_on_single_gpu",
            "reason": "same-GPU oversubscription is slower, true multi-GPU batching needs more visible GPUs, and host/setup/pre-FD local micro routes do not meet the >=5% gate.",
            "allowed_next": [
                "Run true multi-GPU batching when a >=2 GPU platform is available.",
                "Open precision-relaxation only after an explicit tolerance-policy change.",
                "Stop CUDA-core sprint at current-best and package results.",
            ],
            "prohibited": [
                "same-GPU np=2/3 oversubscription reruns for perf_1gpu_6shots",
                "claiming speedup from root-rank printed WP in multi-rank runs",
                "host/setup micro-prototypes without a new >=5% measured hotspot",
                "true multi-GPU benchmark on current single-GPU platform",
            ],
        },
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    best = summary["formal_current_best"]
    same = summary["same_gpu_multirank"]
    host = summary["host_setup"]
    gate = summary["gate"]
    lines = [
        "# Application-Level Scheduling Frontier Gate",
        "",
        "## Summary",
        "",
        "Current single-GPU RTX 5090 platform has no remaining local",
        "application-level scheduling experiment worth running.",
        "",
        "Decision:",
        "",
        "```text",
        gate["decision"],
        "```",
        "",
        "## Formal Current-Best Anchor",
        "",
        "```text",
        f"alias                         {best['alias']}",
        f"rounds                        {best['rounds']}",
        f"mean elapsed                  {best['mean_elapsed']:.6f}s",
        f"mean Gradient                 {best['mean_gradient']:.6f}s",
        f"mean WP                       {best['mean_wp']:.6f}s",
        f"elapsed speedup vs zmem        {fmt_ratio(best['mean_elapsed_speedup_vs_zmem'])}",
        f"Gradient speedup vs zmem       {fmt_ratio(best['mean_gradient_speedup_vs_zmem'])}",
        f"WP speedup vs zmem             {fmt_ratio(best['mean_wp_speedup_vs_zmem'])}",
        f"max rel L2                     {best['max_rel_l2']:.6e}",
        "```",
        "",
        "## Same-GPU Multi-Rank",
        "",
        "```text",
        f"decision                      {same['decision']}",
        f"best elapsed speedup          {fmt_ratio(same['best_elapsed_speedup'])}",
        f"best Gradient speedup         {fmt_ratio(same['best_gradient_speedup'])}",
        "```",
        "",
        "Same-GPU MPI oversubscription is slower and remains prohibited.",
        "",
        "## True Multi-GPU Batching",
        "",
        f"Available GPUs now: `{summary['platform']['available_gpus']}`.",
        "",
        "| GPUs/ranks | runnable now | active shots/rank | ideal speedup | ideal elapsed | ideal Gradient |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in summary["true_multigpu_batching"]["distributions"]:
        lines.append(
            f"| {item['ranks']} | `{item['runnable_on_current_platform']}` | "
            f"`{item['active_counts']}` | `{fmt_ratio(item['ideal_shot_balance_speedup'])}` | "
            f"`{item['ideal_elapsed']:.6f}s` | `{item['ideal_gradient']:.6f}s` |"
        )
    lines.extend(
        [
            "",
            "True multi-GPU batching is the only application-level route with large",
            "theoretical upside, but it cannot be validated on the current one-GPU",
            "server.",
            "",
            "## Host / Setup",
            "",
            "```text",
            f"process-timer elapsed          {host['process_timer_elapsed']:.6f}s",
            f"process-timer Gradient         {host['process_timer_gradient']:.6f}s",
            f"outside process wrapper        {host['outside_process_s']:.6f}s",
            f"MPI_Init                       {host['mpi_init_s']:.6f}s",
            f"gpu_setup/context              {host['gpu_setup_s']:.6f}s",
            f"cal-loop wavefield_prep        {host['cal_loop_wavefield_prep_s']:.6f}s",
            f"wavefield_prep ceiling         {fmt_ratio(host['cal_loop_wavefield_prep_gradient_speedup_ceiling'])}",
            "```",
            "",
            "Host/setup and pre-FD loop micro routes remain below the `>=5%` gate",
            "unless a new measured hotspot appears.",
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- reason: {gate['reason']}",
            "",
            "Allowed next:",
        ]
    )
    for item in gate["allowed_next"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Do not continue:")
    for item in gate["prohibited"]:
        lines.append(f"- {item}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-json", default="reports/day_20260608/formal_vpmlen16_table_20260608_2359/summary.json")
    parser.add_argument("--same-gpu-json", default="reports/day_20260608/multirank_samegpu_sched_20260608_193042/summary.json")
    parser.add_argument("--process-timer-json", default="reports/day_20260608/process_timer_probe_20260608_205311/summary.json")
    parser.add_argument("--cal-loop-json", default="reports/day_20260608/cal_loop_timer_probe_20260608_212019/summary.json")
    parser.add_argument("--available-gpus", type=int, default=1)
    parser.add_argument("--shots", type=int, default=6)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    summary = build_summary(args)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, Path(args.md_out))
    print(summary["gate"]["decision"])
    print(f"current_best_wp_speedup={summary['formal_current_best']['mean_wp_speedup_vs_zmem']:.6f}")
    print(f"same_gpu_best_elapsed_speedup={summary['same_gpu_multirank']['best_elapsed_speedup']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
