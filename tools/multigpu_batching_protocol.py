#!/usr/bin/env python3
"""Build the true multi-GPU / multi-job batching protocol.

The current RTX 5090 server has one GPU, so same-GPU oversubscription can be
tested but true multi-GPU scheduling cannot.  This tool records the gate and the
exact protocol needed when a multi-GPU platform is available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def fmt_ratio(value: float) -> str:
    return f"{value:.4f}x"


def parse_case_manifest(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def parse_input_last_int(path: Path) -> int:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return int(lines[-1])


def load_formal_current_best(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = [r for r in data["records"] if r["label"] == data["current_best_alias"]]
    mean_elapsed = sum(float(r["elapsed_seconds"]) for r in records) / len(records)
    mean_gradient = sum(float(r["gradient"]) for r in records) / len(records)
    mean_wp = sum(float(r["wp"]) for r in records) / len(records)
    summary = data["summary_by_candidate"][data["current_best_alias"]]
    return {
        "alias": data["current_best_alias"],
        "mean_elapsed_s": mean_elapsed,
        "mean_gradient_s": mean_gradient,
        "mean_wp_s": mean_wp,
        "wp_speedup_vs_zmem": float(summary["mean_wp_speedup_vs_zmem"]),
        "gradient_speedup_vs_zmem": float(summary["mean_gradient_speedup_vs_zmem"]),
        "max_rel_l2_vs_zmem": float(summary["max_rel_l2"]),
    }


def load_same_gpu_probe(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "decision": data["decision"],
        "best_elapsed_speedup": float(data["best_elapsed_speedup"]),
        "best_gradient_speedup": float(data["best_gradient_speedup"]),
    }


def shot_distribution(shots: int, ranks: int) -> dict[str, Any]:
    ns_pad = shots if shots % ranks == 0 else (shots // ranks + 1) * ranks
    myns = ns_pad // ranks
    per_rank: list[list[int]] = []
    for rank in range(ranks):
        assigned: list[int] = []
        for ishot in range(myns):
            snum = ishot * ranks + rank
            if snum < shots:
                assigned.append(snum)
        per_rank.append(assigned)
    active_counts = [len(items) for items in per_rank]
    max_count = max(active_counts)
    ideal_speedup = shots / max_count if max_count else 0.0
    efficiency = ideal_speedup / ranks if ranks else 0.0
    return {
        "ranks": ranks,
        "ns_pad": ns_pad,
        "myns": myns,
        "per_rank": per_rank,
        "active_counts": active_counts,
        "max_shots_per_rank": max_count,
        "ideal_shot_balance_speedup": ideal_speedup,
        "ideal_parallel_efficiency": efficiency,
    }


def build_model(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    case_dir = root / "benchmarks/cases/perf_1gpu_6shots"
    manifest = parse_case_manifest(case_dir / "case_manifest.txt")
    input_file = case_dir / "input_perf_1gpu_6shots.in"
    shots = int(manifest["shots"])
    current_input_gpus_p_node = parse_input_last_int(input_file)
    formal = load_formal_current_best(
        root / "reports/day_20260608/formal_current_best_table_20260608_182525/summary.json"
    )
    same_gpu = load_same_gpu_probe(
        root / "reports/day_20260608/multirank_samegpu_sched_20260608_193042/summary.json"
    )
    candidate_gpu_counts = sorted({1, 2, 3, 4, 6, shots})
    distributions = []
    for gpu_count in candidate_gpu_counts:
        dist = shot_distribution(shots, gpu_count)
        dist["runnable_on_current_platform"] = gpu_count <= args.available_gpus
        dist["requires_input_gpus_p_node"] = gpu_count
        dist["requires_mpirun_np"] = gpu_count
        dist["requires_cuda_visible_devices_count"] = gpu_count
        dist["estimated_elapsed_s_ideal"] = formal["mean_elapsed_s"] / dist["ideal_shot_balance_speedup"]
        dist["estimated_gradient_s_ideal"] = formal["mean_gradient_s"] / dist["ideal_shot_balance_speedup"]
        distributions.append(dist)
    decision = (
        "defer_true_multigpu_validation_current_platform_single_gpu"
        if args.available_gpus < 2
        else "ready_for_true_multigpu_repeat_benchmark"
    )
    return {
        "decision": decision,
        "available_gpus": args.available_gpus,
        "case": manifest,
        "current_input_gpus_p_node": current_input_gpus_p_node,
        "formal_current_best": formal,
        "same_gpu_probe": same_gpu,
        "source_evidence": {
            "gpu_mapping": "src/main.cu reads gpus_p_node from input, checks ntids % gpus_p_node, then cudaSetDevice(mytid % gpus_p_node).",
            "shot_distribution": "src/main.cu pads ns_s to ntids and assigns shot sht_num[is * ntids + mytid] to each rank.",
            "multi_rank_metric_warning": "Printed WP computing time is root-rank local under multi-rank scheduling; use elapsed and Gradient TIME all.",
        },
        "distributions": distributions,
        "protocol": {
            "one_rank_per_gpu": True,
            "input_last_line_must_equal_visible_gpu_count": True,
            "required_metrics": [
                "elapsed wall-clock from /usr/bin/time -v",
                "Gradient TIME all",
                "output comparison vs np=1/current-best baseline",
                "GPU count",
                "rank count",
                "CUDA_VISIBLE_DEVICES",
                "input gpus_p_node",
                "shot assignment",
            ],
            "acceptance_gate": "3-round repeat, all output comparisons pass, elapsed or Gradient TIME speedup >= 1.05x vs np=1 on same binary/case.",
        },
        "reopen_conditions": [
            "Run on a platform with at least 2 visible GPUs.",
            "Create input variants whose last non-empty line gpus_p_node equals the visible GPU count.",
            "Use one MPI rank per visible GPU unless a separate model justifies multiple ranks per GPU.",
        ],
    }


def write_markdown(model: dict[str, Any], path: Path) -> None:
    lines = [
        "# True Multi-GPU / Multi-Job Batching Protocol",
        "",
        "## Decision",
        "",
        f"- decision: `{model['decision']}`",
        f"- available GPUs on current RTX 5090 server: `{model['available_gpus']}`",
        "",
        "The current platform cannot validate true multi-GPU batching because it exposes only one GPU.",
        "Same-GPU multi-rank oversubscription has already been rejected, so this protocol is for a future multi-GPU run.",
        "",
        "## Current-Best Anchor",
        "",
        f"- alias: `{model['formal_current_best']['alias']}`",
        f"- mean elapsed: `{model['formal_current_best']['mean_elapsed_s']:.3f}s`",
        f"- mean Gradient: `{model['formal_current_best']['mean_gradient_s']:.6f}s`",
        f"- mean WP: `{model['formal_current_best']['mean_wp_s']:.6f}s`",
        f"- WP speedup vs zmem: `{fmt_ratio(model['formal_current_best']['wp_speedup_vs_zmem'])}`",
        "",
        "## Existing Code Requirements",
        "",
        "- `src/main.cu` maps ranks with `cudaSetDevice(mytid % gpus_p_node)`.",
        "- `gpus_p_node` is read from the input file, not inferred from `CUDA_VISIBLE_DEVICES`.",
        "- For true one-rank-per-GPU runs, `mpirun -np N`, `CUDA_VISIBLE_DEVICES` with `N` devices, and the input file's last line `gpus_p_node=N` must agree.",
        "- Shot assignment uses `sht_num[is * ntids + mytid]`, so rank count directly changes shot distribution.",
        "- Printed `WP computing time` is root-rank local for multi-rank runs; elapsed and `Gradient TIME all` are the scheduling metrics.",
        "",
        "## Shot-Balance Model",
        "",
        "| GPUs/ranks | runnable here | active shots/rank | ideal speedup | ideal efficiency | ideal elapsed | ideal Gradient |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for dist in model["distributions"]:
        lines.append(
            f"| {dist['ranks']} | `{dist['runnable_on_current_platform']}` | "
            f"`{dist['active_counts']}` | `{fmt_ratio(dist['ideal_shot_balance_speedup'])}` | "
            f"`{100.0 * dist['ideal_parallel_efficiency']:.1f}%` | "
            f"`{dist['estimated_elapsed_s_ideal']:.3f}s` | "
            f"`{dist['estimated_gradient_s_ideal']:.6f}s` |"
        )
    lines += [
        "",
        "## Gate",
        "",
        "- Do not run more same-GPU oversubscription probes for this case.",
        "- Do not claim multi-rank speedup from root-rank printed `WP computing time`.",
        "- Promote true multi-GPU batching only after a 3-round repeat where all output comparisons pass and elapsed or `Gradient TIME all` speedup is `>=1.05x`.",
        "",
        "## Minimal Future Command Shape",
        "",
        "```bash",
        "source ./env_5090.sh",
        "# create an input copy whose last line is N, e.g. 3 for three visible GPUs",
        "CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/run_benchmark.py \\",
        "  --case perf_1gpu_6shots --input input_perf_1gpu_6shots_gpus3.in \\",
        "  --np 3 --gpus 0,1,2 --tag true_multigpu_np3",
        "```",
        "",
        "The input override is necessary because `run_benchmark.py --gpus` controls `CUDA_VISIBLE_DEVICES`, while the CUDA device mapping inside the program uses the input file's `gpus_p_node`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--available-gpus", type=int, default=1)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    model = build_model(args)
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    write_markdown(model, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
