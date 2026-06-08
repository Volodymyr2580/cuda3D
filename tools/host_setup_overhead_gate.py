#!/usr/bin/env python3
"""Gate host/setup overhead optimization for the current perf case.

The CUDA kernels now account for most accepted speedups, but formal wall-clock
elapsed time still contains setup outside `Gradient TIME all`.  This model
separates what is measured, what is actually optimizable inside the solver, and
what should not be confused with CUDA-core speedup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def ratio(value: float) -> str:
    return f"{value:.4f}x"


def load_formal(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = sorted({item["label"] for item in data["records"]})
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    for item in data["records"]:
        by_label[item["label"]].append(item)
    rows = {}
    for label, records in by_label.items():
        elapsed = mean(float(r["elapsed_seconds"]) for r in records)
        gradient = mean(float(r["gradient"]) for r in records)
        wp = mean(float(r["wp"]) for r in records)
        setup_vs_gradient = elapsed - gradient
        setup_vs_wp = elapsed - wp
        rows[label] = {
            "mean_elapsed_s": elapsed,
            "mean_gradient_s": gradient,
            "mean_wp_s": wp,
            "elapsed_minus_gradient_s": setup_vs_gradient,
            "elapsed_minus_wp_s": setup_vs_wp,
            "elapsed_minus_gradient_share": setup_vs_gradient / elapsed,
            "elapsed_minus_wp_share": setup_vs_wp / elapsed,
            "perfect_remove_elapsed_minus_gradient_speedup": elapsed / gradient,
            "perfect_remove_elapsed_minus_wp_speedup": elapsed / wp,
        }
    return {
        "current_best_alias": data["current_best_alias"],
        "rows": rows,
        "summary_by_candidate": data["summary_by_candidate"],
        "zmem_mean_elapsed_s": rows["zmem"]["mean_elapsed_s"],
        "zmem_mean_gradient_s": rows["zmem"]["mean_gradient_s"],
        "zmem_mean_wp_s": rows["zmem"]["mean_wp_s"],
    }


def build_model(root: Path) -> dict[str, Any]:
    formal = load_formal(root / "reports/day_20260608/formal_current_best_table_20260608_182525/summary.json")
    best_label = formal["current_best_alias"]
    best = formal["rows"][best_label]
    zmem = formal["rows"]["zmem"]

    # Conservative fractions: most pre-gradient overhead is MPI/process startup,
    # file IO, velocity/nav setup, and one-time allocations.  Without a profile
    # proving a local hotspot, only a small fraction should be treated as a safe
    # implementation target.
    target_fractions = [0.1, 0.25, 0.5, 1.0]
    scenarios = []
    for frac in target_fractions:
        saved = best["elapsed_minus_gradient_s"] * frac
        candidate_elapsed = best["mean_elapsed_s"] - saved
        scenarios.append(
            {
                "removable_fraction_of_elapsed_minus_gradient": frac,
                "saved_s": saved,
                "candidate_elapsed_s": candidate_elapsed,
                "elapsed_speedup_vs_current_best": best["mean_elapsed_s"] / candidate_elapsed,
                "elapsed_speedup_vs_zmem": zmem["mean_elapsed_s"] / candidate_elapsed,
            }
        )

    required_for_5pct_elapsed = best["mean_elapsed_s"] - best["mean_elapsed_s"] / 1.05
    required_fraction = required_for_5pct_elapsed / best["elapsed_minus_gradient_s"]

    decision = "profile_before_host_setup_prototype"
    if required_fraction > 0.5:
        decision = "reject_blind_host_setup_prototype"

    return {
        "decision": decision,
        "formal_current_best": {
            "label": best_label,
            **best,
        },
        "zmem": zmem,
        "current_speedups_vs_zmem": {
            "elapsed": zmem["mean_elapsed_s"] / best["mean_elapsed_s"],
            "gradient": zmem["mean_gradient_s"] / best["mean_gradient_s"],
            "wp": zmem["mean_wp_s"] / best["mean_wp_s"],
        },
        "required_for_5pct_elapsed_speedup": {
            "required_saved_s": required_for_5pct_elapsed,
            "required_fraction_of_elapsed_minus_gradient": required_fraction,
        },
        "scenarios": scenarios,
        "measurement_boundary": {
            "gradient_time_all": "Inside cal_fwi_grad_3d after early allocation/setup and before final MPI finalize.",
            "elapsed": "Whole MPI program execution measured by /usr/bin/time in run_benchmark.py.",
            "not_in_elapsed": "run_benchmark.py output copy occurs after the timed command and is not part of elapsed.",
        },
        "gate": {
            "no_blind_code_change": True,
            "reopen_condition": "Use Nsight Systems / CPU sampling / targeted timers to show a single host/setup hotspot with >=5% elapsed-speedup ceiling.",
            "forbidden_shortcuts": [
                "Do not relabel timing by moving t3 earlier/later and call it a speedup.",
                "Do not optimize output copy for this elapsed metric; it is outside /usr/bin/time.",
                "Do not change input semantics or skip correctness/output generation.",
            ],
        },
    }


def write_markdown(model: dict[str, Any], path: Path) -> None:
    best = model["formal_current_best"]
    lines = [
        "# Host / Setup Overhead Gate",
        "",
        "## Decision",
        "",
        f"- decision: `{model['decision']}`",
        "",
        "## Timing Anchor",
        "",
        f"- current best: `{best['label']}`",
        f"- mean elapsed: `{best['mean_elapsed_s']:.3f}s`",
        f"- mean Gradient TIME all: `{best['mean_gradient_s']:.6f}s`",
        f"- mean WP: `{best['mean_wp_s']:.6f}s`",
        f"- elapsed - Gradient: `{best['elapsed_minus_gradient_s']:.6f}s` / `{100.0 * best['elapsed_minus_gradient_share']:.2f}%`",
        f"- elapsed - WP: `{best['elapsed_minus_wp_s']:.6f}s` / `{100.0 * best['elapsed_minus_wp_share']:.2f}%`",
        "",
        "Current accepted speedups vs zmem:",
        "",
        f"- elapsed: `{ratio(model['current_speedups_vs_zmem']['elapsed'])}`",
        f"- Gradient: `{ratio(model['current_speedups_vs_zmem']['gradient'])}`",
        f"- WP: `{ratio(model['current_speedups_vs_zmem']['wp'])}`",
        "",
        "## 5% Gate",
        "",
        f"- saved time required for `1.05x` elapsed speedup vs current best: `{model['required_for_5pct_elapsed_speedup']['required_saved_s']:.6f}s`",
        f"- required fraction of `elapsed - Gradient`: `{100.0 * model['required_for_5pct_elapsed_speedup']['required_fraction_of_elapsed_minus_gradient']:.2f}%`",
        "",
        "| removable fraction | saved s | candidate elapsed | speedup vs current best | speedup vs zmem |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in model["scenarios"]:
        lines.append(
            f"| `{100.0 * item['removable_fraction_of_elapsed_minus_gradient']:.0f}%` | "
            f"`{item['saved_s']:.6f}` | `{item['candidate_elapsed_s']:.6f}` | "
            f"`{ratio(item['elapsed_speedup_vs_current_best'])}` | "
            f"`{ratio(item['elapsed_speedup_vs_zmem'])}` |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "- `Gradient TIME all` starts inside `cal_fwi_grad_3d` after early allocation/setup.",
        "- elapsed is the whole MPI program execution measured by `/usr/bin/time` in `run_benchmark.py`.",
        "- output copying done by `run_benchmark.py` happens after the timed command, so optimizing it does not improve this elapsed metric.",
        "",
        "## Gate",
        "",
        "Do not make blind host/setup code changes yet.  Reopen only after Nsight Systems, CPU sampling, or targeted timers identify a concrete host/setup hotspot with `>=5%` elapsed-speedup ceiling.",
        "",
        "Forbidden shortcuts:",
        "",
        "- Do not move timing markers and call that a speedup.",
        "- Do not skip output generation or correctness work.",
        "- Do not optimize runner output copy for this elapsed metric.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()

    model = build_model(Path(args.root))
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    write_markdown(model, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
