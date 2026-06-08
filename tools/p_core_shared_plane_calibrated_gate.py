#!/usr/bin/env python3
"""Calibrate p_core shared-plane budgets with the failed ZX prototype."""

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


def local_speedup_from_global(global_speedup: float, local_share: float) -> float | None:
    denom = 1.0 / global_speedup - (1.0 - local_share)
    if denom <= 0.0:
        return None
    return local_share / denom


def global_speedup_from_local(local_speedup: float, local_share: float) -> float:
    return 1.0 / ((1.0 - local_share) + local_share / local_speedup)


def analyze(budget_path: Path, proto_path: Path, target_speedup: float) -> dict[str, Any]:
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    p_core_share = float(budget["profile"]["p_core_share"])

    tested_shape = [16, 16, 1]
    tested_mode = "zx_shared_y_global"
    tested = None
    for item in budget["candidates"]:
        if item["shape_zxy"] == tested_shape and item["mode"] == tested_mode:
            tested = item
            break
    if tested is None:
        raise ValueError("cannot find tested 16x16x1 zx_shared_y_global candidate")

    wp_global = float(proto["wp_speedup_vs_current_best"])
    grad_global = float(proto["gradient_speedup_vs_current_best"])
    observed_wp_local = local_speedup_from_global(wp_global, p_core_share)
    observed_grad_local = local_speedup_from_global(grad_global, p_core_share)
    tested_model_local = float(tested["p_core_byte_speedup_ceiling"])
    calibration_wp = observed_wp_local / tested_model_local if observed_wp_local is not None else None
    calibration_grad = observed_grad_local / tested_model_local if observed_grad_local is not None else None

    candidates: list[dict[str, Any]] = []
    for item in budget["candidates"]:
        model_local = float(item["p_core_byte_speedup_ceiling"])
        calibrated_wp_local = model_local * calibration_wp if calibration_wp is not None else None
        calibrated_grad_local = model_local * calibration_grad if calibration_grad is not None else None
        candidates.append(
            {
                "shape_zxy": item["shape_zxy"],
                "mode": item["mode"],
                "model_p_core_speedup": model_local,
                "model_sampled_main_speedup": item["sampled_main_speedup_ceiling"],
                "calibrated_wp_local_speedup": calibrated_wp_local,
                "calibrated_wp_sampled_speedup": (
                    global_speedup_from_local(calibrated_wp_local, p_core_share)
                    if calibrated_wp_local and calibrated_wp_local > 0
                    else None
                ),
                "calibrated_gradient_local_speedup": calibrated_grad_local,
                "calibrated_gradient_sampled_speedup": (
                    global_speedup_from_local(calibrated_grad_local, p_core_share)
                    if calibrated_grad_local and calibrated_grad_local > 0
                    else None
                ),
                "shared_kib": item["shared_bytes"] / 1024.0,
                "p1_floats_per_output": item["p1_global_floats_per_output"],
            }
        )
    candidates.sort(
        key=lambda item: item["calibrated_wp_sampled_speedup"]
        if item["calibrated_wp_sampled_speedup"] is not None
        else -math.inf,
        reverse=True,
    )
    best_wp = candidates[0]
    best_grad = max(
        candidates,
        key=lambda item: item["calibrated_gradient_sampled_speedup"]
        if item["calibrated_gradient_sampled_speedup"] is not None
        else -math.inf,
    )
    decision = (
        "reject_current_shared_plane_family"
        if (best_wp["calibrated_wp_sampled_speedup"] or 0.0) < target_speedup
        and (best_grad["calibrated_gradient_sampled_speedup"] or 0.0) < target_speedup
        else "allow_shape_specific_reopen"
    )
    return {
        "inputs": {
            "budget_json": str(budget_path),
            "prototype_summary_json": str(proto_path),
            "target_sampled_main_speedup": target_speedup,
        },
        "calibration_anchor": {
            "tested_shape_zxy": tested_shape,
            "tested_mode": tested_mode,
            "p_core_share": p_core_share,
            "model_p_core_speedup": tested_model_local,
            "model_sampled_main_speedup": tested["sampled_main_speedup_ceiling"],
            "observed_wp_global_speedup": wp_global,
            "observed_gradient_global_speedup": grad_global,
            "observed_wp_local_p_core_speedup": observed_wp_local,
            "observed_gradient_local_p_core_speedup": observed_grad_local,
            "wp_model_to_observed_factor": calibration_wp,
            "gradient_model_to_observed_factor": calibration_grad,
        },
        "candidates": candidates,
        "gate": {
            "decision": decision,
            "best_wp_calibrated_candidate": best_wp,
            "best_gradient_calibrated_candidate": best_grad,
            "reason": (
                "The failed 16x16x1 prototype shows shared-fill/control/warp-mapping overhead overwhelms "
                "the byte-model savings.  Applying that empirical calibration pulls every current "
                "shared-plane candidate below the >=5% gate."
                if decision == "reject_current_shared_plane_family"
                else "At least one candidate remains above the gate after calibration."
            ),
            "reopen_condition": (
                "Reopen only with a materially different warp/coalescing design and a source/profile model "
                "that separately accounts for shared fill, synchronization, and control overhead."
            ),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    anchor = result["calibration_anchor"]
    gate = result["gate"]
    lines = [
        "# P-Core Shared-Plane Calibrated Gate",
        "",
        "## Calibration Anchor",
        "",
        f"- tested shape: `{anchor['tested_shape_zxy']}` / `{anchor['tested_mode']}`",
        f"- p_core share: `{pct(anchor['p_core_share'])}`",
        f"- modeled p_core local speedup: `{ratio(anchor['model_p_core_speedup'])}`",
        f"- modeled sampled-main speedup: `{ratio(anchor['model_sampled_main_speedup'])}`",
        f"- observed WP global speedup: `{ratio(anchor['observed_wp_global_speedup'])}`",
        f"- observed Gradient global speedup: `{ratio(anchor['observed_gradient_global_speedup'])}`",
        f"- inferred WP-local p_core speedup: `{ratio(anchor['observed_wp_local_p_core_speedup'])}`",
        f"- inferred Gradient-local p_core speedup: `{ratio(anchor['observed_gradient_local_p_core_speedup'])}`",
        f"- WP model-to-observed factor: `{ratio(anchor['wp_model_to_observed_factor'])}`",
        f"- Gradient model-to-observed factor: `{ratio(anchor['gradient_model_to_observed_factor'])}`",
        "",
        "## Top Calibrated Candidates",
        "",
        "| shape | mode | model sampled | calibrated WP sampled | calibrated Gradient sampled | shared KiB |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in result["candidates"][:10]:
        lines.append(
            f"| `{item['shape_zxy']}` | `{item['mode']}` | `{ratio(item['model_sampled_main_speedup'])}` | "
            f"`{ratio(item['calibrated_wp_sampled_speedup'])}` | "
            f"`{ratio(item['calibrated_gradient_sampled_speedup'])}` | `{item['shared_kib']:.2f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- decision: `{gate['decision']}`",
            f"- reason: {gate['reason']}",
            f"- reopen condition: {gate['reopen_condition']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-json", default="reports/day_20260608/p_core_shared_plane_budget.json")
    parser.add_argument(
        "--prototype-summary-json",
        default="reports/day_20260608/p_core_zx_prototype_20260608_2158/perf6_repeat_summary.json",
    )
    parser.add_argument("--target-speedup", type=float, default=1.05)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    budget_path = Path(args.budget_json)
    proto_path = Path(args.prototype_summary_json)
    if not budget_path.is_absolute():
        budget_path = root / budget_path
    if not proto_path.is_absolute():
        proto_path = root / proto_path
    result = analyze(budget_path, proto_path, args.target_speedup)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_markdown(result)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(md)


if __name__ == "__main__":
    main()
