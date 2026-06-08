#!/usr/bin/env python3
"""Gate math-level pressure state representation candidates.

This model checks whether a pressure-state redesign can really remove the hot
old-p0/cw2 traffic identified after the accepted len16 pressure-PML path.  It
is intentionally conservative: a CUDA prototype is allowed only if the
candidate preserves the numerical model and has a credible >=5% repeat-speed
ceiling after its extra traffic is counted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FLOAT_BYTES = 4


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}x"


def amdahl_speedup(local_speedup: float, share: float) -> float:
    if share <= 0.0:
        return 1.0
    if local_speedup <= 0.0:
        return 0.0
    return 1.0 / ((1.0 - share) + share / local_speedup)


def load_writeback_gate(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = data["inputs"]["profile"]
    source = data["inputs"]["source"]
    derived = data["derived"]
    return {
        "profile": profile,
        "source_groups": source["grouped_sample_shares"],
        "source_top_lines": source["top_lines"],
        "derived": derived,
    }


def load_formal(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "decision": data["decision"],
        "current_best_alias": data["current_best_alias"],
        "summary_by_candidate": data["summary_by_candidate"],
    }


def compute_candidates(writeback: dict[str, Any], formal: dict[str, Any]) -> dict[str, Any]:
    profile = writeback["profile"]
    groups = writeback["source_groups"]
    p_core_share = profile["p_core_us"] / profile["sampled_main_us"]
    v_pml_share = profile["v_pml_us"] / profile["sampled_main_us"]
    pressure_share = profile["p_pml_total_sampled_main_share"]
    len16_share = profile["p_pml_len16_sampled_main_share"]
    final_group_share_in_len16 = groups["final_p0_p1_cw2_update"]
    cw2_line_share_in_len16 = 0.0
    p0_line_share_in_len16 = 0.0
    for item in writeback["source_top_lines"]:
        text = item["text"]
        if "__ldg(cw2+base)" in text:
            cw2_line_share_in_len16 += item["sample_share_of_parsed"]
        if "p0[base]=2*__ldg(p1+base)-p0[base]" in text:
            p0_line_share_in_len16 += item["sample_share_of_parsed"]

    current_bytes = {
        "p_prev_read": FLOAT_BYTES,
        "p_cur_read": FLOAT_BYTES,
        "cw2_read": FLOAT_BYTES,
        "p_next_write": FLOAT_BYTES,
        "total": 4 * FLOAT_BYTES,
    }
    delta_bytes = {
        "p_cur_read": FLOAT_BYTES,
        "delta_read": FLOAT_BYTES,
        "cw2_read": FLOAT_BYTES,
        "p_next_write": FLOAT_BYTES,
        "delta_next_write": FLOAT_BYTES,
        "total": 5 * FLOAT_BYTES,
    }
    delta_local_speedup = current_bytes["total"] / delta_bytes["total"]

    # From cuda_fd3d_p_core_ns: one center value, z values staged from p1, and
    # 14 x-neighbor + 14 y-neighbor global p1 reads.  A q=p/cw2-only state
    # would need cw2 for every pressure value reconstructed for those stencils.
    p_core_min_pressure_values_per_output = 1 + 14 + 14
    q_only_extra_cw2_loads_per_core_output = p_core_min_pressure_values_per_output - 1

    candidates: dict[str, dict[str, Any]] = {}

    candidates["delta_pressure_state"] = {
        "equivalence": "exact algebra for the time recurrence, but changes stored state",
        "state": "store p_cur and delta=p_cur-p_prev",
        "traffic_model": {
            "current_pressure_update_bytes_per_point": current_bytes,
            "candidate_min_bytes_per_point": delta_bytes,
            "local_update_speedup_ceiling": delta_local_speedup,
            "sampled_main_effect_if_all_pressure_updates_followed_this": amdahl_speedup(delta_local_speedup, pressure_share),
        },
        "decision": "reject_cuda_prototype",
        "reason": "It removes the old-p read but adds a delta write; minimum pressure-update bytes rise from 16 to 20 per point before any extra bookkeeping.",
    }

    q_remove_cw2_line_ceiling = amdahl_speedup(math.inf, cw2_line_share_in_len16 * len16_share)
    q_remove_final_group_ceiling = amdahl_speedup(math.inf, final_group_share_in_len16 * len16_share)
    candidates["scaled_pressure_q_only"] = {
        "equivalence": "time update algebra is exact only if p=cw2*q is reconstructed wherever pressure is used",
        "state": "store q=p/cw2 instead of p",
        "traffic_model": {
            "utopian_remove_len16_cw2_source_line_sampled_speedup": q_remove_cw2_line_ceiling,
            "utopian_remove_len16_final_group_sampled_speedup": q_remove_final_group_ceiling,
            "p_core_min_pressure_values_per_output": p_core_min_pressure_values_per_output,
            "extra_cw2_loads_per_p_core_output_min": q_only_extra_cw2_loads_per_core_output,
            "p_core_sampled_main_share_at_risk": p_core_share,
            "v_pml_sampled_main_share_at_risk": v_pml_share,
        },
        "decision": "reject_cuda_prototype",
        "reason": (
            "It can remove one cw2 load at the final update only by adding cw2 reconstruction to p_core/v_pml "
            "pressure stencils.  p_core alone would go from one cw2 read per output to at least 29 pressure-value "
            "reconstructions per output."
        ),
    }

    dual_min_bytes = 8 * FLOAT_BYTES
    candidates["scaled_pressure_dual_p_and_q"] = {
        "equivalence": "can preserve existing pressure stencils if both p and q are kept coherent",
        "state": "store p for stencils and q for cw2-free update",
        "traffic_model": {
            "current_pressure_update_bytes_per_point": current_bytes["total"],
            "candidate_min_bytes_per_point": dual_min_bytes,
            "local_update_speedup_ceiling": current_bytes["total"] / dual_min_bytes,
            "sampled_main_effect_if_all_pressure_updates_followed_this": amdahl_speedup(current_bytes["total"] / dual_min_bytes, pressure_share),
        },
        "decision": "reject_cuda_prototype",
        "reason": "Keeping both representations avoids stencil reconstruction but doubles pressure state write/read work.",
    }

    full_velocity_extra_bytes = 3 * 2 * FLOAT_BYTES
    candidates["first_order_full_domain_velocity_pressure"] = {
        "equivalence": "not bitwise equivalent; would replace the current mixed second-order core with a new first-order scheme",
        "state": "single pressure field plus full-domain vx/vy/vz updates",
        "traffic_model": {
            "old_p_read_saved_per_core_point": FLOAT_BYTES,
            "minimum_extra_velocity_read_write_bytes_per_core_point": full_velocity_extra_bytes,
            "extra_to_saved_byte_ratio": full_velocity_extra_bytes / FLOAT_BYTES,
            "p_core_sampled_main_share": p_core_share,
        },
        "decision": "reject_cuda_prototype",
        "reason": "It saves one old-p read but introduces full-domain velocity state traffic and a different numerical scheme.",
    }

    candidates["precomputed_cw2dt"] = {
        "equivalence": "exact if dt is fixed for a run",
        "state": "store cw2dt instead of cw2 or precompute cw2*dt",
        "traffic_model": {
            "bytes_removed": 0,
            "reason": "The same 4-byte model coefficient is still loaded once per pressure update.",
        },
        "decision": "reject_cuda_prototype",
        "reason": "It removes a multiply, not the global-memory traffic that dominates the source profile.",
    }

    half_cw2_ideal = amdahl_speedup(2.0, cw2_line_share_in_len16 * len16_share)
    candidates["half_or_compressed_cw2"] = {
        "equivalence": "requires precision relaxation or a separate quantization proof",
        "state": "store cw2 in fp16 or compressed form",
        "traffic_model": {
            "ideal_len16_cw2_line_2x_speedup_sampled_main": half_cw2_ideal,
            "known_correctness_gate": "rel L2 <= 1e-5, no NaN/Inf",
        },
        "decision": "reject_for_current_exactness_gate",
        "reason": "Even the ideal len16 cw2-line-only benefit is small, and precision changes are outside the current exactness contract.",
    }

    candidates["cpml_mem_dzz_rescaled_state"] = {
        "equivalence": "algebraic rescaling can keep the recurrence but cannot remove one read and one write per step",
        "state": "store a transformed mem_dzz variable",
        "traffic_model": {
            "mem_dzz_source_share_in_len16": groups["cpml_mem_dzz_update"],
            "required_mem_dzz_group_speedup_if_alone": writeback["derived"]["mem_dzz_group_speedup_required_if_alone"],
        },
        "decision": "reject_cuda_prototype",
        "reason": "The CPML state is recursive and has no intra-step reuse target; rescaling changes arithmetic but not state traffic.",
    }

    accepted: list[str] = []
    rejected = [name for name, item in candidates.items() if item["decision"].startswith("reject")]
    return {
        "inputs": {
            "writeback_gate": writeback,
            "formal": formal,
        },
        "baseline": {
            "sampled_main_us": profile["sampled_main_us"],
            "p_core_share": p_core_share,
            "v_pml_share": v_pml_share,
            "pressure_pml_share": pressure_share,
            "len16_packed_share": len16_share,
            "formal_current_best_wp_speedup_vs_zmem": formal["summary_by_candidate"]["len16_current_best"]["mean_wp_speedup_vs_zmem"],
        },
        "current_pressure_update_bytes_per_point": current_bytes,
        "candidates": candidates,
        "gate": {
            "decision": "reject_pressure_state_representation_cuda_prototype",
            "accepted_candidates": accepted,
            "rejected_candidates": rejected,
            "reason": (
                "No exact pressure-state representation removes old-p/cw2 or mem_dzz traffic without moving larger "
                "traffic into p_core/v_pml stencils, adding another state write, changing the numerical scheme, or "
                "relaxing precision."
            ),
            "next_allowed": [
                "PML vx/vy round-trip ownership design with a >=5% model before CUDA",
                "source-aware multi-step or wavefront design only if it solves synchronization/halo ownership",
                "precision-relaxation study only with an explicit new tolerance policy",
            ],
            "prohibited": [
                "q=p/cw2 pressure-state prototype under the current variable-cw2 stencil path",
                "delta pressure state prototype",
                "first-order full-domain velocity-pressure rewrite as a drop-in optimization",
                "precomputed cw2dt or compressed cw2 prototype under the current exactness gate",
                "CPML mem_dzz algebraic rescaling without state-traffic removal",
            ],
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    b = result["baseline"]
    gate = result["gate"]
    lines = [
        "# Pressure State Representation Gate",
        "",
        "## Context",
        "",
        f"- sampled main: `{b['sampled_main_us']:.3f}us`",
        f"- p_core share: `{pct(b['p_core_share'])}`",
        f"- v_pml share: `{pct(b['v_pml_share'])}`",
        f"- pressure-PML share: `{pct(b['pressure_pml_share'])}`",
        f"- len16 packed pressure-PML share: `{pct(b['len16_packed_share'])}`",
        f"- formal current-best WP speedup vs zmem: `{ratio(b['formal_current_best_wp_speedup_vs_zmem'])}`",
        "",
        "Current second-order pressure update state traffic per point:",
        "",
        "| item | bytes |",
        "| --- | ---: |",
    ]
    for key, value in result["current_pressure_update_bytes_per_point"].items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "## Candidate Decisions",
            "",
            "| candidate | decision | equivalence | traffic signal | reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name, item in result["candidates"].items():
        traffic = item["traffic_model"]
        if name == "delta_pressure_state":
            signal = (
                f"bytes {traffic['current_pressure_update_bytes_per_point']['total']} -> "
                f"{traffic['candidate_min_bytes_per_point']['total']}; "
                f"sampled effect {ratio(traffic['sampled_main_effect_if_all_pressure_updates_followed_this'])}"
            )
        elif name == "scaled_pressure_q_only":
            signal = (
                f"p_core cw2 reconstructions >= {traffic['p_core_min_pressure_values_per_output']} per output; "
                f"p_core+v_pml at risk {pct(traffic['p_core_sampled_main_share_at_risk'] + traffic['v_pml_sampled_main_share_at_risk'])}"
            )
        elif name == "scaled_pressure_dual_p_and_q":
            signal = (
                f"bytes {traffic['current_pressure_update_bytes_per_point']} -> "
                f"{traffic['candidate_min_bytes_per_point']}; "
                f"sampled effect {ratio(traffic['sampled_main_effect_if_all_pressure_updates_followed_this'])}"
            )
        elif name == "first_order_full_domain_velocity_pressure":
            signal = (
                f"save {traffic['old_p_read_saved_per_core_point']}B old-p, "
                f"add >= {traffic['minimum_extra_velocity_read_write_bytes_per_core_point']}B velocity/core point"
            )
        elif name == "half_or_compressed_cw2":
            signal = f"ideal len16 cw2-line sampled speedup {ratio(traffic['ideal_len16_cw2_line_2x_speedup_sampled_main'])}"
        elif name == "cpml_mem_dzz_rescaled_state":
            signal = f"mem_dzz local speedup required {ratio(traffic['required_mem_dzz_group_speedup_if_alone'])}"
        else:
            signal = str(traffic.get("reason", traffic))
        lines.append(
            f"| `{name}` | `{item['decision']}` | {item['equivalence']} | {signal} | {item['reason']} |"
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
    parser.add_argument("--writeback-gate", default="reports/day_20260608/pressure_pml_writeback_state_model.json")
    parser.add_argument("--formal-summary", default="reports/day_20260608/formal_current_best_table_20260608_182525/summary.json")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    writeback_path = Path(args.writeback_gate)
    if not writeback_path.is_absolute():
        writeback_path = root / writeback_path
    formal_path = Path(args.formal_summary)
    if not formal_path.is_absolute():
        formal_path = root / formal_path

    result = compute_candidates(load_writeback_gate(writeback_path), load_formal(formal_path))

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
