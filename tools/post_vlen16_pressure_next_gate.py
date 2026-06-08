#!/usr/bin/env python3
"""Gate pressure-PML next routes after the v-PML len16 candidate.

The velocity half-warp prototype moved a small amount of time out of v-PML, so
the next decision has to be re-anchored on the latest sampled main-kernel
profile.  This tool converts the post-vlen16 NCU summary into Amdahl
requirements and explicitly rejects routes that only repeat known failed
micro-tuning families.
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


def amdahl(local_speedup: float, share: float) -> float:
    if share <= 0.0:
        return 1.0
    if math.isinf(local_speedup):
        return 1.0 / (1.0 - share)
    if local_speedup <= 0.0:
        return 0.0
    return 1.0 / ((1.0 - share) + share / local_speedup)


def required_local_speedup(target_speedup: float, share: float) -> float | None:
    if share <= 0.0:
        return None
    denom = (1.0 / target_speedup) - (1.0 - share)
    if denom <= 0.0:
        return math.inf
    return share / denom


def kernel_duration(profile: dict[str, Any], kernel_name: str) -> float:
    item = profile["kernels"].get(kernel_name)
    if item is None:
        return 0.0
    return float(item["metrics"]["duration_ns"])


def load_latest_profile(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    profiles = {item["label"]: item for item in data["profiles"]}
    if "v_pml_len16_short" not in profiles:
        raise KeyError("expected profile label 'v_pml_len16_short'")
    profile = profiles["v_pml_len16_short"]

    p_core = kernel_duration(profile, "cuda_fd3d_p_core_ns")
    p_len16 = kernel_duration(profile, "cuda_fd3d_p_pml_len16_halfwarp_ns")
    p_residual = kernel_duration(profile, "cuda_fd3d_p_pml_tile_ns")
    v_len16 = kernel_duration(profile, "cuda_fd3d_v_pml_len16_halfwarp_ns")
    v_residual = kernel_duration(profile, "cuda_fd3d_v_pml_tile_ns")
    pressure_total = p_len16 + p_residual
    velocity_total = v_len16 + v_residual
    sampled_main = p_core + pressure_total + velocity_total

    def share(value: float) -> float:
        return value / sampled_main if sampled_main > 0.0 else 0.0

    return {
        "profile_label": "v_pml_len16_short",
        "duration_unit": "us",
        "p_core_us": p_core,
        "p_pml_len16_us": p_len16,
        "p_pml_residual_us": p_residual,
        "p_pml_total_us": pressure_total,
        "v_pml_len16_us": v_len16,
        "v_pml_residual_us": v_residual,
        "v_pml_total_us": velocity_total,
        "sampled_main_us": sampled_main,
        "shares": {
            "p_core": share(p_core),
            "p_pml_len16": share(p_len16),
            "p_pml_residual": share(p_residual),
            "p_pml_total": share(pressure_total),
            "v_pml_len16": share(v_len16),
            "v_pml_residual": share(v_residual),
            "v_pml_total": share(velocity_total),
        },
        "raw_kernel_metrics": profile["kernels"],
    }


def line_share(hotlines: list[dict[str, Any]], *needles: str) -> float:
    total = 0.0
    for item in hotlines:
        text = str(item.get("text", ""))
        if any(needle in text for needle in needles):
            total += float(item["sample_share_of_parsed"])
    return total


def load_source_groups(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    hotlines = data["top_lines"]
    final_update = line_share(
        hotlines,
        "p0[base]=2*__ldg(p1+base)-p0[base]",
        "+__ldg(cw2+base)*dt*(c1+c2+c3);",
    )
    mem_dzz = line_share(hotlines, "mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);")
    z_cache = line_share(hotlines, "vz_line_cache")
    branch_control = line_share(
        hotlines,
        "const PmlTile tile",
        "if (blockIdx.x >= ntile)",
        "} else if",
        "const size_t ts2",
        "const size_t base",
        "const int gtid1",
    )
    grouped = {
        "final_p0_p1_cw2_update": final_update,
        "cpml_mem_dzz_update": mem_dzz,
        "z_cache_shared_loads": z_cache,
        "address_branch_control_visible": branch_control,
    }
    grouped["other_or_unparsed"] = max(0.0, 1.0 - sum(grouped.values()))
    return {
        "source_page": data.get("source_page"),
        "parsed_total_samples": int(data["parsed_total_samples"]),
        "grouped_sample_shares": grouped,
        "top_lines": hotlines,
    }


def subgroup_to_sampled_speedup(
    subgroup_share_in_kernel: float,
    subgroup_speedup: float,
    kernel_share_of_sampled_main: float,
) -> dict[str, float]:
    kernel_speedup = amdahl(subgroup_speedup, subgroup_share_in_kernel)
    return {
        "kernel_speedup": kernel_speedup,
        "sampled_main_speedup": amdahl(kernel_speedup, kernel_share_of_sampled_main),
    }


def compute_result(
    profile: dict[str, Any],
    source: dict[str, Any],
    target_speedup: float,
) -> dict[str, Any]:
    shares = profile["shares"]
    group = source["grouped_sample_shares"]
    final_share = group["final_p0_p1_cw2_update"]
    mem_share = group["cpml_mem_dzz_update"]
    combined_share = final_share + mem_share

    required_by_region = {
        name: required_local_speedup(target_speedup, share)
        for name, share in shares.items()
    }

    p_len16_req = required_by_region["p_pml_len16"]
    subgroup_required = {
        "final_p0_p1_cw2_if_alone": (
            required_local_speedup(p_len16_req, final_share)
            if p_len16_req is not None
            else None
        ),
        "cpml_mem_dzz_if_alone": (
            required_local_speedup(p_len16_req, mem_share)
            if p_len16_req is not None
            else None
        ),
        "final_plus_mem_dzz": (
            required_local_speedup(p_len16_req, combined_share)
            if p_len16_req is not None
            else None
        ),
    }

    pressure_scenarios = {
        "p_pml_total_1_10x": amdahl(1.10, shares["p_pml_total"]),
        "p_pml_total_1_25x": amdahl(1.25, shares["p_pml_total"]),
        "p_pml_len16_1_25x": amdahl(1.25, shares["p_pml_len16"]),
        "p_pml_residual_1_25x": amdahl(1.25, shares["p_pml_residual"]),
        "len16_final_update_2x": subgroup_to_sampled_speedup(
            final_share,
            2.0,
            shares["p_pml_len16"],
        )["sampled_main_speedup"],
        "len16_mem_dzz_2x": subgroup_to_sampled_speedup(
            mem_share,
            2.0,
            shares["p_pml_len16"],
        )["sampled_main_speedup"],
        "len16_final_plus_mem_dzz_1_5x": subgroup_to_sampled_speedup(
            combined_share,
            1.5,
            shares["p_pml_len16"],
        )["sampled_main_speedup"],
    }

    fusion_scenarios = {
        "perfect_remove_v_len16_time": amdahl(math.inf, shares["v_pml_len16"]),
        "perfect_remove_all_v_pml_time": amdahl(math.inf, shares["v_pml_total"]),
        "halve_all_v_pml_time": amdahl(2.0, shares["v_pml_total"]),
    }

    concrete_routes = {
        "pressure_writeback_syntax_microtuning": {
            "decision": "reject",
            "reason": "p0 __ldg and local new_mem variants were already tested at noise-level speedup.",
        },
        "pressure_len23_or_exact_descriptor": {
            "decision": "reject",
            "reason": "Post-len16 compact descriptor budget predicts only about 1.5% calibrated sampled-main speedup.",
        },
        "pressure_branch_only_specialization": {
            "decision": "reject",
            "reason": "Residual branch efficiency is already 83.32%; branch-only routes do not remove the dominant final update/state dependency.",
        },
        "v_pml_descriptor_expansion": {
            "decision": "reject",
            "reason": "After v-len16, total v-PML share is only 18.37% and descriptor overhead has no >=5% repeat-speedup model.",
        },
        "direct_v_pressure_zface_fusion": {
            "decision": "reject_as_previous_failed_family",
            "reason": "Naive fusion belongs to the already rejected z-face VP fusion/shared-VP family unless a new ownership model avoids duplicate halo/control costs.",
        },
        "new_pressure_or_wave_step_ownership_model": {
            "decision": "allow_design_only",
            "reason": "Pressure-PML remains 48.63% of sampled main; only a model that removes real state/writeback traffic should reach the next CUDA prototype gate.",
        },
        "formal_current_best_benchmark_table": {
            "decision": "allow",
            "reason": "Current cumulative speedup is partly multiplicative across sessions; a same-session zmem/direct-fill/pressure-len16/current-best table is needed before a major phase switch.",
        },
    }

    gate = {
        "decision": "no_new_micro_cuda_prototype",
        "reason": (
            "The post-vlen16 profile leaves pressure-PML as the dominant target, but the remaining "
            "pressure time is concentrated in required pressure writeback and recursive CPML z-state "
            "operations.  Existing syntax/cache/descriptor branches do not meet the >=5% repeat gate."
        ),
        "next_action": (
            "Run a formal same-session zmem/direct-fill/pressure-len16/current-best benchmark table, "
            "then open only a design-level pressure/wave-step ownership model with an explicit >=5% ceiling."
        ),
        "allowed_next": [
            "formal same-session benchmark table for zmem, direct-fill, pressure-len16, and current-best",
            "design-only pressure state/writeback ownership model with equivalence proof",
            "design-only wave-step ownership model that reduces vx/vy or pressure writeback global traffic",
        ],
        "prohibited_next": [
            "repeat p0 __ldg, local new_mem, ptxas cache-policy, z-cache fill, or shared-z-cache tuning",
            "pressure length-23 or exact active-point descriptor CUDA prototype",
            "v-PML descriptor/point-list expansion after accepted v-len16",
            "direct z-face VP fusion/shared-VP retry without a materially new ownership proof",
            "random p-core block/register/shared-plane sweeps from the rejected family",
        ],
    }

    return {
        "inputs": {
            "target_speedup": target_speedup,
            "profile": profile,
            "source_groups": source,
        },
        "derived": {
            "required_local_speedup_by_region": required_by_region,
            "len16_source_subgroup_required_speedup": subgroup_required,
            "pressure_scenario_sampled_speedups": pressure_scenarios,
            "fusion_scenario_sampled_speedups": fusion_scenarios,
        },
        "concrete_routes": concrete_routes,
        "gate": gate,
    }


def render_markdown(result: dict[str, Any]) -> str:
    profile = result["inputs"]["profile"]
    shares = profile["shares"]
    derived = result["derived"]
    source = result["inputs"]["source_groups"]
    groups = source["grouped_sample_shares"]
    gate = result["gate"]

    lines = [
        "# Post-VLen16 Pressure Next Gate",
        "",
        "## Context",
        "",
        f"- profile: `{profile['profile_label']}`",
        f"- duration unit: `{profile['duration_unit']}`",
        f"- sampled main total: `{profile['sampled_main_us']:.3f}us`",
        f"- target local gate: `{ratio(result['inputs']['target_speedup'])}` sampled-main speedup",
        "",
        "| region | duration | sampled-main share | local speedup required for 5% sampled-main |",
        "| --- | ---: | ---: | ---: |",
    ]
    req = derived["required_local_speedup_by_region"]
    for name, duration in [
        ("p_core", profile["p_core_us"]),
        ("p_pml_len16", profile["p_pml_len16_us"]),
        ("p_pml_residual", profile["p_pml_residual_us"]),
        ("p_pml_total", profile["p_pml_total_us"]),
        ("v_pml_len16", profile["v_pml_len16_us"]),
        ("v_pml_residual", profile["v_pml_residual_us"]),
        ("v_pml_total", profile["v_pml_total_us"]),
    ]:
        lines.append(
            f"| `{name}` | `{duration:.3f}us` | `{pct(shares[name])}` | `{ratio(req[name])}` |"
        )

    lines.extend(
        [
            "",
            "## Pressure Source Anchor",
            "",
            f"- source profile: `{source['source_page']}`",
            f"- parsed samples: `{source['parsed_total_samples']}`",
            "",
            "| len16 source group | share of packed pressure kernel samples |",
            "| --- | ---: |",
            f"| final `p0/p1/cw2` update | `{pct(groups['final_p0_p1_cw2_update'])}` |",
            f"| CPML `mem_dzz` update | `{pct(groups['cpml_mem_dzz_update'])}` |",
            f"| z-cache shared loads | `{pct(groups['z_cache_shared_loads'])}` |",
            f"| visible address/branch control | `{pct(groups['address_branch_control_visible'])}` |",
            f"| other/unparsed | `{pct(groups['other_or_unparsed'])}` |",
            "",
            "Required source-group speedup to make packed pressure-len16 alone clear the 5% sampled-main gate:",
            "",
            "| group | required speedup |",
            "| --- | ---: |",
        ]
    )
    sub_req = derived["len16_source_subgroup_required_speedup"]
    for name, value in sub_req.items():
        lines.append(f"| `{name}` | `{ratio(value)}` |")

    lines.extend(
        [
            "",
            "## Scenario Ceilings",
            "",
            "| scenario | sampled-main speedup | interpretation |",
            "| --- | ---: | --- |",
        ]
    )
    interpretations = {
        "p_pml_total_1_10x": "barely below the 5% gate; pressure needs more than a small local win",
        "p_pml_total_1_25x": "would be meaningful, but requires real state/writeback reduction",
        "p_pml_len16_1_25x": "packed pressure-len16 alone is almost but not quite enough",
        "p_pml_residual_1_25x": "barely clears the gate in the model, but requires a large residual-only gain",
        "len16_final_update_2x": "mathematically interesting, but syntax variants already failed",
        "len16_mem_dzz_2x": "not enough by itself",
        "len16_final_plus_mem_dzz_1_5x": "large enough only for a broader state representation change",
    }
    for name, value in derived["pressure_scenario_sampled_speedups"].items():
        lines.append(f"| `{name}` | `{ratio(value)}` | {interpretations[name]} |")

    lines.extend(
        [
            "",
            "## Fusion Sanity Check",
            "",
            "| scenario | sampled-main speedup | decision |",
            "| --- | ---: | --- |",
        ]
    )
    fusion_decisions = {
        "perfect_remove_v_len16_time": "theoretical headroom, but direct z-face VP fusion is a rejected family",
        "perfect_remove_all_v_pml_time": "large upper bound, requires a materially new wave-step ownership model",
        "halve_all_v_pml_time": "large in theory, but far beyond what descriptor packing has modeled",
    }
    for name, value in derived["fusion_scenario_sampled_speedups"].items():
        lines.append(f"| `{name}` | `{ratio(value)}` | {fusion_decisions[name]} |")

    lines.extend(
        [
            "",
            "## Concrete Route Gate",
            "",
            "| route | decision | reason |",
            "| --- | --- | --- |",
        ]
    )
    for name, item in result["concrete_routes"].items():
        lines.append(f"| `{name}` | `{item['decision']}` | {item['reason']} |")

    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- reason: {gate['reason']}",
            f"- next action: {gate['next_action']}",
            "",
            "Allowed next:",
            "",
        ]
    )
    for item in gate["allowed_next"]:
        lines.append(f"- {item}")
    lines.extend(["", "Prohibited next:", ""])
    for item in gate["prohibited_next"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="reports/day_20260608/v_pml_len16_ncu_short_20260608_2315/summary.json",
    )
    parser.add_argument(
        "--source-hotlines",
        default="reports/day_20260608/len16_source_profile_20260608_1646/source_hotlines.json",
    )
    parser.add_argument("--target-speedup", type=float, default=1.05)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = root / summary_path
    hotlines_path = Path(args.source_hotlines)
    if not hotlines_path.is_absolute():
        hotlines_path = root / hotlines_path

    profile = load_latest_profile(summary_path)
    source = load_source_groups(hotlines_path)
    result = compute_result(profile, source, args.target_speedup)

    if args.json_out:
        json_out = Path(args.json_out)
        if not json_out.is_absolute():
            json_out = root / json_out
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if args.md_out:
        md_out = Path(args.md_out)
        if not md_out.is_absolute():
            md_out = root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_markdown(result), encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(render_markdown(result))


if __name__ == "__main__":
    main()
