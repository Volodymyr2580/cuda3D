#!/usr/bin/env python3
"""Gate pressure-PML writeback and CPML z-state ownership routes.

The accepted len16 half-warp pressure-PML kernel is no longer dominated by
inactive lanes or z-cache fill.  This model converts the current Nsight Compute
source hot lines into Amdahl ceilings and checks whether any concrete
writeback/state micro-route can justify a CUDA prototype.
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


def duration_us(kernels: dict[str, Any], name: str) -> float:
    item = kernels.get(name)
    if item is None:
        return 0.0
    return float(item["metrics"]["duration_ns"]) / 1000.0


def amdahl_speedup(local_speedup: float, share: float) -> float:
    if share <= 0.0:
        return 1.0
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


def line_share(hotlines: list[dict[str, Any]], *needles: str) -> float:
    share = 0.0
    for item in hotlines:
        text = str(item.get("text", ""))
        if any(needle in text for needle in needles):
            share += float(item["sample_share_of_parsed"])
    return share


def load_profile(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    profiles = {item["label"]: item for item in data["profiles"]}
    profile = profiles["len16"]
    kernels = profile["kernels"]
    p_core = duration_us(kernels, "cuda_fd3d_p_core_ns")
    v_pml = duration_us(kernels, "cuda_fd3d_v_pml_tile_ns")
    p_residual = duration_us(kernels, "cuda_fd3d_p_pml_tile_ns")
    p_len16 = duration_us(kernels, "cuda_fd3d_p_pml_len16_halfwarp_ns")
    sampled_main = p_core + v_pml + p_residual + p_len16
    return {
        "profile_label": "len16",
        "p_core_us": p_core,
        "v_pml_us": v_pml,
        "p_pml_residual_us": p_residual,
        "p_pml_len16_us": p_len16,
        "p_pml_total_us": p_residual + p_len16,
        "sampled_main_us": sampled_main,
        "p_pml_len16_sampled_main_share": p_len16 / sampled_main,
        "p_pml_total_sampled_main_share": (p_residual + p_len16) / sampled_main,
    }


def load_source_hotlines(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    hotlines = data["top_lines"]
    final_p0_cw2_share = line_share(
        hotlines,
        "p0[base]=2*__ldg(p1+base)-p0[base]",
        "+__ldg(cw2+base)*dt*(c1+c2+c3);",
    )
    mem_dzz_share = line_share(hotlines, "mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);")
    z_cache_share = line_share(hotlines, "vz_line_cache")
    branch_address_share = line_share(
        hotlines,
        "const PmlTile tile",
        "gtid1<npml",
        "} else if",
        "const size_t pind",
        "const size_t ts2",
        "const size_t base",
        "const int gtid1",
    )
    grouped = {
        "final_p0_p1_cw2_update": final_p0_cw2_share,
        "cpml_mem_dzz_update": mem_dzz_share,
        "z_cache_shared_loads": z_cache_share,
        "address_branch_control_visible": branch_address_share,
    }
    grouped["other_or_unparsed"] = max(0.0, 1.0 - sum(grouped.values()))
    return {
        "source_page": data.get("source_page"),
        "parsed_total_samples": data["parsed_total_samples"],
        "top_lines": hotlines,
        "grouped_sample_shares": grouped,
    }


def compute_model(profile: dict[str, Any], source: dict[str, Any], target_speedup: float) -> dict[str, Any]:
    packed_share = profile["p_pml_len16_sampled_main_share"]
    total_pressure_share = profile["p_pml_total_sampled_main_share"]
    grouped = source["grouped_sample_shares"]
    final_share = grouped["final_p0_p1_cw2_update"]
    mem_share = grouped["cpml_mem_dzz_update"]
    combined_share = final_share + mem_share
    packed_required = required_local_speedup(target_speedup, packed_share)
    pressure_required = required_local_speedup(target_speedup, total_pressure_share)

    def subgroup_case(local_share: float, subgroup_speedup: float) -> dict[str, Any]:
        packed_speedup = amdahl_speedup(subgroup_speedup, local_share)
        return {
            "local_share_in_len16_kernel": local_share,
            "subgroup_speedup": "infinite" if math.isinf(subgroup_speedup) else subgroup_speedup,
            "packed_kernel_speedup": packed_speedup,
            "sampled_main_speedup": amdahl_speedup(packed_speedup, packed_share),
        }

    ceilings = {
        "eliminate_final_p0_p1_cw2_update_utopian": subgroup_case(final_share, math.inf),
        "eliminate_cpml_mem_dzz_update_utopian": subgroup_case(mem_share, math.inf),
        "eliminate_both_final_and_mem_dzz_utopian": subgroup_case(combined_share, math.inf),
        "make_mem_dzz_2x_faster": subgroup_case(mem_share, 2.0),
        "make_final_update_2x_faster": subgroup_case(final_share, 2.0),
        "make_final_plus_mem_dzz_1_25x_faster": subgroup_case(combined_share, 1.25),
        "make_final_plus_mem_dzz_1_5x_faster": subgroup_case(combined_share, 1.5),
    }

    concrete_routes = {
        "p0_read_syntax_ldg": {
            "status": "rejected_by_existing_perf_repeat",
            "known_wp_speedup_vs_directfill": 1.000054,
            "reason": "Old-p0 read-only load was already tested on the pressure-PML path and landed at noise level.",
        },
        "cpml_local_new_mem_accumulation": {
            "status": "rejected_by_existing_perf_repeat",
            "known_wp_speedup_vs_directfill": 1.000647,
            "reason": "Explicit local new_mem expression did not change the memory dependency enough to matter.",
        },
        "ptxas_cache_policy_sweep": {
            "status": "rejected_by_existing_perf_repeat",
            "known_wp_speedup_vs_directfill": {
                "dlcm_ca": 0.999263,
                "dlcm_cg": 0.859344,
            },
            "reason": "Global load cache policy does not remove the final writeback or CPML state dependency.",
        },
        "branch_only_lower_upper_specialization": {
            "status": "reject_without_cuda",
            "required_packed_kernel_speedup_for_5pct_sampled_main": packed_required,
            "reason": "Visible branch/control source samples are too small and would add tile-list and launch overhead.",
        },
        "remove_old_p0_or_cw2_traffic": {
            "status": "requires_math_or_state_representation_redesign",
            "reason": (
                "Second-order pressure update needs old pressure state, current pressure state, velocity model, "
                "and a new pressure write.  A syntax rewrite only moves this traffic; it does not remove it."
            ),
        },
        "remove_mem_dzz_state_traffic": {
            "status": "requires_cpml_model_redesign_or_precision_relaxation",
            "reason": (
                "mem_dzz is a recursive CPML state.  It has one use per step and is already contiguous in the "
                "len16 z-line mapping, so shared/cache micro-tuning has no proven reuse target."
            ),
        },
    }

    final_needed = required_local_speedup(packed_required, final_share) if packed_required else None
    mem_needed = required_local_speedup(packed_required, mem_share) if packed_required else None
    combined_needed = required_local_speedup(packed_required, combined_share) if packed_required else None

    gate = {
        "decision": "reject_writeback_state_micro_cuda_prototype",
        "reason": (
            "The only groups large enough to matter are mathematically required pressure writeback traffic "
            "and recursive CPML z-state traffic.  Existing syntax/cache variants already show noise-level gains, "
            "while a >=5% sampled-main gain would require about "
            f"{ratio(packed_required)} speedup of the packed len16 kernel or "
            f"{ratio(combined_needed)} speedup of the combined final-writeback+mem_dzz group."
        ),
        "reopen_condition": (
            "Reopen only for a state-representation or time-integration design that proves the old-p0/cw2 or "
            "mem_dzz traffic is actually removed, not merely reloaded through a different syntax, and has a "
            ">=5% perf_1gpu_6shots repeat ceiling after extra storage/control costs."
        ),
        "next_allowed": [
            "math-level state representation design for pressure update, with equivalence proof before CUDA",
            "PML vx/vy round-trip ownership redesign that reduces global traffic without doubling component work",
            "formal same-session benchmark table for zmem/direct-fill/len16/current best before a larger phase switch",
        ],
        "prohibited": [
            "len16 p0 __ldg or old-p0 read syntax retry",
            "len16 explicit local new_mem retry",
            "ptxas cache-policy retry for this path",
            "branch-only lower/upper z-PML specialization without a new >=5% model",
            "shared z-cache or z-cache fill micro-tuning inside the accepted len16 kernel",
        ],
    }

    return {
        "inputs": {
            "profile": profile,
            "source": source,
            "target_speedup": target_speedup,
        },
        "derived": {
            "packed_len16_kernel_share_of_sampled_main": packed_share,
            "total_pressure_pml_share_of_sampled_main": total_pressure_share,
            "packed_kernel_speedup_required_for_5pct_sampled_main": packed_required,
            "total_pressure_speedup_required_for_5pct_sampled_main": pressure_required,
            "final_group_speedup_required_if_alone": final_needed,
            "mem_dzz_group_speedup_required_if_alone": mem_needed,
            "final_plus_mem_dzz_group_speedup_required": combined_needed,
        },
        "ceilings": ceilings,
        "concrete_routes": concrete_routes,
        "gate": gate,
    }


def render_markdown(result: dict[str, Any]) -> str:
    profile = result["inputs"]["profile"]
    source = result["inputs"]["source"]
    grouped = source["grouped_sample_shares"]
    derived = result["derived"]
    gate = result["gate"]
    lines = [
        "# Pressure-PML Writeback / CPML State Gate",
        "",
        "## Context",
        "",
        f"- profile: `{profile['profile_label']}`",
        f"- sampled main: `{profile['sampled_main_us']:.3f}us`",
        f"- len16 packed pressure-PML: `{profile['p_pml_len16_us']:.3f}us` / `{pct(profile['p_pml_len16_sampled_main_share'])}`",
        f"- total pressure-PML: `{profile['p_pml_total_us']:.3f}us` / `{pct(profile['p_pml_total_sampled_main_share'])}`",
        f"- target sampled-main speedup gate: `{ratio(result['inputs']['target_speedup'])}`",
        f"- packed len16 kernel speedup required for target: `{ratio(derived['packed_kernel_speedup_required_for_5pct_sampled_main'])}`",
        f"- total pressure-PML speedup required for target: `{ratio(derived['total_pressure_speedup_required_for_5pct_sampled_main'])}`",
        "",
        "## Source Sample Groups",
        "",
        f"Parsed source samples: `{source['parsed_total_samples']}`",
        "",
        "| group | share of len16 source samples | interpretation |",
        "| --- | ---: | --- |",
        f"| final `p0/p1/cw2` update | `{pct(grouped['final_p0_p1_cw2_update'])}` | required second-order pressure writeback and model load |",
        f"| CPML `mem_dzz` update | `{pct(grouped['cpml_mem_dzz_update'])}` | recursive z-PML state update |",
        f"| z-cache shared loads | `{pct(grouped['z_cache_shared_loads'])}` | visible but no longer dominant after direct-fill cache |",
        f"| address/control visible lines | `{pct(grouped['address_branch_control_visible'])}` | tile/address/branch overhead visible in source page |",
        f"| other/unparsed | `{pct(grouped['other_or_unparsed'])}` | all remaining sampled lines |",
        "",
        "Required local speedups if only one group is improved:",
        "",
        "| group | local speedup required | note |",
        "| --- | ---: | --- |",
        f"| final `p0/p1/cw2` group alone | `{ratio(derived['final_group_speedup_required_if_alone'])}` | would require removing a large part of the time-update traffic |",
        f"| CPML `mem_dzz` group alone | `{ratio(derived['mem_dzz_group_speedup_required_if_alone'])}` | effectively requires eliminating most recursive state traffic |",
        f"| final + `mem_dzz` together | `{ratio(derived['final_plus_mem_dzz_group_speedup_required'])}` | only plausible through a broader state representation change |",
        "",
        "## Amdahl Ceilings",
        "",
        "| scenario | packed-kernel speedup | sampled-main speedup | status |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, item in result["ceilings"].items():
        status = "utopian" if "eliminate" in name else "modeled"
        lines.append(
            f"| `{name}` | `{ratio(item['packed_kernel_speedup'])}` | "
            f"`{ratio(item['sampled_main_speedup'])}` | {status} |"
        )

    lines.extend(
        [
            "",
            "## Concrete Route Check",
            "",
            "| route | status | known/measured signal | reason |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for name, item in result["concrete_routes"].items():
        signal = item.get("known_wp_speedup_vs_directfill")
        if isinstance(signal, dict):
            signal_text = ", ".join(f"{key}={ratio(value)}" for key, value in signal.items())
        elif signal is None:
            signal_text = "n/a"
        else:
            signal_text = ratio(float(signal))
        lines.append(f"| `{name}` | `{item['status']}` | {signal_text} | {item['reason']} |")

    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- reason: {gate['reason']}",
            f"- reopen condition: {gate['reopen_condition']}",
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
    parser.add_argument("--summary", default="reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json")
    parser.add_argument("--source-hotlines", default="reports/day_20260608/len16_source_profile_20260608_1646/source_hotlines.json")
    parser.add_argument("--target-speedup", type=float, default=1.05)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    summary = Path(args.summary)
    if not summary.is_absolute():
        summary = root / summary
    hotlines = Path(args.source_hotlines)
    if not hotlines.is_absolute():
        hotlines = root / hotlines

    profile = load_profile(summary)
    source = load_source_hotlines(hotlines)
    result = compute_model(profile, source, args.target_speedup)

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
