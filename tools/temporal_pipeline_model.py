#!/usr/bin/env python3
"""Byte and synchronization model for CUDA3D K-step temporal pipelines."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FLOAT_BYTES = 4


@dataclass(frozen=True)
class Geometry:
    case_dir: str
    ny: int
    nx: int
    nz: int
    nt: int
    shots: int
    receivers_per_shot: int
    npml: int
    radius: int
    core_pml_margin: int
    n3: int
    n2: int
    n1: int
    core_z: int
    core_x: int
    core_y: int
    core_points: int
    k2_deep_z: int
    k2_deep_x: int
    k2_deep_y: int
    k2_deep_points: int
    k2_deep_share: float


@dataclass(frozen=True)
class CoreByteModel:
    pblock_z: int
    pblock_x: int
    pblock_y: int
    p1_global_floats_per_output: float
    p0_cw2_store_floats_per_output: float
    bytes_per_output_current: float
    bytes_per_core_step_current: float
    bytes_per_two_core_steps_current: float
    ideal_k2_saved_bytes_per_deep_output: float
    ideal_k2_saved_bytes_per_pair: float
    ideal_k2_p_core_pair_reduction: float
    ideal_k2_sampled_main_speedup: float


@dataclass(frozen=True)
class CtaCandidate:
    name: str
    out_z: int
    out_x: int
    out_y: int
    final_outputs: int
    p_mid_z: int
    p_mid_x: int
    p_mid_y: int
    p_mid_elements: int
    p_mid_shared_bytes: int
    p_mid_elements_per_final_output: float
    local_pair_bytes_per_final_output: float
    baseline_pair_bytes_per_final_output: float
    local_pair_byte_ratio_vs_baseline: float
    fits_99kb: bool
    verdict: str


def parse_kv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _as_int(text: str) -> int | None:
    try:
        return int(float(text.strip()))
    except ValueError:
        return None


def parse_input_npml(path: Path, ny: int, nx: int, nz: int) -> int:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    for i in range(0, max(0, len(lines) - 10)):
        if _as_int(lines[i]) != ny:
            continue
        if _as_int(lines[i + 1]) == nx and _as_int(lines[i + 2]) == nz:
            npml = _as_int(lines[i + 6])
            if npml is not None:
                return npml
    raise ValueError(f"could not parse npml from {path}")


def load_geometry(case_dir: Path, radius: int, core_pml_margin: int) -> Geometry:
    manifest = parse_kv(case_dir / "case_manifest.txt")
    ny = int(manifest["ny"])
    nx = int(manifest["nx"])
    nz = int(manifest["nz"])
    nt = int(manifest.get("nt", 0))
    shots = int(manifest.get("shots", 0))
    receivers = int(manifest.get("receivers_per_shot", 0))
    input_name = manifest.get("input")
    if not input_name:
        raise ValueError(f"{case_dir}/case_manifest.txt does not define input=")
    npml = parse_input_npml(case_dir / input_name, ny, nx, nz)
    n3 = ny + 2 * npml
    n2 = nx + 2 * npml
    n1 = nz + 2 * npml
    core_z = max(0, n1 - 2 * (npml + core_pml_margin))
    core_x = max(0, n2 - 2 * (npml + core_pml_margin))
    core_y = max(0, n3 - 2 * (npml + core_pml_margin))
    core_points = core_z * core_x * core_y
    k2_deep_z = max(0, core_z - 2 * radius)
    k2_deep_x = max(0, core_x - 2 * radius)
    k2_deep_y = max(0, core_y - 2 * radius)
    k2_deep_points = k2_deep_z * k2_deep_x * k2_deep_y
    return Geometry(
        case_dir=str(case_dir),
        ny=ny,
        nx=nx,
        nz=nz,
        nt=nt,
        shots=shots,
        receivers_per_shot=receivers,
        npml=npml,
        radius=radius,
        core_pml_margin=core_pml_margin,
        n3=n3,
        n2=n2,
        n1=n1,
        core_z=core_z,
        core_x=core_x,
        core_y=core_y,
        core_points=core_points,
        k2_deep_z=k2_deep_z,
        k2_deep_x=k2_deep_x,
        k2_deep_y=k2_deep_y,
        k2_deep_points=k2_deep_points,
        k2_deep_share=k2_deep_points / core_points if core_points else 0.0,
    )


def load_phase4_summary(root: Path, path: str | None) -> dict[str, Any]:
    if path is None:
        path = "reports/day_20260608/phase4_global_temporal_pipeline_design_summary.json"
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return json.loads(p.read_text(encoding="utf-8"))


def core_byte_model(
    geometry: Geometry,
    pblock_z: int,
    pblock_x: int,
    pblock_y: int,
    phase4: dict[str, Any],
) -> CoreByteModel:
    # Current p_core shares only the z-axis pressure loads in shared memory.
    z_halo_loads_per_output = (2 * geometry.radius) / pblock_z
    p1_global = 1.0 + z_halo_loads_per_output + 4 * geometry.radius
    p0_cw2_store = 3.0
    bytes_per_output = (p1_global + p0_cw2_store) * FLOAT_BYTES
    bytes_per_step = bytes_per_output * geometry.core_points
    bytes_per_two_steps = 2.0 * bytes_per_step

    # An impossible-but-useful upper bound: the second step over the K=2
    # deep core reuses all p(t+1) stencil values locally, so it avoids the
    # global p1 stencil loads for those outputs but still reads p(t), cw2,
    # and writes p(t+2).
    saved_per_deep = p1_global * FLOAT_BYTES
    saved_pair = saved_per_deep * geometry.k2_deep_points
    p_core_reduction = saved_pair / bytes_per_two_steps if bytes_per_two_steps else 0.0
    p_core_share = float(phase4["sampled_share"]["cuda_fd3d_p_core_ns"])
    sampled_speedup = 1.0 / (1.0 - p_core_share * p_core_reduction)

    return CoreByteModel(
        pblock_z=pblock_z,
        pblock_x=pblock_x,
        pblock_y=pblock_y,
        p1_global_floats_per_output=p1_global,
        p0_cw2_store_floats_per_output=p0_cw2_store,
        bytes_per_output_current=bytes_per_output,
        bytes_per_core_step_current=bytes_per_step,
        bytes_per_two_core_steps_current=bytes_per_two_steps,
        ideal_k2_saved_bytes_per_deep_output=saved_per_deep,
        ideal_k2_saved_bytes_per_pair=saved_pair,
        ideal_k2_p_core_pair_reduction=p_core_reduction,
        ideal_k2_sampled_main_speedup=sampled_speedup,
    )


def cta_candidates(radius: int, max_smem: int, bytes_model: CoreByteModel) -> list[CtaCandidate]:
    shapes = [
        ("T1", 32, 4, 4),
        ("T2", 16, 8, 4),
        ("T3", 16, 4, 8),
        ("T4", 32, 8, 2),
        ("T5", 64, 4, 2),
        ("T6", 24, 6, 4),
    ]
    out: list[CtaCandidate] = []
    for name, z, x, y in shapes:
        mid_z = z + 2 * radius
        mid_x = x + 2 * radius
        mid_y = y + 2 * radius
        mid = mid_z * mid_x * mid_y
        final = z * x * y
        bytes_ = mid * FLOAT_BYTES
        p_mid_compute_bytes = (bytes_model.p1_global_floats_per_output + 2.0) * FLOAT_BYTES
        second_step_bytes = bytes_model.p0_cw2_store_floats_per_output * FLOAT_BYTES
        local_pair_bytes = (mid / final) * p_mid_compute_bytes + second_step_bytes
        baseline_pair_bytes = 2.0 * bytes_model.bytes_per_output_current
        local_pair_ratio = local_pair_bytes / baseline_pair_bytes
        if bytes_ > max_smem:
            verdict = "fail_smem"
        elif local_pair_ratio > 1.0:
            verdict = "fail_halo_duplication"
        else:
            verdict = "fits_but_cta_local_forbidden"
        out.append(
            CtaCandidate(
                name=name,
                out_z=z,
                out_x=x,
                out_y=y,
                final_outputs=final,
                p_mid_z=mid_z,
                p_mid_x=mid_x,
                p_mid_y=mid_y,
                p_mid_elements=mid,
                p_mid_shared_bytes=bytes_,
                p_mid_elements_per_final_output=mid / final,
                local_pair_bytes_per_final_output=local_pair_bytes,
                baseline_pair_bytes_per_final_output=baseline_pair_bytes,
                local_pair_byte_ratio_vs_baseline=local_pair_ratio,
                fits_99kb=bytes_ <= max_smem,
                verdict=verdict,
            )
        )
    return out


def cooperative_gate(geometry: Geometry, pblock_z: int, pblock_x: int, pblock_y: int, sm_count: int, active_blocks_per_sm: int) -> dict[str, Any]:
    grid_z = math.ceil(geometry.core_z / pblock_z)
    grid_x = math.ceil(geometry.core_x / pblock_x)
    grid_y = math.ceil(geometry.core_y / pblock_y)
    blocks = grid_z * grid_x * grid_y
    resident = sm_count * active_blocks_per_sm
    return {
        "grid_blocks": blocks,
        "grid_zxy": [grid_z, grid_x, grid_y],
        "assumed_sm_count": sm_count,
        "assumed_active_blocks_per_sm": active_blocks_per_sm,
        "resident_block_capacity": resident,
        "fits_cooperative_grid": blocks <= resident,
        "over_capacity_factor": blocks / resident if resident else math.inf,
    }


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case)
    if not case_dir.is_absolute():
        case_dir = root / case_dir
    geometry = load_geometry(case_dir, args.radius, args.core_pml_margin)
    phase4 = load_phase4_summary(root, args.phase4_summary)
    bytes_model = core_byte_model(geometry, args.pblock_z, args.pblock_x, args.pblock_y, phase4)
    coop = cooperative_gate(geometry, args.pblock_z, args.pblock_x, args.pblock_y, args.sm_count, args.active_blocks_per_sm)
    ctas = cta_candidates(args.radius, args.max_block_smem, bytes_model)

    global_mid_speedup = 1.0
    verdict = "stop_cuda_prototype"
    reasons = [
        "A global-middle K=2 design is safe but does not remove the p(t+1) global stencil traffic, so it has no meaningful byte saving.",
        "A cooperative grid-wide sync design cannot launch the full p_core grid resident at once under conservative RTX 5090 assumptions.",
        "The no-duplication ideal has >5% sampled-main upside, but concrete CTA-local p_mid candidates become slower after halo duplication.",
        "CTA-local p_mid reuse is also the previously rejected/forbidden two-step family unless redesigned as a source-aware swept/wavefront ownership algorithm.",
    ]
    if bytes_model.ideal_k2_sampled_main_speedup >= args.meaningful_speedup:
        ideal_read = "ideal_cta_local_upper_bound_passes_but_is_forbidden"
    else:
        ideal_read = "even_ideal_upper_bound_fails"

    return {
        "geometry": asdict(geometry),
        "phase4_input": phase4,
        "core_byte_model": asdict(bytes_model),
        "cooperative_grid_gate": coop,
        "cta_local_candidates": [asdict(item) for item in ctas],
        "safe_global_middle": {
            "sampled_main_speedup_estimate": global_mid_speedup,
            "verdict": "fails_meaningful_gate",
            "reason": "Using global p(t+1) plus a synchronization point preserves correctness but largely reproduces two normal p_core passes.",
        },
        "ideal_upper_bound": {
            "read": ideal_read,
            "sampled_main_speedup": bytes_model.ideal_k2_sampled_main_speedup,
            "p_core_pair_reduction": bytes_model.ideal_k2_p_core_pair_reduction,
        },
        "gate": {
            "verdict": verdict,
            "meaningful_speedup_required": args.meaningful_speedup,
            "reasons": reasons,
            "next_allowed_work": "source-aware swept/wavefront temporal design or explicit Pro-approved CTA-local temporal research, not direct CUDA prototype",
        },
    }


def mib(value: float) -> float:
    return value / (1024.0 * 1024.0)


def render_markdown(result: dict[str, Any]) -> str:
    g = result["geometry"]
    b = result["core_byte_model"]
    coop = result["cooperative_grid_gate"]
    gate = result["gate"]
    lines = [
        "# Temporal Pipeline Byte/Synchronization Model",
        "",
        "## Case Geometry",
        "",
        f"- case_dir: `{g['case_dir']}`",
        f"- logical ny/nx/nz: `{g['ny']}/{g['nx']}/{g['nz']}`",
        f"- domain n3/n2/n1: `{g['n3']}/{g['n2']}/{g['n1']}`",
        f"- npml/radius/CorePmlMargin: `{g['npml']}/{g['radius']}/{g['core_pml_margin']}`",
        f"- pressure core z/x/y: `{g['core_z']}/{g['core_x']}/{g['core_y']}`",
        f"- pressure core points: `{g['core_points']}`",
        f"- K=2 deep core z/x/y: `{g['k2_deep_z']}/{g['k2_deep_x']}/{g['k2_deep_y']}`",
        f"- K=2 deep core share: `{g['k2_deep_share']:.2%}`",
        "",
        "## Current P-Core Byte Model",
        "",
        f"- p_core block z/x/y: `{b['pblock_z']}/{b['pblock_x']}/{b['pblock_y']}`",
        f"- estimated p1 global floats/output: `{b['p1_global_floats_per_output']:.6f}`",
        f"- p0/cw2/store floats/output: `{b['p0_cw2_store_floats_per_output']:.6f}`",
        f"- estimated current bytes/output: `{b['bytes_per_output_current']:.3f}`",
        f"- estimated current bytes/core step: `{mib(b['bytes_per_core_step_current']):.3f} MiB`",
        f"- estimated current bytes/two core steps: `{mib(b['bytes_per_two_core_steps_current']):.3f} MiB`",
        "",
        "## Ideal K=2 Upper Bound",
        "",
        "This is an impossible upper bound unless the second-step `p(t+1)` stencil is reused locally without unsafe CTA-boundary reads.",
        "",
        f"- saved bytes/deep output: `{b['ideal_k2_saved_bytes_per_deep_output']:.3f}`",
        f"- saved bytes/pair: `{mib(b['ideal_k2_saved_bytes_per_pair']):.3f} MiB`",
        f"- p_core pair reduction upper bound: `{b['ideal_k2_p_core_pair_reduction']:.2%}`",
        f"- sampled-main speedup upper bound: `{b['ideal_k2_sampled_main_speedup']:.3f}x`",
        "",
        "## Cooperative Grid Gate",
        "",
        f"- p_core grid blocks: `{coop['grid_blocks']}` with grid z/x/y `{coop['grid_zxy']}`",
        f"- assumed resident block capacity: `{coop['resident_block_capacity']}`",
        f"- over capacity factor: `{coop['over_capacity_factor']:.2f}x`",
        f"- fits cooperative grid: `{coop['fits_cooperative_grid']}`",
        "",
        "## CTA-Local P-Mid Candidates",
        "",
        "| name | output z/x/y | p_mid z/x/y | shared KiB | p_mid/output | local pair bytes / baseline | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in result["cta_local_candidates"]:
        lines.append(
            f"| {item['name']} | {item['out_z']}x{item['out_x']}x{item['out_y']} | "
            f"{item['p_mid_z']}x{item['p_mid_x']}x{item['p_mid_y']} | "
            f"{item['p_mid_shared_bytes'] / 1024.0:.1f} | "
            f"{item['p_mid_elements_per_final_output']:.2f} | "
            f"{item['local_pair_byte_ratio_vs_baseline']:.2f}x | {item['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- meaningful speedup required: `{gate['meaningful_speedup_required']:.3f}x`",
            "",
            "Reasons:",
        ]
    )
    for reason in gate["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            f"Next allowed work: {gate['next_allowed_work']}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="benchmarks/cases/perf_1gpu_6shots")
    parser.add_argument("--phase4-summary")
    parser.add_argument("--radius", type=int, default=7)
    parser.add_argument("--core-pml-margin", type=int, default=4)
    parser.add_argument("--pblock-z", type=int, default=128)
    parser.add_argument("--pblock-x", type=int, default=2)
    parser.add_argument("--pblock-y", type=int, default=1)
    parser.add_argument("--sm-count", type=int, default=170)
    parser.add_argument("--active-blocks-per-sm", type=int, default=8)
    parser.add_argument("--max-block-smem", type=int, default=99_000)
    parser.add_argument("--meaningful-speedup", type=float, default=1.05)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    result = build_result(args)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_markdown(result)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(md)


if __name__ == "__main__":
    main()
