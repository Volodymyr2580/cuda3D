#!/usr/bin/env python3
"""Gate PML len16 compact-state ownership for CUDA3D.

The model is intentionally conservative.  It uses accepted len16 tile counts
from real run logs as the ownership anchor, then estimates whether replacing
full CPML state indexing with compact per-line state can plausibly clear the
prototype gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


FLOAT_BYTES = 4
PML_TILE_Z = 32
PML_TILE_X = 4
PML_TILE_Y = 2
LEN16_ACTIVE_Z = 16
POINTS_PER_LEN16_TILE = PML_TILE_X * PML_TILE_Y * LEN16_ACTIVE_Z
CORE_PML_MARGIN = 4


@dataclass(frozen=True)
class ShotTileCounts:
    shot_index: int
    v_residual_tiles: int
    v_len16_tiles: int
    p_residual_tiles: int
    p_len16_tiles: int


@dataclass(frozen=True)
class KernelMetric:
    kernel: str
    duration_us: float | None
    dram_throughput_pct: float | None
    memory_throughput_tb_s: float | None
    l1_hit_rate_pct: float | None
    l2_hit_rate_pct: float | None


def parse_case_manifest(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_input_npml(path: Path, ny: int, nx: int, nz: int) -> int:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    for i in range(0, max(0, len(lines) - 10)):
        try:
            if int(float(lines[i])) != ny:
                continue
            if int(float(lines[i + 1])) == nx and int(float(lines[i + 2])) == nz:
                return int(float(lines[i + 6]))
        except ValueError:
            continue
    raise ValueError(f"could not parse npml from {path}")


def parse_tile_counts(log_path: Path) -> list[ShotTileCounts]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    pending_v: tuple[int, int] | None = None
    shots: list[ShotTileCounts] = []
    v_re = re.compile(
        r"PML velocity len16 halfwarp enabled: len16_tiles=(\d+) residual_v_tiles=(\d+)"
    )
    p_re = re.compile(
        r"PML pressure len16 halfwarp enabled: len16_tiles=(\d+) residual_p_tiles=(\d+)"
    )
    for line in lines:
        vm = v_re.search(line)
        if vm:
            pending_v = (int(vm.group(2)), int(vm.group(1)))
            continue
        pm = p_re.search(line)
        if pm and pending_v is not None:
            shots.append(
                ShotTileCounts(
                    shot_index=len(shots),
                    v_residual_tiles=pending_v[0],
                    v_len16_tiles=pending_v[1],
                    p_residual_tiles=int(pm.group(2)),
                    p_len16_tiles=int(pm.group(1)),
                )
            )
            pending_v = None
    return shots


def metric_float(value: str) -> float | None:
    value = value.strip()
    if not value or value == "no data":
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def parse_ncu_csv(path: Path) -> dict[str, KernelMetric]:
    by_kernel: dict[str, dict[str, float | None]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            kernel = row["Kernel Name"].split("(")[0]
            metric = row["Metric Name"]
            value = metric_float(row["Metric Value"])
            item = by_kernel.setdefault(kernel, {})
            if metric == "Duration":
                item["duration_us"] = value
            elif metric == "DRAM Throughput":
                item["dram_throughput_pct"] = value
            elif metric == "Memory Throughput" and row["Metric Unit"] == "Tbyte/s":
                item["memory_throughput_tb_s"] = value
            elif metric == "L1/TEX Hit Rate":
                item["l1_hit_rate_pct"] = value
            elif metric == "L2 Hit Rate":
                item["l2_hit_rate_pct"] = value
    return {
        kernel: KernelMetric(
            kernel=kernel,
            duration_us=metrics.get("duration_us"),
            dram_throughput_pct=metrics.get("dram_throughput_pct"),
            memory_throughput_tb_s=metrics.get("memory_throughput_tb_s"),
            l1_hit_rate_pct=metrics.get("l1_hit_rate_pct"),
            l2_hit_rate_pct=metrics.get("l2_hit_rate_pct"),
        )
        for kernel, metrics in by_kernel.items()
    }


def state_sizes(ny: int, nx: int, nz: int, npml: int) -> dict[str, dict[str, int | str]]:
    n3 = ny + 2 * npml
    n2 = nx + 2 * npml
    n1 = nz + 2 * npml
    mem_y = 2 * npml * n2 * n1
    mem_x = n3 * 2 * npml * n1
    mem_z = n3 * n2 * 2 * npml
    arrays = {
        "memory_dy": ("2*npml*n2*n1", mem_y),
        "memory_dx": ("n3*2*npml*n1", mem_x),
        "memory_dz": ("n3*n2*2*npml", mem_z),
        "memory_dy_next": ("2*npml*n2*n1", mem_y),
        "memory_dx_next": ("n3*2*npml*n1", mem_x),
        "memory_dz_next": ("n3*n2*2*npml", mem_z),
        "memory_dyy": ("2*npml*n2*n1", mem_y),
        "memory_dxx": ("n3*2*npml*n1", mem_x),
        "memory_dzz": ("n3*n2*2*npml", mem_z),
    }
    return {
        name: {"formula": formula, "elements": elems, "bytes": elems * FLOAT_BYTES}
        for name, (formula, elems) in arrays.items()
    }


def mib(value: float) -> float:
    return value / (1024.0 * 1024.0)


def build_result(args: argparse.Namespace) -> dict:
    case_dir = Path(args.case_dir)
    manifest = parse_case_manifest(case_dir / "case_manifest.txt")
    ny = int(manifest["ny"])
    nx = int(manifest["nx"])
    nz = int(manifest["nz"])
    nt = int(manifest.get("nt", 0))
    shots = int(manifest.get("shots", 0))
    input_name = manifest["input"]
    npml = parse_input_npml(case_dir / input_name, ny, nx, nz)

    tile_counts = parse_tile_counts(Path(args.perf_log))
    if not tile_counts:
        raise ValueError(f"no len16 tile-count lines found in {args.perf_log}")
    ncu = parse_ncu_csv(Path(args.ncu_csv)) if args.ncu_csv else {}

    arrays = state_sizes(ny, nx, nz, npml)
    len16_p_points = sum(s.p_len16_tiles for s in tile_counts) * POINTS_PER_LEN16_TILE
    len16_v_points = sum(s.v_len16_tiles for s in tile_counts) * POINTS_PER_LEN16_TILE
    residual_p_points = sum(s.p_residual_tiles for s in tile_counts) * PML_TILE_Z * PML_TILE_X * PML_TILE_Y
    residual_v_points = sum(s.v_residual_tiles for s in tile_counts) * PML_TILE_Z * PML_TILE_X * PML_TILE_Y

    # Exact compact-state candidates for the accepted len16 path.
    #
    # pressure len16:
    # - memory_dzz update at central active z only.
    # - z-recompute cache reads memory_dz/memory_dz_next around a 16-point line.
    #   It touches up to 23 z positions per line.  For compact ownership, this
    #   needs either a 23-slot read window or a 16-slot central state plus a
    #   halo rule.  Use 23 slots for a conservative descriptor/storage budget.
    #
    # velocity len16:
    # - current accepted kernel only writes vx/vy derivative arrays, not CPML
    #   memory_dx/memory_dy/memory_dz state.
    p_len16_lines = sum(s.p_len16_tiles for s in tile_counts) * PML_TILE_X * PML_TILE_Y
    compact_pressure_state_elements = p_len16_lines * (LEN16_ACTIVE_Z + 23 + 23)
    compact_pressure_state_bytes = compact_pressure_state_elements * FLOAT_BYTES

    full_pressure_related_state_bytes = (
        arrays["memory_dzz"]["bytes"]
        + arrays["memory_dz"]["bytes"]
        + arrays["memory_dz_next"]["bytes"]
    ) * max(1, len(tile_counts))
    compact_to_full_pressure_state_ratio = (
        compact_pressure_state_bytes / full_pressure_related_state_bytes
        if full_pressure_related_state_bytes
        else 0.0
    )

    p_len16_kernel = ncu.get("cuda_fd3d_p_pml_len16_halfwarp_ns")
    p_residual_kernel = ncu.get("cuda_fd3d_p_pml_tile_ns")
    v_len16_kernel = ncu.get("cuda_fd3d_v_pml_len16_halfwarp_ns")
    v_residual_kernel = ncu.get("cuda_fd3d_v_pml_tile_ns")
    p_len16_us = p_len16_kernel.duration_us if p_len16_kernel else None
    p_residual_us = p_residual_kernel.duration_us if p_residual_kernel else None
    v_len16_us = v_len16_kernel.duration_us if v_len16_kernel else None
    v_residual_us = v_residual_kernel.duration_us if v_residual_kernel else None

    main_kernel_us = sum(
        v
        for v in [p_len16_us, p_residual_us, v_len16_us, v_residual_us, args.p_core_us]
        if isinstance(v, (int, float))
    )
    p_len16_share = (p_len16_us / main_kernel_us) if p_len16_us and main_kernel_us else 0.0

    # Conservative speed ceiling: only p_len16 benefits, and only the fraction
    # of p_len16 time plausibly tied to CPML state traffic can be removed.
    # Default 0.35 is intentionally optimistic enough to let a real opportunity
    # pass the gate but not enough to rubber-stamp tiny memory-layout changes.
    removable_fraction = args.p_len16_state_fraction
    p_len16_speedup_ceiling = 1.0 / max(1e-9, 1.0 - removable_fraction)
    whole_job_speedup_ceiling = 1.0 / (
        1.0 - p_len16_share + p_len16_share / p_len16_speedup_ceiling
    )

    if whole_job_speedup_ceiling < 1.03:
        decision = "stop_implementation"
    elif whole_job_speedup_ceiling < 1.05:
        decision = "mirror_only"
    else:
        decision = "allow_commit_prototype_after_design"

    return {
        "case": {
            "case_dir": str(case_dir),
            "ny": ny,
            "nx": nx,
            "nz": nz,
            "nt": nt,
            "shots": shots,
            "npml": npml,
            "input": input_name,
        },
        "state_arrays": arrays,
        "tile_counts": [asdict(s) for s in tile_counts],
        "aggregate": {
            "shots_seen_in_log": len(tile_counts),
            "p_len16_tiles": sum(s.p_len16_tiles for s in tile_counts),
            "p_residual_tiles": sum(s.p_residual_tiles for s in tile_counts),
            "v_len16_tiles": sum(s.v_len16_tiles for s in tile_counts),
            "v_residual_tiles": sum(s.v_residual_tiles for s in tile_counts),
            "p_len16_active_points": len16_p_points,
            "v_len16_active_points": len16_v_points,
            "p_residual_launched_lane_points": residual_p_points,
            "v_residual_launched_lane_points": residual_v_points,
            "p_len16_lines": p_len16_lines,
            "compact_pressure_state_elements": compact_pressure_state_elements,
            "compact_pressure_state_bytes": compact_pressure_state_bytes,
            "compact_pressure_state_mib": mib(compact_pressure_state_bytes),
            "full_pressure_related_state_bytes_x_shots": full_pressure_related_state_bytes,
            "full_pressure_related_state_mib_x_shots": mib(full_pressure_related_state_bytes),
            "compact_to_full_pressure_state_ratio": compact_to_full_pressure_state_ratio,
        },
        "ncu_kernels": {name: asdict(metric) for name, metric in ncu.items()},
        "model": {
            "p_core_us_assumed": args.p_core_us,
            "sampled_main_kernel_us": main_kernel_us,
            "p_len16_us": p_len16_us,
            "p_residual_us": p_residual_us,
            "v_len16_us": v_len16_us,
            "v_residual_us": v_residual_us,
            "p_len16_share_of_sampled_main": p_len16_share,
            "p_len16_state_fraction_assumed": removable_fraction,
            "p_len16_speedup_ceiling": p_len16_speedup_ceiling,
            "whole_job_speedup_ceiling": whole_job_speedup_ceiling,
        },
        "decision": decision,
    }


def render_markdown(result: dict) -> str:
    case = result["case"]
    agg = result["aggregate"]
    model = result["model"]
    decision = result["decision"]
    arrays = result["state_arrays"]
    lines = [
        "# PML Len16 Compact-State Traffic Audit",
        "",
        "## Case",
        "",
        f"- case_dir: `{case['case_dir']}`",
        f"- logical grid: `{case['ny']} x {case['nx']} x {case['nz']}`",
        f"- nt: `{case['nt']}`",
        f"- shots in manifest/log: `{case['shots']}` / `{agg['shots_seen_in_log']}`",
        f"- npml: `{case['npml']}`",
        "",
        "## Full CPML State Arrays",
        "",
        "| array | formula | elements | MiB |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, item in arrays.items():
        lines.append(
            f"| `{name}` | `{item['formula']}` | `{item['elements']}` | `{mib(item['bytes']):.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Len16 Ownership From Runtime Log",
            "",
            f"- pressure len16 tiles: `{agg['p_len16_tiles']}`",
            f"- pressure residual tiles: `{agg['p_residual_tiles']}`",
            f"- velocity len16 tiles: `{agg['v_len16_tiles']}`",
            f"- velocity residual tiles: `{agg['v_residual_tiles']}`",
            f"- pressure len16 active points: `{agg['p_len16_active_points']}`",
            f"- velocity len16 active points: `{agg['v_len16_active_points']}`",
            f"- pressure len16 compact lines: `{agg['p_len16_lines']}`",
            "",
            "## Compact Pressure-State Budget",
            "",
            "The accepted velocity len16 kernel currently writes `vx/vy` derivative",
            "fields and does not update `memory_dx/memory_dy/memory_dz` state.",
            "The compact-state opportunity is therefore anchored on pressure",
            "len16 state: `memory_dzz` plus the z-recompute `memory_dz` old/next",
            "window.",
            "",
            f"- compact pressure-state bytes: `{agg['compact_pressure_state_mib']:.3f} MiB`",
            f"- full pressure-related state bytes x shots: `{agg['full_pressure_related_state_mib_x_shots']:.3f} MiB`",
            f"- compact/full ratio: `{agg['compact_to_full_pressure_state_ratio']:.6f}`",
            "",
            "## NCU-Anchored Ceiling",
            "",
            f"- sampled main kernel us: `{model['sampled_main_kernel_us']:.3f}`",
            f"- p_len16 us: `{model['p_len16_us']}`",
            f"- p_len16 share of sampled main: `{model['p_len16_share_of_sampled_main']:.4f}`",
            f"- assumed removable p_len16 state fraction: `{model['p_len16_state_fraction_assumed']:.2f}`",
            f"- p_len16 speedup ceiling: `{model['p_len16_speedup_ceiling']:.4f}x`",
            f"- whole-job sampled-main speedup ceiling: `{model['whole_job_speedup_ceiling']:.4f}x`",
            "",
            "## Gate Decision",
            "",
            f"- decision: `{decision}`",
            "",
        ]
    )
    if decision == "stop_implementation":
        lines.append("Estimated ceiling is below `3%`; do not implement compact-state CUDA.")
    elif decision == "mirror_only":
        lines.append("Estimated ceiling is `3%..5%`; debug mirror is allowed, commit path is not.")
    else:
        lines.append("Estimated ceiling is at least `5%`; design and mirror must still pass before commit path.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--perf-log", required=True)
    parser.add_argument("--ncu-csv", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    parser.add_argument("--p-core-us", type=float, default=75.0)
    parser.add_argument("--p-len16-state-fraction", type=float, default=0.35)
    args = parser.parse_args()

    result = build_result(args)
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "model": result["model"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
