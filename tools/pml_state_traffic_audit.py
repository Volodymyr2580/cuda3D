#!/usr/bin/env python3
"""Audit CPML state footprint and traffic gates for CUDA3D.

This is a gate tool, not an optimizer.  It answers whether compact CPML
state storage has enough byte budget to justify writing a CUDA prototype.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FLOAT_BYTES = 4
DEFAULT_RADIUS = 4
DEFAULT_CORE_PML_MARGIN = 4


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
    nypad: int
    nxpad: int
    nzpad: int
    domain_points: int
    padded_wavefield_points: int
    core_work_points: int
    pml_work_points: int


@dataclass(frozen=True)
class StateArray:
    name: str
    axis: str
    elements: int
    bytes: int
    formula: str
    owner: str


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
    nypad = ny + 2 * (npml + radius)
    nxpad = nx + 2 * (npml + radius)
    nzpad = nz + 2 * (npml + radius)
    domain_points = n3 * n2 * n1
    padded_points = nypad * nxpad * nzpad
    core_z = max(0, n1 - 2 * (npml + core_pml_margin))
    core_x = max(0, n2 - 2 * (npml + core_pml_margin))
    core_y = max(0, n3 - 2 * (npml + core_pml_margin))
    core_work = core_z * core_x * core_y
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
        nypad=nypad,
        nxpad=nxpad,
        nzpad=nzpad,
        domain_points=domain_points,
        padded_wavefield_points=padded_points,
        core_work_points=core_work,
        pml_work_points=domain_points - core_work,
    )


def state_arrays(g: Geometry, variant: str) -> list[StateArray]:
    mem_y = 2 * g.npml * g.n2 * g.n1
    mem_x = g.n3 * 2 * g.npml * g.n1
    mem_z = g.n3 * g.n2 * 2 * g.npml
    arrays = [
        StateArray("memory_dy", "y", mem_y, mem_y * FLOAT_BYTES, "2*npml*n2*n1", "v_pml"),
        StateArray("memory_dx", "x", mem_x, mem_x * FLOAT_BYTES, "n3*2*npml*n1", "v_pml"),
        StateArray("memory_dz", "z", mem_z, mem_z * FLOAT_BYTES, "n3*n2*2*npml", "p_pml zmem old"),
        StateArray("memory_dyy", "y", mem_y, mem_y * FLOAT_BYTES, "2*npml*n2*n1", "p_pml"),
        StateArray("memory_dxx", "x", mem_x, mem_x * FLOAT_BYTES, "n3*2*npml*n1", "p_pml"),
        StateArray("memory_dzz", "z", mem_z, mem_z * FLOAT_BYTES, "n3*n2*2*npml", "p_pml"),
    ]
    if variant in {"zmem", "cpml_dbuf"}:
        arrays.append(StateArray("memory_dz_next", "z", mem_z, mem_z * FLOAT_BYTES, "n3*n2*2*npml", "p_pml zmem next"))
    if variant == "cpml_dbuf":
        arrays.append(StateArray("memory_dy_next", "y", mem_y, mem_y * FLOAT_BYTES, "2*npml*n2*n1", "v_pml next"))
        arrays.append(StateArray("memory_dx_next", "x", mem_x, mem_x * FLOAT_BYTES, "n3*2*npml*n1", "v_pml next"))
    return arrays


def pml_region_breakdown(g: Geometry) -> dict[str, int]:
    y_band = 2 * g.npml
    x_band = 2 * g.npml
    z_band = 2 * g.npml
    y_int = g.n3 - y_band
    x_int = g.n2 - x_band
    z_int = g.n1 - z_band
    return {
        "z_only": z_band * x_int * y_int,
        "x_only": x_band * z_int * y_int,
        "y_only": y_band * z_int * x_int,
        "xy_edge": x_band * y_band * z_int,
        "xz_edge": x_band * z_band * y_int,
        "yz_edge": y_band * z_band * x_int,
        "xyz_corner": x_band * y_band * z_band,
    }


def zface_counts(g: Geometry) -> dict[str, int | float]:
    true_zface = 2 * g.npml * g.nx * g.ny
    safe_x = max(0, g.n2 - 2 * (g.npml + g.core_pml_margin))
    safe_y = max(0, g.n3 - 2 * (g.npml + g.core_pml_margin))
    safe_zface = 2 * g.npml * safe_x * safe_y
    mem_z = g.n3 * g.n2 * 2 * g.npml
    return {
        "memory_dz_elements": mem_z,
        "true_zface_elements": true_zface,
        "safe_zface_elements": safe_zface,
        "true_zface_of_memory_dz": true_zface / mem_z if mem_z else 0.0,
        "safe_zface_of_memory_dz": safe_zface / mem_z if mem_z else 0.0,
        "residual_edge_corner_elements_after_safe_zface": mem_z - safe_zface,
    }


def count_zmem_recompute_reads(g: Geometry) -> dict[str, int | float]:
    core_z0 = g.npml + g.core_pml_margin
    core_z1 = g.n1 - g.npml - g.core_pml_margin
    core_x_len = max(0, g.n2 - 2 * (g.npml + g.core_pml_margin))
    core_y_len = max(0, g.n3 - 2 * (g.npml + g.core_pml_margin))
    xy_total = g.n2 * g.n3
    xy_outside_core = xy_total - core_x_len * core_y_len
    offsets = (0, -1, 1, -2, 2, -3, 3, -4)

    calls = 0
    mem_reads = 0
    current_z_writes = 0
    for z in range(g.n1):
        xy_points = xy_total if (z < core_z0 or z >= core_z1) else xy_outside_core
        calls += xy_points * len(offsets)
        for dz in offsets:
            zz = z + dz
            if 0 <= zz < g.n1 and (zz < g.npml or zz >= g.n1 - g.npml):
                mem_reads += xy_points
        if z < g.npml or z >= g.n1 - g.npml:
            current_z_writes += xy_points

    return {
        "recompute_vz_calls_per_step": calls,
        "memory_dz_old_reads_per_step": mem_reads,
        "memory_dz_old_read_bytes_per_step": mem_reads * FLOAT_BYTES,
        "memory_dz_next_writes_per_step": current_z_writes,
        "memory_dz_next_write_bytes_per_step": current_z_writes * FLOAT_BYTES,
    }


def static_traffic(g: Geometry) -> dict[str, int | float]:
    mem_y = 2 * g.npml * g.n2 * g.n1
    mem_x = g.n3 * 2 * g.npml * g.n1
    mem_z = g.n3 * g.n2 * 2 * g.npml
    state_update_elements = 2 * (mem_y + mem_x + mem_z)
    zmem_reads = count_zmem_recompute_reads(g)
    pml_pressure_vxy_load_bytes = g.pml_work_points * 16 * FLOAT_BYTES
    pml_pressure_p0p1cw2_bytes = g.pml_work_points * 4 * FLOAT_BYTES
    mandatory_state_update_bytes = state_update_elements * 2 * FLOAT_BYTES
    return {
        "state_update_elements_per_step": state_update_elements,
        "mandatory_state_update_bytes_per_step": mandatory_state_update_bytes,
        "zmem_recompute": zmem_reads,
        "pml_pressure_vx_vy_load_bytes_per_step_est": pml_pressure_vxy_load_bytes,
        "pml_pressure_p0_p1_cw2_store_bytes_per_step_floor": pml_pressure_p0p1cw2_bytes,
        "pml_work_points": g.pml_work_points,
    }


def read_ncu_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ncu_kernel_duration_ns(summary: dict[str, Any] | None, label: str, kernel_substr: str) -> float | None:
    if not summary:
        return None
    for profile in summary.get("profiles", []):
        if profile.get("label") != label:
            continue
        for name, data in profile.get("kernels", {}).items():
            if kernel_substr in name:
                value = data.get("metrics", {}).get("duration_ns")
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def bytes_to_mib(value: float) -> float:
    return value / (1024.0 * 1024.0)


def render_markdown(result: dict[str, Any]) -> str:
    g = result["geometry"]
    arrays = result["state_arrays"]
    totals = result["totals"]
    zface = result["zface_counts"]
    traffic = result["traffic"]
    zmem_recompute = traffic["zmem_recompute"]
    gate = result["gate"]

    lines = [
        "# PML Compact-State Audit",
        "",
        "## Case",
        "",
        f"- case_dir: `{g['case_dir']}`",
        f"- logical ny/nx/nz: `{g['ny']}/{g['nx']}/{g['nz']}`",
        f"- domain n3/n2/n1 without radius: `{g['n3']}/{g['n2']}/{g['n1']}`",
        f"- padded wavefield nypad/nxpad/nzpad: `{g['nypad']}/{g['nxpad']}/{g['nzpad']}`",
        f"- npml/radius/CorePmlMargin: `{g['npml']}/{g['radius']}/{g['core_pml_margin']}`",
        f"- nt/shots/receivers_per_shot: `{g['nt']}/{g['shots']}/{g['receivers_per_shot']}`",
        f"- PML kernel work points: `{g['pml_work_points']}`",
        "",
        "## State Allocation",
        "",
        "| array | axis | elements | MiB | formula | owner |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for item in arrays:
        lines.append(
            f"| `{item['name']}` | {item['axis']} | {item['elements']} | "
            f"{bytes_to_mib(item['bytes']):.3f} | `{item['formula']}` | {item['owner']} |"
        )

    lines.extend(
        [
            "",
            f"- total CPML state footprint for `{result['variant']}`: `{bytes_to_mib(totals['state_bytes']):.3f} MiB`",
            f"- wavefield/cw2 floor footprint, six padded arrays: `{bytes_to_mib(totals['six_wavefield_bytes']):.3f} MiB`",
            f"- state as share of six wavefield arrays: `{totals['state_vs_six_wavefields']:.2%}`",
            "",
            "Important read: the current code already stores CPML memory as axis slabs, not as full padded-domain arrays.",
            "",
            "## PML Face/Edge/Corner Distribution",
            "",
            "| region | points |",
            "| --- | ---: |",
        ]
    )
    for key, value in result["pml_region_breakdown"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Z-Face Compact Coverage",
            "",
            f"- `memory_dz` elements: `{zface['memory_dz_elements']}`",
            f"- true z-face elements: `{zface['true_zface_elements']}` ({zface['true_zface_of_memory_dz']:.2%} of `memory_dz`)",
            f"- safe z-face elements with CorePmlMargin: `{zface['safe_zface_elements']}` ({zface['safe_zface_of_memory_dz']:.2%} of `memory_dz`)",
            f"- residual z edge/corner elements that still need state: `{zface['residual_edge_corner_elements_after_safe_zface']}`",
            "",
            "A safe compact z-face layout can be affine-indexed without division/mod, but it does not remove the residual z edge/corner state.",
            "",
            "## Static Traffic Floor",
            "",
            f"- mandatory CPML state update traffic floor: `{bytes_to_mib(traffic['mandatory_state_update_bytes_per_step']):.3f} MiB/step`",
            f"- zmem `memory_dz` old reads from recompute path: `{bytes_to_mib(zmem_recompute['memory_dz_old_read_bytes_per_step']):.3f} MiB/step`",
            f"- zmem `memory_dz_next` writes: `{bytes_to_mib(zmem_recompute['memory_dz_next_write_bytes_per_step']):.3f} MiB/step`",
            f"- pressure PML vx/vy load estimate: `{bytes_to_mib(traffic['pml_pressure_vx_vy_load_bytes_per_step_est']):.3f} MiB/step`",
            f"- pressure PML p0/p1/cw2/store floor: `{bytes_to_mib(traffic['pml_pressure_p0_p1_cw2_store_bytes_per_step_floor']):.3f} MiB/step`",
            "",
            "## Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- estimated compact-state WP speedup ceiling: `{gate['estimated_wp_speedup_ceiling']:.3f}x`",
            f"- reason: {gate['reason']}",
        ]
    )
    if result.get("ncu"):
        lines.extend(["", "## NCU Link", "", "```json", json.dumps(result["ncu"], indent=2), "```"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="benchmarks/cases/perf_1gpu_6shots")
    parser.add_argument("--variant", choices=["zmem", "cpml_dbuf"], default="cpml_dbuf")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--core-pml-margin", type=int, default=DEFAULT_CORE_PML_MARGIN)
    parser.add_argument("--ncu-summary-json")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case)
    if not case_dir.is_absolute():
        case_dir = root / case_dir
    geometry = load_geometry(case_dir, args.radius, args.core_pml_margin)
    arrays = state_arrays(geometry, args.variant)
    state_bytes = sum(item.bytes for item in arrays)
    six_wavefield_bytes = geometry.padded_wavefield_points * FLOAT_BYTES * 6
    traffic = static_traffic(geometry)

    ncu_summary = read_ncu_summary(Path(args.ncu_summary_json) if args.ncu_summary_json else None)
    zmem_pml_duration = ncu_kernel_duration_ns(ncu_summary, "zmem", "cuda_fd3d_p_pml")
    cpml_pml_duration = ncu_kernel_duration_ns(ncu_summary, "cpml_dbuf", "cuda_fd3d_p_pml")
    ncu = None
    if ncu_summary:
        ncu = {
            "summary_json": args.ncu_summary_json,
            "zmem_p_pml_duration_ns": zmem_pml_duration,
            "cpml_dbuf_p_pml_duration_ns": cpml_pml_duration,
        }

    # Because the arrays are already axis-slab compact, a safe compact-zface
    # state layout mostly changes indexing/layout.  We allow only a small
    # optimistic ceiling unless NCU later proves CPML state traffic dominates.
    safe_zface_share = zface_counts(geometry)["safe_zface_of_memory_dz"]
    impossible_z_residual_share = 1.0 - float(safe_zface_share)
    estimated_wp_speedup_ceiling = 1.0 + min(0.02, 0.03 * impossible_z_residual_share)
    gate_pass = estimated_wp_speedup_ceiling >= 1.05
    gate = {
        "verdict": "continue" if gate_pass else "stop_compact_state",
        "estimated_wp_speedup_ceiling": estimated_wp_speedup_ceiling,
        "reason": (
            "Current CPML memory is already stored as y/x/z slabs.  Safe z-face compacting "
            "does not remove edge/corner state and has no >=5% static WP ceiling without "
            "new profiler evidence that CPML state layout dominates stalls."
        ),
    }

    result = {
        "variant": args.variant,
        "geometry": asdict(geometry),
        "state_arrays": [asdict(item) for item in arrays],
        "totals": {
            "state_bytes": state_bytes,
            "six_wavefield_bytes": six_wavefield_bytes,
            "state_vs_six_wavefields": state_bytes / six_wavefield_bytes if six_wavefield_bytes else math.nan,
        },
        "pml_region_breakdown": pml_region_breakdown(geometry),
        "zface_counts": zface_counts(geometry),
        "traffic": traffic,
        "ncu": ncu,
        "gate": gate,
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_markdown(result)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(md)


if __name__ == "__main__":
    main()
