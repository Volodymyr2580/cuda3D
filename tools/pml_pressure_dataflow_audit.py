#!/usr/bin/env python3
"""Audit pressure-PML tile dataflow and structural optimization gates.

This tool mirrors the host-side pressure PML tile list construction and counts
where the current `cuda_fd3d_p_pml_tile_ns` work is spent.  It intentionally
does not edit CUDA code; it is a gate before opening another prototype.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FLOAT_BYTES = 4
DEFAULT_RADIUS = 7
DEFAULT_CORE_PML_MARGIN = 4
DEFAULT_B1 = 32
DEFAULT_B2 = 4
DEFAULT_B3 = 2

PML_TILE_MASK_Z = 1
PML_TILE_MASK_X = 2
PML_TILE_MASK_Y = 4
PML_TILE_MASK_MIXED = 8


@dataclass(frozen=True)
class ShotDomain:
    shot: int
    source_index_yxz: list[int]
    receiver_count: int
    receiver_min_yx: list[int]
    receiver_max_yx: list[int]
    domain_origin_yx: list[int]
    domain_size_yx: list[int]
    n3n2n1: list[int]


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


def _as_float(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def parse_input(path: Path, ny: int, nx: int, nz: int) -> dict[str, float | int]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    for i in range(0, max(0, len(lines) - 8)):
        if _as_int(lines[i]) != ny:
            continue
        if _as_int(lines[i + 1]) == nx and _as_int(lines[i + 2]) == nz:
            dy = _as_float(lines[i + 3])
            dx = _as_float(lines[i + 4])
            dz = _as_float(lines[i + 5])
            npml = _as_int(lines[i + 6])
            xpad = _as_float(lines[i + 8])
            if None in (dy, dx, dz, npml, xpad):
                continue
            return {
                "dy": float(dy),
                "dx": float(dx),
                "dz": float(dz),
                "npml": int(npml),
                "xpad": float(xpad),
            }
    raise ValueError(f"could not parse geometry from {path}")


def read_nav(path: Path) -> list[tuple[float, float, float, float, float, float]]:
    raw = path.read_bytes()
    if len(raw) % (6 * 4) != 0:
        raise ValueError(f"nav byte length is not divisible by 6 float32 values: {path}")
    count = len(raw) // 4
    vals = struct.unpack(f"<{count}f", raw)
    return [tuple(vals[i : i + 6]) for i in range(0, count, 6)]  # type: ignore[misc]


def idx(value: float, spacing: float) -> int:
    return int(value * 1000.0 / (spacing * 1000.0))


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def interval_len(lo: int, hi: int) -> int:
    return max(0, hi - lo)


def intersection_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def box_intersection_volume(
    z0: int,
    z1: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    core1_lo: int,
    core1_hi: int,
    core2_lo: int,
    core2_hi: int,
    core3_lo: int,
    core3_hi: int,
) -> int:
    return (
        intersection_len(z0, z1, core1_lo, core1_hi)
        * intersection_len(x0, x1, core2_lo, core2_hi)
        * intersection_len(y0, y1, core3_lo, core3_hi)
    )


def true_pml_volume_in_box(z0: int, z1: int, x0: int, x1: int, y0: int, y1: int, n1: int, n2: int, n3: int, npml: int) -> int:
    volume = (z1 - z0) * (x1 - x0) * (y1 - y0)
    interior = (
        intersection_len(z0, z1, npml, n1 - npml)
        * intersection_len(x0, x1, npml, n2 - npml)
        * intersection_len(y0, y1, npml, n3 - npml)
    )
    return volume - interior


def tile_fully_inside_box(
    z0: int,
    x0: int,
    y0: int,
    bz: int,
    bx: int,
    by: int,
    n1: int,
    n2: int,
    n3: int,
    zlo: int,
    zhi: int,
    xlo: int,
    xhi: int,
    ylo: int,
    yhi: int,
) -> bool:
    z1 = min(z0 + bz, n1)
    x1 = min(x0 + bx, n2)
    y1 = min(y0 + by, n3)
    return z0 >= zlo and z1 <= zhi and x0 >= xlo and x1 <= xhi and y0 >= ylo and y1 <= yhi


def make_pml_tile_mask(z0: int, x0: int, y0: int, bz: int, bx: int, by: int, n1: int, n2: int, n3: int, npml: int) -> int:
    z1 = min(z0 + bz, n1)
    x1 = min(x0 + bx, n2)
    y1 = min(y0 + by, n3)
    z_active = z0 < npml or z1 > n1 - npml
    x_active = x0 < npml or x1 > n2 - npml
    y_active = y0 < npml or y1 > n3 - npml
    mask = 0
    axes = 0
    if z_active:
        mask |= PML_TILE_MASK_Z
        axes += 1
    if x_active:
        mask |= PML_TILE_MASK_X
        axes += 1
    if y_active:
        mask |= PML_TILE_MASK_Y
        axes += 1
    if axes > 1:
        mask |= PML_TILE_MASK_MIXED
    return mask


def mask_label(mask: int) -> str:
    axes = []
    if mask & PML_TILE_MASK_Z:
        axes.append("z")
    if mask & PML_TILE_MASK_X:
        axes.append("x")
    if mask & PML_TILE_MASK_Y:
        axes.append("y")
    return "shell" if not axes else "".join(axes)


def active_line_points(
    z0: int,
    z1: int,
    x: int,
    y: int,
    core1_lo: int,
    core1_hi: int,
    core2_lo: int,
    core2_hi: int,
    core3_lo: int,
    core3_hi: int,
) -> int:
    z_count = z1 - z0
    if core2_lo <= x < core2_hi and core3_lo <= y < core3_hi:
        z_count -= intersection_len(z0, z1, core1_lo, core1_hi)
    return max(0, z_count)


def axis_category_counts(n3: int, n2: int, n1: int, npml: int, core_pml_margin: int) -> dict[str, int]:
    core1_lo = npml + core_pml_margin
    core2_lo = npml + core_pml_margin
    core3_lo = npml + core_pml_margin
    core1_hi = n1 - npml - core_pml_margin
    core2_hi = n2 - npml - core_pml_margin
    core3_hi = n3 - npml - core_pml_margin
    domain = n3 * n2 * n1
    core = (
        interval_len(core1_lo, core1_hi)
        * interval_len(core2_lo, core2_hi)
        * interval_len(core3_lo, core3_hi)
    )
    active = domain - core

    z_band = 2 * npml
    x_band = 2 * npml
    y_band = 2 * npml
    z_int = n1 - z_band
    x_int = n2 - x_band
    y_int = n3 - y_band
    true_counts = {
        "z_only": z_band * x_int * y_int,
        "x_only": x_band * z_int * y_int,
        "y_only": y_band * z_int * x_int,
        "xy_edge": x_band * y_band * z_int,
        "xz_edge": x_band * z_band * y_int,
        "yz_edge": y_band * z_band * x_int,
        "xyz_corner": x_band * y_band * z_band,
    }
    true_pml = sum(true_counts.values())
    shell = active - true_pml
    return {"active_total": active, "shell": shell, **true_counts, "true_pml_total": true_pml}


def shot_domains(case_dir: Path) -> tuple[dict[str, Any], list[ShotDomain]]:
    manifest = parse_kv(case_dir / "case_manifest.txt")
    ny = int(manifest["ny"])
    nx = int(manifest["nx"])
    nz = int(manifest["nz"])
    shots = int(manifest["shots"])
    rec_per_shot = int(manifest["receivers_per_shot"])
    input_name = manifest["input"]
    nav_name = manifest["nav"]
    geo = parse_input(case_dir / input_name, ny, nx, nz)
    dy = float(geo["dy"])
    dx = float(geo["dx"])
    dz = float(geo["dz"])
    npml = int(geo["npml"])
    xpad = float(geo["xpad"])
    xypad = int(xpad / dx)
    nav = read_nav(case_dir / nav_name)
    if len(nav) != shots * rec_per_shot:
        raise ValueError(f"nav trace count {len(nav)} != shots*receivers {shots * rec_per_shot}")

    domains: list[ShotDomain] = []
    for shot in range(shots):
        traces = nav[shot * rec_per_shot : (shot + 1) * rec_per_shot]
        sx, sy, sz = traces[0][0], traces[0][1], traces[0][2]
        source = [idx(sy, dy), idx(sx, dx), idx(sz, dz)]
        receivers = [[idx(t[4], dy), idx(t[3], dx), idx(t[5], dz)] for t in traces]
        ry = [r[0] for r in receivers]
        rx = [r[1] for r in receivers]
        min_y = min([source[0], *ry])
        max_y = max([source[0], *ry])
        min_x = min([source[1], *rx])
        max_x = max([source[1], *rx])
        yl = max(0, min_y - xypad)
        yr = min(ny - 1, max_y + xypad)
        xl = max(0, min_x - xypad)
        xr = min(nx - 1, max_x + xypad)
        ny_new = yr - yl + 1
        nx_new = xr - xl + 1
        n3 = ny_new + 2 * npml
        n2 = nx_new + 2 * npml
        n1 = nz + 2 * npml
        domains.append(
            ShotDomain(
                shot=shot,
                source_index_yxz=source,
                receiver_count=len(receivers),
                receiver_min_yx=[min(ry), min(rx)],
                receiver_max_yx=[max(ry), max(rx)],
                domain_origin_yx=[yl, xl],
                domain_size_yx=[ny_new, nx_new],
                n3n2n1=[n3, n2, n1],
            )
        )
    return {
        "manifest": manifest,
        "ny_nx_nz": [ny, nx, nz],
        "shots": shots,
        "receivers_per_shot": rec_per_shot,
        "spacing_dy_dx_dz": [dy, dx, dz],
        "npml": npml,
        "xpad": xpad,
        "xypad_grid_points": xypad,
    }, domains


def add_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + int(value)


def analyze_domain(domain: ShotDomain, npml: int, core_pml_margin: int, b1: int, b2: int, b3: int) -> dict[str, Any]:
    n3, n2, n1 = domain.n3n2n1
    grid1 = ceil_div(n1, b1)
    grid2 = ceil_div(n2, b2)
    grid3 = ceil_div(n3, b3)
    block_threads = b1 * b2 * b3
    core1_lo = npml + core_pml_margin
    core2_lo = npml + core_pml_margin
    core3_lo = npml + core_pml_margin
    core1_hi = n1 - npml - core_pml_margin
    core2_hi = n2 - npml - core_pml_margin
    core3_hi = n3 - npml - core_pml_margin

    tile_mask_counts: dict[str, int] = {}
    tile_active_by_mask: dict[str, int] = {}
    tile_shell_by_mask: dict[str, int] = {}
    axis_counts = axis_category_counts(n3, n2, n1, npml, core_pml_margin)

    kept_tiles = 0
    valid_domain_threads = 0
    active_threads = 0
    returned_core_threads = 0
    boundary_padding_threads = 0
    shared_vz_cache_calls = 0
    line_slots_with_active_work = 0

    for by in range(grid3):
        y0 = by * b3
        y1 = min(y0 + b3, n3)
        for bx in range(grid2):
            x0 = bx * b2
            x1 = min(x0 + b2, n2)
            for bz in range(grid1):
                z0 = bz * b1
                z1 = min(z0 + b1, n1)
                skip = tile_fully_inside_box(
                    z0,
                    x0,
                    y0,
                    b1,
                    b2,
                    b3,
                    n1,
                    n2,
                    n3,
                    core1_lo,
                    core1_hi,
                    core2_lo,
                    core2_hi,
                    core3_lo,
                    core3_hi,
                )
                if skip:
                    continue
                kept_tiles += 1
                mask = make_pml_tile_mask(z0, x0, y0, b1, b2, b3, n1, n2, n3, npml)
                label = mask_label(mask)
                tile_mask_counts[label] = tile_mask_counts.get(label, 0) + 1

                valid_volume = (z1 - z0) * (x1 - x0) * (y1 - y0)
                core_volume = box_intersection_volume(
                    z0,
                    z1,
                    x0,
                    x1,
                    y0,
                    y1,
                    core1_lo,
                    core1_hi,
                    core2_lo,
                    core2_hi,
                    core3_lo,
                    core3_hi,
                )
                active_volume = valid_volume - core_volume
                true_pml_volume = true_pml_volume_in_box(z0, z1, x0, x1, y0, y1, n1, n2, n3, npml)
                tile_shell = active_volume - true_pml_volume
                valid_domain_threads += valid_volume
                active_threads += active_volume
                returned_core_threads += core_volume
                boundary_padding_threads += block_threads - valid_volume
                tile_active_by_mask[label] = tile_active_by_mask.get(label, 0) + active_volume

                for y in range(y0, y1):
                    for x in range(x0, x1):
                        active_line = active_line_points(
                            z0,
                            z1,
                            x,
                            y,
                            core1_lo,
                            core1_hi,
                            core2_lo,
                            core2_hi,
                            core3_lo,
                            core3_hi,
                        )
                        if active_line == 0:
                            continue
                        line_slots_with_active_work += 1
                        shared_vz_cache_calls += max(0, min(n1, z1 + 3) - max(0, z0 - 4))
                tile_shell_by_mask[label] = tile_shell_by_mask.get(label, 0) + tile_shell

    launched_threads = kept_tiles * block_threads
    current_vz_recompute_calls = active_threads * 8
    current_vz_p1_loads = current_vz_recompute_calls * 8
    shared_vz_p1_loads = shared_vz_cache_calls * 8
    return {
        "shot": domain.shot,
        "domain": asdict(domain),
        "grid_zxy": [grid1, grid2, grid3],
        "block_zxy": [b1, b2, b3],
        "tiles": {
            "kept": kept_tiles,
            "full_grid": grid1 * grid2 * grid3,
            "skipped_core": grid1 * grid2 * grid3 - kept_tiles,
            "mask_counts": tile_mask_counts,
            "active_points_by_tile_mask": tile_active_by_mask,
            "shell_points_by_tile_mask": tile_shell_by_mask,
        },
        "threads": {
            "launched": launched_threads,
            "valid_domain": valid_domain_threads,
            "active_after_core_return": active_threads,
            "returned_core": returned_core_threads,
            "boundary_padding": boundary_padding_threads,
            "active_thread_efficiency": active_threads / launched_threads if launched_threads else math.nan,
            "valid_domain_efficiency": valid_domain_threads / launched_threads if launched_threads else math.nan,
        },
        "point_categories": axis_counts,
        "z_recompute": {
            "current_vz_recompute_calls": current_vz_recompute_calls,
            "shared_line_cache_vz_recompute_calls": shared_vz_cache_calls,
            "line_slots_with_active_work": line_slots_with_active_work,
            "current_vz_p1_loads_est": current_vz_p1_loads,
            "shared_line_cache_vz_p1_loads_est": shared_vz_p1_loads,
            "vz_recompute_call_reduction": 1.0 - shared_vz_cache_calls / current_vz_recompute_calls
            if current_vz_recompute_calls
            else 0.0,
            "vz_p1_load_reduction_est": 1.0 - shared_vz_p1_loads / current_vz_p1_loads
            if current_vz_p1_loads
            else 0.0,
        },
    }


def read_ncu_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ncu_kernel_duration_ns(summary: dict[str, Any] | None, kernel_substr: str) -> float | None:
    if not summary:
        return None
    for profile in summary.get("profiles", []):
        for name, data in profile.get("kernels", {}).items():
            if kernel_substr in name:
                value = data.get("metrics", {}).get("duration_ns")
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def aggregate(case_meta: dict[str, Any], shot_results: list[dict[str, Any]], ncu: dict[str, Any] | None) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "tiles": {"kept": 0, "full_grid": 0, "skipped_core": 0, "mask_counts": {}, "active_points_by_tile_mask": {}, "shell_points_by_tile_mask": {}},
        "threads": {"launched": 0, "valid_domain": 0, "active_after_core_return": 0, "returned_core": 0, "boundary_padding": 0},
        "point_categories": {},
        "z_recompute": {
            "current_vz_recompute_calls": 0,
            "shared_line_cache_vz_recompute_calls": 0,
            "line_slots_with_active_work": 0,
            "current_vz_p1_loads_est": 0,
            "shared_line_cache_vz_p1_loads_est": 0,
        },
    }
    for item in shot_results:
        for key in ("kept", "full_grid", "skipped_core"):
            totals["tiles"][key] += item["tiles"][key]
        add_counts(totals["tiles"]["mask_counts"], item["tiles"]["mask_counts"])
        add_counts(totals["tiles"]["active_points_by_tile_mask"], item["tiles"]["active_points_by_tile_mask"])
        add_counts(totals["tiles"]["shell_points_by_tile_mask"], item["tiles"]["shell_points_by_tile_mask"])
        for key in ("launched", "valid_domain", "active_after_core_return", "returned_core", "boundary_padding"):
            totals["threads"][key] += item["threads"][key]
        add_counts(totals["point_categories"], item["point_categories"])
        for key in totals["z_recompute"]:
            totals["z_recompute"][key] += item["z_recompute"][key]

    launched = totals["threads"]["launched"]
    active = totals["threads"]["active_after_core_return"]
    current_calls = totals["z_recompute"]["current_vz_recompute_calls"]
    shared_calls = totals["z_recompute"]["shared_line_cache_vz_recompute_calls"]
    totals["threads"]["active_thread_efficiency"] = active / launched if launched else math.nan
    totals["threads"]["valid_domain_efficiency"] = totals["threads"]["valid_domain"] / launched if launched else math.nan
    totals["z_recompute"]["vz_recompute_call_reduction"] = 1.0 - shared_calls / current_calls if current_calls else 0.0
    totals["z_recompute"]["vz_p1_load_reduction_est"] = totals["z_recompute"]["vz_recompute_call_reduction"]

    p_pml_duration = ncu_kernel_duration_ns(ncu, "cuda_fd3d_p_pml") if ncu else None
    p_core_duration = ncu_kernel_duration_ns(ncu, "cuda_fd3d_p_core") if ncu else None
    v_pml_duration = ncu_kernel_duration_ns(ncu, "cuda_fd3d_v_pml") if ncu else None
    sampled_main = sum(v for v in (p_pml_duration, p_core_duration, v_pml_duration) if v is not None)
    p_pml_sampled_share = p_pml_duration / sampled_main if p_pml_duration and sampled_main else None

    z_reduction = totals["z_recompute"]["vz_recompute_call_reduction"]
    # p_pml does x/y velocity divergence and pressure stores too.  Treat z
    # recompute as a large but not total fraction until hardware counters prove
    # otherwise.
    modeled_p_pml_speedup_if_shared = 1.0 / max(1e-9, 1.0 - 0.45 * z_reduction)
    modeled_sampled_main_speedup = (
        1.0 / ((1.0 - p_pml_sampled_share) + p_pml_sampled_share / modeled_p_pml_speedup_if_shared)
        if p_pml_sampled_share is not None
        else None
    )
    shell_share = totals["point_categories"].get("shell", 0) / active if active else 0.0
    true_pml_share = totals["point_categories"].get("true_pml_total", 0) / active if active else 0.0
    tile_shell_share = totals["tiles"]["mask_counts"].get("shell", 0) / totals["tiles"]["kept"] if totals["tiles"]["kept"] else 0.0

    gate_pass = modeled_sampled_main_speedup is not None and modeled_sampled_main_speedup >= 1.05
    gate = {
        "verdict": "open_p_pml_z_recompute_line_cache_prototype" if gate_pass else "stop_pressure_pml_structural_prototype",
        "reason": (
            "Shared z-line recompute has a >=5% sampled-main model ceiling and is not the forbidden z-face/shared-VP route."
            if gate_pass
            else "Pressure PML audit does not show a >=5% sampled-main structural ceiling without new profiler evidence."
        ),
        "allowed_prototype": (
            "CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE"
            if gate_pass
            else None
        ),
        "forbidden_repeats": [
            "CUDA3D_PML_TILE_MASK_FASTPATH",
            "CUDA3D_PML_ZFACE_P_SPECIALIZE",
            "CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY",
            "CUDA3D_PML_ZFACE_SHARED_VP_DEBUG",
            "RECOMPUTE_X/Y/XYZ",
        ],
    }
    return {
        "case": case_meta,
        "totals": totals,
        "derived": {
            "shell_active_point_share": shell_share,
            "true_pml_active_point_share": true_pml_share,
            "shell_tile_share": tile_shell_share,
            "p_pml_duration_ns": p_pml_duration,
            "p_core_duration_ns": p_core_duration,
            "v_pml_duration_ns": v_pml_duration,
            "p_pml_sampled_main_share": p_pml_sampled_share,
            "modeled_p_pml_speedup_if_shared_z_recompute": modeled_p_pml_speedup_if_shared,
            "modeled_sampled_main_speedup_if_shared_z_recompute": modeled_sampled_main_speedup,
        },
        "gate": gate,
    }


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.2%}"


def ratio(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}x"


def mib_from_float_count(count: int) -> float:
    return count * FLOAT_BYTES / (1024.0 * 1024.0)


def render_markdown(result: dict[str, Any]) -> str:
    totals = result["totals"]
    derived = result["derived"]
    gate = result["gate"]
    zrec = totals["z_recompute"]
    threads = totals["threads"]
    cats = totals["point_categories"]

    lines = [
        "# Pressure PML Dataflow Audit",
        "",
        "## Case",
        "",
        f"- case_dir: `{result['case']['case_dir']}`",
        f"- logical ny/nx/nz: `{result['case']['ny_nx_nz'][0]}/{result['case']['ny_nx_nz'][1]}/{result['case']['ny_nx_nz'][2]}`",
        f"- shots/receivers_per_shot: `{result['case']['shots']}/{result['case']['receivers_per_shot']}`",
        f"- npml/xpad: `{result['case']['npml']}/{result['case']['xpad']}`",
        f"- PML tile block z/x/y: `{result['case']['tile_block_zxy'][0]}/{result['case']['tile_block_zxy'][1]}/{result['case']['tile_block_zxy'][2]}`",
        "",
        "## Tile And Thread Shape",
        "",
        f"- kept pressure-PML tiles: `{totals['tiles']['kept']}` / full grid `{totals['tiles']['full_grid']}`",
        f"- active thread efficiency after core return: `{pct(threads['active_thread_efficiency'])}`",
        f"- valid-domain thread efficiency before core return: `{pct(threads['valid_domain_efficiency'])}`",
        f"- boundary padding threads: `{threads['boundary_padding']}`",
        f"- returned-core threads inside kept tiles: `{threads['returned_core']}`",
        "",
        "### Tile Masks",
        "",
        "| mask | tiles | active points | shell points |",
        "| --- | ---: | ---: | ---: |",
    ]
    mask_keys = sorted(totals["tiles"]["mask_counts"], key=lambda k: (k != "shell", k))
    for key in mask_keys:
        lines.append(
            f"| `{key}` | {totals['tiles']['mask_counts'].get(key, 0)} | "
            f"{totals['tiles']['active_points_by_tile_mask'].get(key, 0)} | "
            f"{totals['tiles']['shell_points_by_tile_mask'].get(key, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Point Categories",
            "",
            "| category | points | share of active |",
            "| --- | ---: | ---: |",
        ]
    )
    active = cats.get("active_total", 0)
    for key in ("shell", "z_only", "x_only", "y_only", "xy_edge", "xz_edge", "yz_edge", "xyz_corner", "true_pml_total"):
        value = cats.get(key, 0)
        lines.append(f"| `{key}` | {value} | {pct(value / active if active else None)} |")

    lines.extend(
        [
            "",
            "## Z-Recompute Reuse Budget",
            "",
            f"- current `recompute_vz_after_update` calls: `{zrec['current_vz_recompute_calls']}`",
            f"- shared z-line cache calls estimate: `{zrec['shared_line_cache_vz_recompute_calls']}`",
            f"- call reduction estimate: `{pct(zrec['vz_recompute_call_reduction'])}`",
            f"- current p1 load estimate inside z recompute: `{mib_from_float_count(zrec['current_vz_p1_loads_est']):.3f} MiB/step aggregate-shots`",
            f"- shared-cache p1 load estimate: `{mib_from_float_count(zrec['shared_line_cache_vz_p1_loads_est']):.3f} MiB/step aggregate-shots`",
            "",
            "## NCU Link And Model",
            "",
            f"- p_pml duration: `{derived['p_pml_duration_ns']}` ns",
            f"- p_core duration: `{derived['p_core_duration_ns']}` ns",
            f"- v_pml duration: `{derived['v_pml_duration_ns']}` ns",
            f"- p_pml sampled-main share: `{pct(derived['p_pml_sampled_main_share'])}`",
            f"- modeled p_pml speedup if shared z recompute succeeds: `{ratio(derived['modeled_p_pml_speedup_if_shared_z_recompute'])}`",
            f"- modeled sampled-main speedup: `{ratio(derived['modeled_sampled_main_speedup_if_shared_z_recompute'])}`",
            "",
            "## Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- allowed prototype: `{gate['allowed_prototype']}`",
            f"- reason: {gate['reason']}",
            "- discipline: this gate does not reopen tile-mask fastpath, z-face specialize/fusion, or RECOMPUTE_X/Y/XYZ.",
            "",
            "## Shot Table",
            "",
            "| shot | domain y/x | active points | shell share | z-recompute reduction | active thread efficiency |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for shot in result["shots_detail"]:
        shot_active = shot["threads"]["active_after_core_return"]
        shot_shell = shot["point_categories"]["shell"]
        lines.append(
            f"| {shot['shot']} | {shot['domain']['domain_size_yx'][0]}x{shot['domain']['domain_size_yx'][1]} | "
            f"{shot_active} | {pct(shot_shell / shot_active if shot_active else None)} | "
            f"{pct(shot['z_recompute']['vz_recompute_call_reduction'])} | "
            f"{pct(shot['threads']['active_thread_efficiency'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="benchmarks/cases/perf_1gpu_6shots")
    parser.add_argument("--core-pml-margin", type=int, default=DEFAULT_CORE_PML_MARGIN)
    parser.add_argument("--tile-b1", type=int, default=DEFAULT_B1)
    parser.add_argument("--tile-b2", type=int, default=DEFAULT_B2)
    parser.add_argument("--tile-b3", type=int, default=DEFAULT_B3)
    parser.add_argument("--ncu-summary-json", default="reports/day_20260608/zmem_core_pml_sol_ncu_summary.json")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case)
    if not case_dir.is_absolute():
        case_dir = root / case_dir
    case_meta, domains = shot_domains(case_dir)
    case_meta["case_dir"] = str(case_dir)
    case_meta["core_pml_margin"] = args.core_pml_margin
    case_meta["tile_block_zxy"] = [args.tile_b1, args.tile_b2, args.tile_b3]
    ncu_path = Path(args.ncu_summary_json)
    if not ncu_path.is_absolute():
        ncu_path = root / ncu_path
    ncu = read_ncu_summary(ncu_path)

    shot_results = [
        analyze_domain(domain, int(case_meta["npml"]), args.core_pml_margin, args.tile_b1, args.tile_b2, args.tile_b3)
        for domain in domains
    ]
    summary = aggregate(case_meta, shot_results, ncu)
    result = {**summary, "shots_detail": shot_results}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_markdown(result)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(md)


if __name__ == "__main__":
    main()
