#!/usr/bin/env python3
import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path


RADIUS = 7
CORE_PML_MARGIN = 4
FLOAT_BYTES = 4
THREADS_PER_CTA = 256
WARPS_PER_CTA = THREADS_PER_CTA // 32
SM120_SHARED_PER_SM = 128 * 1024
SM120_SHARED_PER_BLOCK = 99 * 1024
SM120_MAX_WARPS_PER_SM = 48
SM120_MAX_BLOCKS_PER_SM = 32


@dataclass(frozen=True)
class TilePlan:
    name: str
    mz: int
    mx: int
    my: int
    implement: bool

    @property
    def m_volume(self):
        return self.mz * self.mx * self.my

    @property
    def shared_bytes(self):
        return self.m_volume * FLOAT_BYTES

    @property
    def o_dims(self):
        return (self.mz + 2 * RADIUS, self.mx + 2 * RADIUS, self.my + 2 * RADIUS)

    @property
    def c_dims(self):
        return (
            max(0, self.mz - 2 * RADIUS),
            max(0, self.mx - 2 * RADIUS),
            max(0, self.my - 2 * RADIUS),
        )

    @property
    def c_volume(self):
        cz, cx, cy = self.c_dims
        return cz * cx * cy


PLANS = [
    TilePlan("A", 32, 24, 20, True),
    TilePlan("B", 32, 24, 24, False),
    TilePlan("C", 40, 24, 20, False),
    TilePlan("D", 40, 24, 24, True),
    TilePlan("E", 48, 20, 24, False),
]


def read_manifest(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def read_nav(path):
    raw = path.read_bytes()
    count = len(raw) // 4
    return list(struct.unpack(f"<{count}f", raw))


def coord_to_index(coord, spacing):
    return int(math.floor(coord / spacing))


def parse_case(case_dir):
    manifest = read_manifest(case_dir / "case_manifest.txt")
    ny = int(manifest["ny"])
    nx = int(manifest["nx"])
    nz = int(manifest["nz"])
    npml = int(manifest["npml"])
    nav = read_nav(case_dir / manifest["nav"])
    spacing = 0.025

    nby = ny + 2 * npml
    nbx = nx + 2 * npml
    nbz = nz + 2 * npml
    core = {
        "z0": npml + CORE_PML_MARGIN,
        "z1": nbz - npml - CORE_PML_MARGIN,
        "x0": npml + CORE_PML_MARGIN,
        "x1": nbx - npml - CORE_PML_MARGIN,
        "y0": npml + CORE_PML_MARGIN,
        "y1": nby - npml - CORE_PML_MARGIN,
    }
    core_points = (
        (core["z1"] - core["z0"])
        * (core["x1"] - core["x0"])
        * (core["y1"] - core["y0"])
    )

    source = (
        npml + coord_to_index(nav[2], spacing),
        npml + coord_to_index(nav[0], spacing),
        npml + coord_to_index(nav[1], spacing),
    )
    receivers = []
    for i in range(0, len(nav), 6):
        receivers.append(
            (
                npml + coord_to_index(nav[i + 5], spacing),
                npml + coord_to_index(nav[i + 3], spacing),
                npml + coord_to_index(nav[i + 4], spacing),
            )
        )
    return manifest, core, core_points, source, receivers


def region_contains(region, point):
    z, x, y = point
    return (
        region["z0"] <= z < region["z1"]
        and region["x0"] <= x < region["x1"]
        and region["y0"] <= y < region["y1"]
    )


def make_region(z0, x0, y0, z_len, x_len, y_len):
    return {
        "z0": z0,
        "z1": z0 + z_len,
        "x0": x0,
        "x1": x0 + x_len,
        "y0": y0,
        "y1": y0 + y_len,
    }


def tile_starts(lo, hi, size):
    usable = max(0, hi - lo)
    count = usable // size
    return [lo + i * size for i in range(count)]


def analyze_plan(plan, core, core_points, source, receivers):
    m_lo = {
        "z": core["z0"] + RADIUS,
        "x": core["x0"] + RADIUS,
        "y": core["y0"] + RADIUS,
    }
    m_hi = {
        "z": core["z1"] - RADIUS,
        "x": core["x1"] - RADIUS,
        "y": core["y1"] - RADIUS,
    }
    starts_z = tile_starts(m_lo["z"], m_hi["z"], plan.mz)
    starts_x = tile_starts(m_lo["x"], m_hi["x"], plan.mx)
    starts_y = tile_starts(m_lo["y"], m_hi["y"], plan.my)

    proposed_tiles = len(starts_z) * len(starts_x) * len(starts_y)
    kept_tiles = 0
    rejected_source_o = 0
    rejected_receiver_o = 0
    source_in_any_o = False
    source_in_any_c = False
    receivers_in_any_o = 0
    receivers_in_any_c = 0
    commit_points = 0

    cz, cx, cy = plan.c_dims
    for z0 in starts_z:
        for x0 in starts_x:
            for y0 in starts_y:
                m = make_region(z0, x0, y0, plan.mz, plan.mx, plan.my)
                o = make_region(
                    m["z0"] - RADIUS,
                    m["x0"] - RADIUS,
                    m["y0"] - RADIUS,
                    plan.mz + 2 * RADIUS,
                    plan.mx + 2 * RADIUS,
                    plan.my + 2 * RADIUS,
                )
                c = make_region(
                    m["z0"] + RADIUS,
                    m["x0"] + RADIUS,
                    m["y0"] + RADIUS,
                    cz,
                    cx,
                    cy,
                )
                src_o = region_contains(o, source)
                src_c = region_contains(c, source)
                rec_o = sum(1 for rec in receivers if region_contains(o, rec))
                rec_c = sum(1 for rec in receivers if region_contains(c, rec))
                source_in_any_o = source_in_any_o or src_o
                source_in_any_c = source_in_any_c or src_c
                receivers_in_any_o += rec_o
                receivers_in_any_c += rec_c
                if src_o:
                    rejected_source_o += 1
                    continue
                if rec_o:
                    rejected_receiver_o += 1
                    continue
                kept_tiles += 1
                commit_points += plan.c_volume

    cta_by_shared_sm = SM120_SHARED_PER_SM // plan.shared_bytes if plan.shared_bytes else 0
    cta_by_warps = SM120_MAX_WARPS_PER_SM // WARPS_PER_CTA
    estimated_cta_per_sm = min(cta_by_shared_sm, cta_by_warps, SM120_MAX_BLOCKS_PER_SM)
    if plan.shared_bytes > SM120_SHARED_PER_BLOCK:
        estimated_cta_per_sm = 0

    return {
        "name": plan.name,
        "implemented_first": plan.implement,
        "M": [plan.mz, plan.mx, plan.my],
        "O": list(plan.o_dims),
        "C": list(plan.c_dims),
        "shared_bytes": plan.shared_bytes,
        "shared_kib": plan.shared_bytes / 1024.0,
        "threads": THREADS_PER_CTA,
        "warps": WARPS_PER_CTA,
        "fits_block_shared_limit": plan.shared_bytes <= SM120_SHARED_PER_BLOCK,
        "max_cta_per_sm_by_shared": cta_by_shared_sm,
        "estimated_cta_per_sm": estimated_cta_per_sm,
        "active_warps_by_estimate": estimated_cta_per_sm * WARPS_PER_CTA,
        "c_points_per_tile": plan.c_volume,
        "c_over_m_ratio": plan.c_volume / plan.m_volume if plan.m_volume else 0.0,
        "m_tile_grid": [len(starts_z), len(starts_x), len(starts_y)],
        "proposed_m_tiles": proposed_tiles,
        "kept_m_tiles": kept_tiles,
        "rejected_source_o_tiles": rejected_source_o,
        "rejected_receiver_o_tiles": rejected_receiver_o,
        "commit_points": commit_points,
        "commit_ratio_vs_core": commit_points / core_points if core_points else 0.0,
        "source_in_any_o": source_in_any_o,
        "source_in_any_c": source_in_any_c,
        "receivers_in_any_o": receivers_in_any_o,
        "receivers_in_any_c": receivers_in_any_c,
    }


def markdown_report(case_dir, manifest, core, core_points, source, receivers, results):
    case_label = case_dir.as_posix()
    lines = [
        "# Core 2-Step Stage-4 Tile Budget",
        "",
        "Generated by `tools/core2step_tile_budget.py`.",
        "",
        "## Case",
        "",
        f"- Case: `{case_label}`",
        f"- Dimensions: `ny={manifest['ny']} nx={manifest['nx']} nz={manifest['nz']} nt={manifest['nt']} npml={manifest['npml']}`",
        f"- Core region: `z=[{core['z0']},{core['z1']}) x=[{core['x0']},{core['x1']}) y=[{core['y0']},{core['y1']})`",
        f"- Core points: `{core_points}`",
        f"- Source index `(z,x,y)`: `{source}`",
        f"- Receiver count: `{len(receivers)}`",
        "",
        "## Hardware Constants",
        "",
        "- `R = 7`.",
        "- `p_next_local` only is staged in shared memory.",
        "- sm_120 budget used here: `128 KiB/SM`, `99 KiB/block`, `48 warps/SM`.",
        "- Source: NVIDIA Blackwell Tuning Guide, CUDA 13.x: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html",
        "",
        "## Budget Table",
        "",
        "| Plan | M | O | C | Shared KiB | C/tile | C/M | Tile grid | Kept M tiles | Commit pts | Commit/core | CTA/SM est | First impl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            "| {name} | {M} | {O} | {C} | {shared_kib:.1f} | {c_points_per_tile} | "
            "{c_over_m_ratio:.4f} | {m_tile_grid} | {kept_m_tiles} | {commit_points} | "
            "{commit_ratio_vs_core:.4f} | {estimated_cta_per_sm} | {implemented_first} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Gate Decision",
            "",
        ]
    )
    first_impl = [item for item in results if item["implemented_first"]]
    passed = [item for item in first_impl if item["commit_ratio_vs_core"] >= 0.10]
    if passed:
        lines.append("At least one first-implementation candidate reaches the `10%` commit-ratio gate.")
    else:
        lines.extend(
            [
                "No first-implementation candidate reaches the `10%` commit-ratio gate.",
                "",
                "Decision: stop before implementing `CUDA3D_CORE_2STEP_FUSED_COMMIT_V2`.",
                "",
                "Reason: with non-overlapping CTA-local M tiles and `R=7`, the eroded C region has too much surface loss. A/D both commit only about `3.2%` of the core points in the meaningful case.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        default="benchmarks/cases/core_2step_meaningful_1gpu",
        help="Case directory containing case_manifest.txt and nav file.",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    manifest, core, core_points, source, receivers = parse_case(case_dir)
    results = [analyze_plan(plan, core, core_points, source, receivers) for plan in PLANS]

    report = {
        "case_dir": str(case_dir),
        "radius": RADIUS,
        "core_region": core,
        "core_points": core_points,
        "source_zyx": source,
        "receiver_count": len(receivers),
        "sm120": {
            "shared_per_sm": SM120_SHARED_PER_SM,
            "shared_per_block": SM120_SHARED_PER_BLOCK,
            "max_warps_per_sm": SM120_MAX_WARPS_PER_SM,
            "threads_per_cta": THREADS_PER_CTA,
        },
        "results": results,
        "first_impl_gate_pass": any(
            item["implemented_first"] and item["commit_ratio_vs_core"] >= 0.10
            for item in results
        ),
    }

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            markdown_report(case_dir, manifest, core, core_points, source, receivers, results),
            encoding="utf-8",
        )

    print(f"case={case_dir}")
    print(f"core_points={core_points}")
    print(f"source_zyx={source}")
    print(f"receiver_count={len(receivers)}")
    print("plan M O C shared_KiB C_per_tile C_over_M tile_grid kept_tiles commit_points commit_ratio CTA_per_SM first_impl")
    for item in results:
        print(
            "{name} {M} {O} {C} {shared_kib:.1f} {c_points_per_tile} "
            "{c_over_m_ratio:.4f} {m_tile_grid} {kept_m_tiles} {commit_points} "
            "{commit_ratio_vs_core:.4f} {estimated_cta_per_sm} {implemented_first}".format(
                **item
            )
        )
    if report["first_impl_gate_pass"]:
        print("GATE=PASS")
    else:
        print("GATE=STOP commit_ratio_lt_10_percent")


if __name__ == "__main__":
    main()
