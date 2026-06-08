#!/usr/bin/env python3
"""Source/receiver aware gate for CUDA3D temporal pipeline ideas."""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShotGeometry:
    shot: int
    source_index_yxz: list[int]
    receiver_count: int
    receiver_min_yx: list[int]
    receiver_max_yx: list[int]
    domain_origin_yx: list[int]
    domain_size_yx: list[int]
    n3n2n1: list[int]
    core_lo_zxy: list[int]
    core_hi_zxy: list[int]
    k2_deep_lo_zxy: list[int]
    k2_deep_hi_zxy: list[int]
    core_points: int
    k2_deep_points: int
    k2_deep_share: float
    source_footprint_zxy: list[list[int]]
    source_influence_zxy: list[list[int]]
    source_influence_overlaps_k2_deep: bool
    receiver_footprint_zxy: list[list[int]]
    receiver_footprint_overlaps_k2_deep: bool


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
            return {"dy": float(dy), "dx": float(dx), "dz": float(dz), "npml": int(npml), "xpad": float(xpad)}
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


def intervals_overlap(a: list[int], b: list[int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def box_overlap(a: list[list[int]], b: list[list[int]]) -> bool:
    return all(intervals_overlap(a[i], b[i]) for i in range(3))


def clamp_interval(lo: int, hi: int, n: int) -> list[int]:
    return [max(0, lo), min(n - 1, hi)]


def analyze_shots(case_dir: Path, radius: int, core_pml_margin: int, nbell: int) -> dict[str, Any]:
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

    shot_results: list[ShotGeometry] = []
    total_core = 0
    total_deep = 0
    source_overlap_count = 0
    receiver_overlap_count = 0
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

        core_lo = [npml + core_pml_margin, npml + core_pml_margin, npml + core_pml_margin]
        core_hi = [n1 - npml - core_pml_margin - 1, n2 - npml - core_pml_margin - 1, n3 - npml - core_pml_margin - 1]
        k2_lo = [core_lo[0] + radius, core_lo[1] + radius, core_lo[2] + radius]
        k2_hi = [core_hi[0] - radius, core_hi[1] - radius, core_hi[2] - radius]
        core_z = max(0, core_hi[0] - core_lo[0] + 1)
        core_x = max(0, core_hi[1] - core_lo[1] + 1)
        core_y = max(0, core_hi[2] - core_lo[2] + 1)
        deep_z = max(0, k2_hi[0] - k2_lo[0] + 1)
        deep_x = max(0, k2_hi[1] - k2_lo[1] + 1)
        deep_y = max(0, k2_hi[2] - k2_lo[2] + 1)
        core_points = core_z * core_x * core_y
        deep_points = deep_z * deep_x * deep_y
        total_core += core_points
        total_deep += deep_points

        src_local_y = source[0] - yl + npml
        src_local_x = source[1] - xl + npml
        src_local_z = source[2] + npml
        # nbell=1 plus trilinear +1 support writes a four-cell range.
        src_fp = [
            clamp_interval(src_local_z - nbell, src_local_z + nbell + 1, n1),
            clamp_interval(src_local_x - nbell, src_local_x + nbell + 1, n2),
            clamp_interval(src_local_y - nbell, src_local_y + nbell + 1, n3),
        ]
        src_influence = [
            clamp_interval(src_fp[0][0] - radius, src_fp[0][1] + radius, n1),
            clamp_interval(src_fp[1][0] - radius, src_fp[1][1] + radius, n2),
            clamp_interval(src_fp[2][0] - radius, src_fp[2][1] + radius, n3),
        ]
        deep_box = [[k2_lo[0], k2_hi[0]], [k2_lo[1], k2_hi[1]], [k2_lo[2], k2_hi[2]]]
        source_overlap = box_overlap(src_influence, deep_box)
        if source_overlap:
            source_overlap_count += 1

        rec_local = [[r[0] - yl + npml, r[1] - xl + npml, r[2] + npml] for r in receivers]
        rec_fp = [
            [min(r[2] for r in rec_local), max(r[2] for r in rec_local) + 1],
            [min(r[1] for r in rec_local), max(r[1] for r in rec_local) + 1],
            [min(r[0] for r in rec_local), max(r[0] for r in rec_local) + 1],
        ]
        rec_fp = [clamp_interval(v[0], v[1], n) for v, n in zip(rec_fp, [n1, n2, n3])]
        receiver_overlap = box_overlap(rec_fp, deep_box)
        if receiver_overlap:
            receiver_overlap_count += 1

        shot_results.append(
            ShotGeometry(
                shot=shot,
                source_index_yxz=source,
                receiver_count=len(receivers),
                receiver_min_yx=[min(ry), min(rx)],
                receiver_max_yx=[max(ry), max(rx)],
                domain_origin_yx=[yl, xl],
                domain_size_yx=[ny_new, nx_new],
                n3n2n1=[n3, n2, n1],
                core_lo_zxy=core_lo,
                core_hi_zxy=core_hi,
                k2_deep_lo_zxy=k2_lo,
                k2_deep_hi_zxy=k2_hi,
                core_points=core_points,
                k2_deep_points=deep_points,
                k2_deep_share=deep_points / core_points if core_points else 0.0,
                source_footprint_zxy=src_fp,
                source_influence_zxy=src_influence,
                source_influence_overlaps_k2_deep=source_overlap,
                receiver_footprint_zxy=rec_fp,
                receiver_footprint_overlaps_k2_deep=receiver_overlap,
            )
        )

    return {
        "case_dir": str(case_dir),
        "ny_nx_nz": [ny, nx, nz],
        "shots": shots,
        "receivers_per_shot": rec_per_shot,
        "spacing_dy_dx_dz": [dy, dx, dz],
        "npml": npml,
        "xpad": xpad,
        "xypad_grid_points": xypad,
        "radius": radius,
        "nbell": nbell,
        "total_core_points": total_core,
        "total_k2_deep_points": total_deep,
        "aggregate_k2_deep_share": total_deep / total_core if total_core else 0.0,
        "source_overlap_shots": source_overlap_count,
        "receiver_overlap_shots": receiver_overlap_count,
        "shots_detail": [asdict(item) for item in shot_results],
    }


def render_markdown(result: dict[str, Any], temporal_model: dict[str, Any] | None) -> str:
    lines = [
        "# Source-Aware Swept/Wavefront Temporal Model",
        "",
        "## Case",
        "",
        f"- case_dir: `{result['case_dir']}`",
        f"- logical ny/nx/nz: `{result['ny_nx_nz'][0]}/{result['ny_nx_nz'][1]}/{result['ny_nx_nz'][2]}`",
        f"- shots/receivers_per_shot: `{result['shots']}/{result['receivers_per_shot']}`",
        f"- npml/radius/nbell: `{result['npml']}/{result['radius']}/{result['nbell']}`",
        f"- xpad: `{result['xpad']}` = `{result['xypad_grid_points']}` grid points",
        f"- aggregate K=2 deep-core share across shot-local subdomains: `{result['aggregate_k2_deep_share']:.2%}`",
        f"- source influence overlaps K=2 deep core in `{result['source_overlap_shots']}` shots",
        f"- receiver footprint overlaps K=2 deep core in `{result['receiver_overlap_shots']}` shots",
        "",
        "## Shot Table",
        "",
        "| shot | domain y/x | K2 deep share | src influence z/x/y | src overlaps deep | rec footprint z/x/y | rec overlaps deep |",
        "| ---: | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    for shot in result["shots_detail"]:
        lines.append(
            f"| {shot['shot']} | {shot['domain_size_yx'][0]}x{shot['domain_size_yx'][1]} | "
            f"{shot['k2_deep_share']:.2%} | {shot['source_influence_zxy']} | "
            f"{shot['source_influence_overlaps_k2_deep']} | {shot['receiver_footprint_zxy']} | "
            f"{shot['receiver_footprint_overlaps_k2_deep']} |"
        )

    lines.extend(
        [
            "",
            "## Schedule Findings",
            "",
            "- Source injection is shallow enough that its radius-7 influence does not overlap the K=2 deep-core region in this benchmark.",
            "- Receiver extraction is also shallow and does not overlap K=2 deep core.",
            "- Therefore source/receiver placement does not kill temporal blocking for this benchmark.",
            "- The remaining blocker is still ownership/synchronization of `p(t+1)` and the byte cost of computing `p_mid` halos.",
            "",
            "## Gate",
            "",
            "- verdict: `stop_swept_wavefront_cuda_prototype`",
            "- reason: source/receiver are compatible, but no implementable swept/wavefront schedule yet beats the Phase 4.1 byte/sync gate.",
        ]
    )
    if temporal_model:
        gate = temporal_model.get("gate", {})
        lines.extend(
            [
                "",
                "## Linked Phase 4.1 Byte Gate",
                "",
                f"- direct temporal verdict: `{gate.get('verdict')}`",
                f"- ideal no-dup sampled-main speedup: `{temporal_model['core_byte_model']['ideal_k2_sampled_main_speedup']:.3f}x`",
                f"- CTA-local byte ratio range: `{temporal_model['gate'].get('cta_local_pair_byte_ratio_range_vs_baseline', 'see summary')}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="benchmarks/cases/perf_1gpu_6shots")
    parser.add_argument("--radius", type=int, default=7)
    parser.add_argument("--core-pml-margin", type=int, default=4)
    parser.add_argument("--nbell", type=int, default=1)
    parser.add_argument("--temporal-model-json", default="reports/day_20260608/temporal_pipeline_model.json")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case)
    if not case_dir.is_absolute():
        case_dir = root / case_dir
    result = analyze_shots(case_dir, args.radius, args.core_pml_margin, args.nbell)
    temporal_path = Path(args.temporal_model_json)
    if not temporal_path.is_absolute():
        temporal_path = root / temporal_path
    temporal = json.loads(temporal_path.read_text(encoding="utf-8")) if temporal_path.exists() else None
    result["gate"] = {
        "verdict": "stop_swept_wavefront_cuda_prototype",
        "source_receiver_compatible": result["source_overlap_shots"] == 0 and result["receiver_overlap_shots"] == 0,
        "reason": "source/receiver placement is compatible with K=2 deep core, but the remaining p_mid ownership and halo-duplication gates still fail",
        "next_allowed_work": "look for a non-duplicating p_mid ownership mechanism or switch away from temporal blocking",
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_markdown(result, temporal)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(md)


if __name__ == "__main__":
    main()
