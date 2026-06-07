#!/usr/bin/env python3
"""Budget PML z-face shared-tile VP fusion before writing CUDA code.

The goal is to answer one narrow gate question: can a CTA-local shared
pressure tile plausibly fit on the target GPU while covering enough z-face
PML work to justify a debug prototype?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CANDIDATES = (
    ("S1", 8, 16, 16, 256),
    ("S2", 12, 16, 12, 256),
    ("S3", 8, 24, 12, 256),
    ("S4", 12, 12, 12, 256),
)


@dataclass(frozen=True)
class CaseGeometry:
    case_dir: str
    ny: int
    nx: int
    nz: int
    nt: int
    shots: int
    receivers_per_shot: int
    npml: int
    xpad: float
    itop: int
    n3: int
    n2: int
    n1: int
    core_margin: int
    core_y_len: int
    core_x_len: int
    core_z_len: int
    total_points: int
    core_points: int
    pml_points_estimate: int
    zface_points_estimate: int
    zface_pml_coverage: float


@dataclass(frozen=True)
class TileBudget:
    name: str
    out_z: int
    out_x: int
    out_y: int
    threads: int
    halo: int
    velocity_radius: int
    composed_pressure_radius: int
    shared_z: int
    shared_x: int
    shared_y: int
    shared_floats: int
    shared_bytes: int
    output_points: int
    outputs_per_thread: float
    shared_p_loads_per_output: float
    direct_xy_second_derivative_loads_per_output_est: int
    p_load_reduction_vs_direct_xy_est: float
    saved_velocity_global_bytes_per_output_est: int
    fits_optin_shared_memory: bool
    estimated_blocks_per_sm_by_shared: int
    verdict: str


def parse_kv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
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


def parse_input_geometry(path: Path, ny: int, nx: int, nz: int) -> dict[str, float | int]:
    """Parse the compact numeric tail of the legacy input file.

    The benchmark input is line-oriented. After the velocity file name, the
    numeric block contains ny/nx/nz, spacing, npml, xpad, itop, shots, and
    gpu count. We anchor on the ny/nx/nz triple to avoid hard-coding line
    numbers.
    """

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    for i in range(0, max(0, len(lines) - 11)):
        if _as_int(lines[i]) != ny:
            continue
        if _as_int(lines[i + 1]) != nx or _as_int(lines[i + 2]) != nz:
            continue
        dy = _as_float(lines[i + 3])
        dx = _as_float(lines[i + 4])
        dz = _as_float(lines[i + 5])
        npml = _as_int(lines[i + 6])
        xpad = _as_float(lines[i + 8])
        itop = _as_int(lines[i + 9])
        if None in (dy, dx, dz, npml, xpad, itop):
            continue
        return {
            "dy": float(dy),
            "dx": float(dx),
            "dz": float(dz),
            "npml": int(npml),
            "xpad": float(xpad),
            "itop": int(itop),
        }
    raise ValueError(f"Could not parse ny/nx/nz anchored geometry from {path}")


def load_geometry(case_dir: Path, core_margin: int) -> CaseGeometry:
    manifest = parse_kv(case_dir / "case_manifest.txt")
    if not manifest:
        raise FileNotFoundError(f"missing case_manifest.txt in {case_dir}")

    ny = int(manifest["ny"])
    nx = int(manifest["nx"])
    nz = int(manifest["nz"])
    nt = int(manifest.get("nt", 0))
    shots = int(manifest.get("shots", 0))
    receivers = int(manifest.get("receivers_per_shot", 0))
    input_name = manifest.get("input")
    if not input_name:
        raise ValueError(f"manifest in {case_dir} does not define input=")

    input_geo = parse_input_geometry(case_dir / input_name, ny, nx, nz)
    npml = int(input_geo["npml"])
    xpad = float(input_geo["xpad"])
    itop = int(input_geo["itop"])

    n3 = ny + 2 * npml
    n2 = nx + 2 * npml
    n1 = nz + 2 * npml
    core_y_len = max(0, n3 - 2 * (npml + core_margin))
    core_x_len = max(0, n2 - 2 * (npml + core_margin))
    core_z_len = max(0, n1 - 2 * (npml + core_margin))
    total = n1 * n2 * n3
    core = core_z_len * core_x_len * core_y_len
    pml_est = max(0, total - core)
    zface = 2 * npml * core_x_len * core_y_len
    coverage = float(zface / pml_est) if pml_est else 0.0

    return CaseGeometry(
        case_dir=str(case_dir),
        ny=ny,
        nx=nx,
        nz=nz,
        nt=nt,
        shots=shots,
        receivers_per_shot=receivers,
        npml=npml,
        xpad=xpad,
        itop=itop,
        n3=n3,
        n2=n2,
        n1=n1,
        core_margin=core_margin,
        core_y_len=core_y_len,
        core_x_len=core_x_len,
        core_z_len=core_z_len,
        total_points=total,
        core_points=core,
        pml_points_estimate=pml_est,
        zface_points_estimate=zface,
        zface_pml_coverage=coverage,
    )


def candidate_budget(
    name: str,
    out_z: int,
    out_x: int,
    out_y: int,
    threads: int,
    halo: int,
    velocity_radius: int,
    composed_pressure_radius: int,
    max_block_smem: int,
    smem_per_sm: int,
) -> TileBudget:
    shared_z = out_z + 2 * halo
    shared_x = out_x + 2 * halo
    shared_y = out_y + 2 * halo
    shared_floats = shared_z * shared_x * shared_y
    shared_bytes = shared_floats * 4
    output_points = out_z * out_x * out_y
    outputs_per_thread = output_points / threads
    shared_loads = shared_floats / output_points

    direct_xy_loads = 30
    p_load_reduction = 1.0 - (shared_loads / direct_xy_loads)

    fits = shared_bytes <= max_block_smem
    blocks_by_shared = smem_per_sm // shared_bytes if shared_bytes else 0
    if not fits:
        verdict = "fail: shared memory exceeds opt-in block limit"
    elif blocks_by_shared < 1:
        verdict = "fail: no block fits per SM by shared memory"
    elif output_points < threads:
        verdict = "weak: fewer outputs than threads"
    else:
        verdict = "pass: debug-prototype budget candidate"

    return TileBudget(
        name=name,
        out_z=out_z,
        out_x=out_x,
        out_y=out_y,
        threads=threads,
        halo=halo,
        velocity_radius=velocity_radius,
        composed_pressure_radius=composed_pressure_radius,
        shared_z=shared_z,
        shared_x=shared_x,
        shared_y=shared_y,
        shared_floats=shared_floats,
        shared_bytes=shared_bytes,
        output_points=output_points,
        outputs_per_thread=outputs_per_thread,
        shared_p_loads_per_output=shared_loads,
        direct_xy_second_derivative_loads_per_output_est=direct_xy_loads,
        p_load_reduction_vs_direct_xy_est=p_load_reduction,
        saved_velocity_global_bytes_per_output_est=72,
        fits_optin_shared_memory=fits,
        estimated_blocks_per_sm_by_shared=blocks_by_shared,
        verdict=verdict,
    )


def render_markdown(geometry: CaseGeometry, budgets: Iterable[TileBudget], assumptions: dict[str, object]) -> str:
    rows = []
    for item in budgets:
        rows.append(
            "| {name} | {out_z}x{out_x}x{out_y} | {threads} | {shared_z}x{shared_x}x{shared_y} | "
            "{shared_bytes} | {output_points} | {shared_p_loads_per_output:.3f} | "
            "{p_load_reduction_vs_direct_xy_est:.2%} | {estimated_blocks_per_sm_by_shared} | {verdict} |".format(
                **asdict(item)
            )
        )

    assumptions_json = json.dumps(assumptions, indent=2, sort_keys=True)
    return "\n".join(
        [
            "# PML Z-Face Shared-Tile VP Budget",
            "",
            "## Case Geometry",
            "",
            f"- case_dir: `{geometry.case_dir}`",
            f"- logical model: ny/nx/nz = `{geometry.ny}/{geometry.nx}/{geometry.nz}`",
            f"- padded domain estimate: n3/n2/n1 = `{geometry.n3}/{geometry.n2}/{geometry.n1}`",
            f"- npml/core_margin: `{geometry.npml}/{geometry.core_margin}`",
            f"- nt/shots/receivers_per_shot: `{geometry.nt}/{geometry.shots}/{geometry.receivers_per_shot}`",
            f"- estimated total points: `{geometry.total_points}`",
            f"- estimated core points: `{geometry.core_points}`",
            f"- estimated PML points: `{geometry.pml_points_estimate}`",
            f"- estimated pure z-face PML points: `{geometry.zface_points_estimate}`",
            f"- z-face share of estimated PML pressure work: `{geometry.zface_pml_coverage:.2%}`",
            "",
            "## Tile Candidates",
            "",
            "| candidate | output z/x/y | threads | shared z/x/y | shared bytes | outputs | shared p loads/output | p-load reduction vs direct xy est | blocks/SM by shared | verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *rows,
            "",
            "## Assumptions",
            "",
            "```json",
            assumptions_json,
            "```",
            "",
            "## Gate Read",
            "",
            "- All candidates are design-only until Nsight Compute evidence shows the z-face path is still memory-traffic limited.",
            "- Passing this budget only means the shared pressure tile fits under the configured opt-in shared-memory limit.",
            "- The estimate intentionally treats source/receiver exclusion as unknown because `.nav` is a binary float file; a real kernel must either reject those tiles at runtime or prove they cannot appear in the fused z-face region.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="benchmarks/cases/perf_1gpu_6shots")
    parser.add_argument("--core-margin", type=int, default=4)
    parser.add_argument("--velocity-radius", type=int, default=4)
    parser.add_argument("--composed-pressure-radius", type=int, default=7)
    parser.add_argument("--halo", type=int, default=7)
    parser.add_argument("--max-block-smem", type=int, default=99_000)
    parser.add_argument("--smem-per-sm", type=int, default=128 * 1024)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    case_dir = (root / args.case).resolve() if not Path(args.case).is_absolute() else Path(args.case)
    geometry = load_geometry(case_dir, args.core_margin)

    budgets = [
        candidate_budget(
            name,
            out_z,
            out_x,
            out_y,
            threads,
            args.halo,
            args.velocity_radius,
            args.composed_pressure_radius,
            args.max_block_smem,
            args.smem_per_sm,
        )
        for name, out_z, out_x, out_y, threads in DEFAULT_CANDIDATES
    ]
    assumptions = {
        "halo": args.halo,
        "halo_reason": "composed p-current footprint for local velocity-gradient plus pressure-divergence path",
        "velocity_radius": args.velocity_radius,
        "composed_pressure_radius": args.composed_pressure_radius,
        "direct_xy_second_derivative_loads_per_output_est": 30,
        "saved_velocity_global_bytes_per_output_est": "2 velocity stores + 16 velocity stencil reads = 72 bytes/output",
        "source_receiver_exclusion": "unknown_binary_nav",
        "max_block_smem_bytes": args.max_block_smem,
        "smem_per_sm_bytes": args.smem_per_sm,
    }
    output = {
        "geometry": asdict(geometry),
        "assumptions": assumptions,
        "budgets": [asdict(item) for item in budgets],
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    md = render_markdown(geometry, budgets, assumptions)
    if args.md_out:
        Path(args.md_out).write_text(md, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(md)


if __name__ == "__main__":
    main()
