#!/usr/bin/env python3
import argparse
import math
import struct
from pathlib import Path


def write_float32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))


def velocity_values(ny, nx, nz, dz):
    for iy in range(ny):
        y_term = 0.018 * math.sin(iy / 17.0)
        for ix in range(nx):
            x_term = 0.024 * math.cos(ix / 19.0)
            for iz in range(nz):
                depth = iz * dz
                yield 2.05 + 0.28 * depth + x_term + y_term


def build_nav(dx, dy, dz):
    # Keep acquisition shallow and near one corner. With xpad large enough to
    # keep the full model, the central fused region stays source/receiver free.
    sx = 0.30
    sy = 0.30
    sz = dz
    offsets = [-0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075]
    nav = []
    for oy in offsets:
        for ox in offsets:
            nav.extend([sx, sy, sz, sx + ox, sy + oy, dz])
    return nav


def input_text(nav_name, vel_name, trace_count, ny, nx, nz, nt, npml, xpad, gpu_count):
    return f"""./d_obs/d_obs_core_2step_meaningful_shot_
1.
noinput
0
./{nav_name}
ricker1_core_2step_meaningful
noinput
0
./tmut_zz/
0
./bmut_zz/
out.dir
{trace_count}
0.0
100.
0.0
2
0.002
{nt}
./{vel_name}
0
3.2
{ny}
{nx}
{nz}
0.025
0.025
0.025
{npml}
0.1
{xpad}
0
6
{gpu_count}
"""


def index_from_coord(coord, spacing):
    return int(math.floor(coord / spacing))


def region_stats(ny, nx, nz, npml, core_pml_margin, radius, nav, dx, dy):
    nby = ny + 2 * npml
    nbx = nx + 2 * npml
    nbz = nz + 2 * npml
    core = {
        "z0": npml + core_pml_margin,
        "z1": nbz - npml - core_pml_margin,
        "x0": npml + core_pml_margin,
        "x1": nbx - npml - core_pml_margin,
        "y0": npml + core_pml_margin,
        "y1": nby - npml - core_pml_margin,
    }
    margin = 2 * radius
    fused = {
        "z0": core["z0"] + margin,
        "z1": core["z1"] - margin,
        "x0": core["x0"] + margin,
        "x1": core["x1"] - margin,
        "y0": core["y0"] + margin,
        "y1": core["y1"] - margin,
    }

    core_points = (
        (core["z1"] - core["z0"])
        * (core["x1"] - core["x0"])
        * (core["y1"] - core["y0"])
    )
    fused_points = max(0, fused["z1"] - fused["z0"]) * max(0, fused["x1"] - fused["x0"]) * max(0, fused["y1"] - fused["y0"])
    ratio = fused_points / core_points if core_points else 0.0

    sx = index_from_coord(nav[0], dx)
    sy = index_from_coord(nav[1], dy)
    source_local = (npml + sx, npml + sy)
    source_in = (
        fused["x0"] <= source_local[0] < fused["x1"]
        and fused["y0"] <= source_local[1] < fused["y1"]
    )

    receivers_in = 0
    for i in range(0, len(nav), 6):
        rx = npml + index_from_coord(nav[i + 3], dx)
        ry = npml + index_from_coord(nav[i + 4], dy)
        if fused["x0"] <= rx < fused["x1"] and fused["y0"] <= ry < fused["y1"]:
            receivers_in += 1

    return core, fused, core_points, fused_points, ratio, source_in, receivers_in


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=160)
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--nz", type=int, default=96)
    parser.add_argument("--nt", type=int, default=501)
    parser.add_argument("--npml", type=int, default=12)
    parser.add_argument("--xpad", type=float, default=4.0)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--case-name", default="core_2step_meaningful_1gpu")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    case_dir = root / "benchmarks" / "cases" / args.case_name
    (case_dir / "d_obs").mkdir(parents=True, exist_ok=True)

    dy = dx = dz = 0.025
    vel_name = f"vel_{args.case_name}_ny{args.ny}_nx{args.nx}_nz{args.nz}.dir"
    nav_name = "nav_core_2step_meaningful_1shot_49rec_corner.nav"
    input_name = "input_core_2step_meaningful_1gpu.in"

    vel = list(velocity_values(args.ny, args.nx, args.nz, dz))
    nav = build_nav(dx, dy, dz)
    write_float32(case_dir / vel_name, vel)
    write_float32(case_dir / nav_name, nav)
    (case_dir / input_name).write_text(
        input_text(nav_name, vel_name, len(nav) // 6, args.ny, args.nx, args.nz, args.nt, args.npml, args.xpad, args.gpu_count),
        encoding="utf-8",
    )

    radius = 7
    core_pml_margin = 4
    core, fused, core_points, fused_points, ratio, source_in, receivers_in = region_stats(
        args.ny, args.nx, args.nz, args.npml, core_pml_margin, radius, nav, dx, dy
    )

    manifest = [
        f"case={args.case_name}",
        f"ny={args.ny}",
        f"nx={args.nx}",
        f"nz={args.nz}",
        f"nt={args.nt}",
        f"npml={args.npml}",
        f"xpad={args.xpad}",
        "shots=1",
        "receivers_per_shot=49",
        "gpu_count=1",
        f"velocity={vel_name}",
        f"nav={nav_name}",
        f"input={input_name}",
        f"core_points={core_points}",
        f"fused_eligible_points={fused_points}",
        f"eligible_ratio={ratio:.6f}",
        f"source_in_fused_region={'yes' if source_in else 'no'}",
        f"receivers_in_fused_region={receivers_in}",
        f"core_region=z:[{core['z0']},{core['z1']}),x:[{core['x0']},{core['x1']}),y:[{core['y0']},{core['y1']})",
        f"fused_region=z:[{fused['z0']},{fused['z1']}),x:[{fused['x0']},{fused['x1']}),y:[{fused['y0']},{fused['y1']})",
    ]
    (case_dir / "case_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print(case_dir)
    print(f"input={input_name}")
    print(f"core_points={core_points}")
    print(f"fused_eligible_points={fused_points}")
    print(f"eligible_ratio={ratio:.6f}")
    print(f"source_in_fused_region={'yes' if source_in else 'no'}")
    print(f"receivers_in_fused_region={receivers_in}")
    print(f"fused_region={fused['z0']}:{fused['z1']},{fused['x0']}:{fused['x1']},{fused['y0']}:{fused['y1']}")


if __name__ == "__main__":
    main()
