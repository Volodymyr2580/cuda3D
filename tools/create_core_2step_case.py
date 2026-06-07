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
        y_term = 0.025 * math.sin(iy / 10.0)
        for ix in range(nx):
            x_term = 0.035 * math.cos(ix / 12.0)
            for iz in range(nz):
                depth = iz * dz
                yield 2.0 + 0.35 * depth + x_term + y_term


def build_nav(nx, ny, dx, dy, dz):
    # Keep source/receivers shallow in z so the first strict-interior two-step
    # prototype can exclude source/receiver timing while still running the
    # normal injection and extraction kernels each timestep.
    sx = 0.5 * (nx - 1) * dx
    sy = 0.5 * (ny - 1) * dy
    sz = dz
    offsets = [-0.05, 0.0, 0.05]
    nav = []
    xmin, xmax = 0.05, (nx - 2) * dx
    ymin, ymax = 0.05, (ny - 2) * dy
    for oy in offsets:
        for ox in offsets:
            rx = min(max(sx + ox, xmin), xmax)
            ry = min(max(sy + oy, ymin), ymax)
            nav.extend([sx, sy, sz, rx, ry, dz])
    return nav


def input_text(nav_name, vel_name, trace_count, ny, nx, nz, nt, npml, gpu_count):
    return f"""./d_obs/d_obs_core_2step_shot_
1.
noinput
0
./{nav_name}
ricker1_core_2step
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
0.05
0
6
{gpu_count}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--nz", type=int, default=80)
    parser.add_argument("--nt", type=int, default=6)
    parser.add_argument("--npml", type=int, default=8)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--case-name", default="core_2step_interior_1gpu")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    case_dir = root / "benchmarks" / "cases" / args.case_name
    (case_dir / "d_obs").mkdir(parents=True, exist_ok=True)

    dy = dx = dz = 0.025
    vel_name = f"vel_{args.case_name}_ny{args.ny}_nx{args.nx}_nz{args.nz}.dir"
    nav_name = "nav_core_2step_1shot_9rec_shallow.nav"
    input_name = "input_core_2step_interior_1gpu.in"

    vel = list(velocity_values(args.ny, args.nx, args.nz, dz))
    nav = build_nav(args.nx, args.ny, dx, dy, dz)
    write_float32(case_dir / vel_name, vel)
    write_float32(case_dir / nav_name, nav)
    (case_dir / input_name).write_text(
        input_text(nav_name, vel_name, len(nav) // 6, args.ny, args.nx, args.nz, args.nt, args.npml, args.gpu_count),
        encoding="utf-8",
    )

    core_stencil_radius = 7
    core_pml_margin = 4
    margin = 2 * core_stencil_radius
    nbz = args.nz + 2 * args.npml
    nbx = args.nx + 2 * args.npml
    nby = args.ny + 2 * args.npml
    safe = {
        "z0": args.npml + core_pml_margin + margin,
        "z1": nbz - args.npml - core_pml_margin - margin,
        "x0": args.npml + core_pml_margin + margin,
        "x1": nbx - args.npml - core_pml_margin - margin,
        "y0": args.npml + core_pml_margin + margin,
        "y1": nby - args.npml - core_pml_margin - margin,
    }

    manifest = [
        f"case={args.case_name}",
        f"ny={args.ny}",
        f"nx={args.nx}",
        f"nz={args.nz}",
        f"nt={args.nt}",
        f"npml={args.npml}",
        "shots=1",
        "receivers_per_shot=9",
        "gpu_count=1",
        f"velocity={vel_name}",
        f"nav={nav_name}",
        f"input={input_name}",
        "source_z_index=1",
        "receiver_z_index=1",
        "source_receiver_policy=shallow_z_outside_default_strict_interior",
        f"default_margin={margin}",
        "default_region="
        f"z:[{safe['z0']},{safe['z1']}),"
        f"x:[{safe['x0']},{safe['x1']}),"
        f"y:[{safe['y0']},{safe['y1']})",
    ]
    (case_dir / "case_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(case_dir)
    print(f"input={input_name}")
    print(f"default_region=z:{safe['z0']}:{safe['z1']},x:{safe['x0']}:{safe['x1']},y:{safe['y0']}:{safe['y1']}")


if __name__ == "__main__":
    main()
