#!/usr/bin/env python3
import argparse
import math
import struct
from pathlib import Path


CASES = {
    "correctness": {
        "ny": 96,
        "nx": 96,
        "nz": 64,
        "nt": 201,
        "shots": 6,
        "receiver_grid": 7,
        "receiver_spacing": 0.025,
        "xpad": 0.05,
        "gpu_count": 1,
        "vmax": 3.2,
        "npml": 8,
    },
    "perf_3gpu": {
        "ny": 384,
        "nx": 384,
        "nz": 95,
        "nt": 1501,
        "shots": 9,
        "receiver_grid": 21,
        "receiver_spacing": 0.25,
        "xpad": 0.5,
        "gpu_count": 3,
        "vmax": 4.0,
        "npml": 12,
    },
    "profile_1gpu": {
        "ny": 384,
        "nx": 384,
        "nz": 95,
        "nt": 501,
        "shots": 1,
        "receiver_grid": 21,
        "receiver_spacing": 0.25,
        "xpad": 0.5,
        "gpu_count": 1,
        "vmax": 4.0,
        "npml": 12,
    },
}


def write_float32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))


def velocity_values(ny, nx, nz, dz):
    for iy in range(ny):
        y_term = 0.03 * math.sin(iy / 11.0)
        for ix in range(nx):
            x_term = 0.04 * math.cos(ix / 13.0)
            for iz in range(nz):
                depth = iz * dz
                yield 1.8 + 0.55 * depth + x_term + y_term


def shot_positions(count, nx, ny, dx, dy):
    if count == 1:
        return [(0.5 * (nx - 1) * dx, 0.5 * (ny - 1) * dy)]

    rows = int(math.floor(math.sqrt(count)))
    cols = int(math.ceil(count / rows))
    x0, x1 = 0.25 * (nx - 1) * dx, 0.75 * (nx - 1) * dx
    y0, y1 = 0.25 * (ny - 1) * dy, 0.75 * (ny - 1) * dy
    positions = []
    for row in range(rows):
        for col in range(cols):
            if len(positions) >= count:
                break
            x = x0 if cols == 1 else x0 + (x1 - x0) * col / (cols - 1)
            y = y0 if rows == 1 else y0 + (y1 - y0) * row / (rows - 1)
            positions.append((x, y))
    return positions


def build_nav(config, dx, dy, dz):
    nx, ny = config["nx"], config["ny"]
    half = config["receiver_grid"] // 2
    spacing = config["receiver_spacing"]
    xmin, xmax = 0.05, (nx - 2) * dx
    ymin, ymax = 0.05, (ny - 2) * dy
    nav = []
    for sx, sy in shot_positions(config["shots"], nx, ny, dx, dy):
        for oy in range(-half, half + 1):
            for ox in range(-half, half + 1):
                rx = min(max(sx + ox * spacing, xmin), xmax)
                ry = min(max(sy + oy * spacing, ymin), ymax)
                nav.extend([sx, sy, dz, rx, ry, dz])
    return nav


def input_text(case, config, nav_name, vel_name, trace_count):
    return f"""./d_obs/d_obs_{case}_shot_
1.
noinput
0
./{nav_name}
ricker1_{case}
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
{config["nt"]}
./{vel_name}
0
{config["vmax"]}
{config["ny"]}
{config["nx"]}
{config["nz"]}
0.025
0.025
0.025
{config["npml"]}
0.1
{config["xpad"]}
0
6
{config["gpu_count"]}
"""


def create_case(root, case):
    config = CASES[case]
    case_dir = root / "benchmarks" / "cases" / case
    (case_dir / "d_obs").mkdir(parents=True, exist_ok=True)

    dy = dx = dz = 0.025
    vel_name = f"vel_{case}_ny{config['ny']}_nx{config['nx']}_nz{config['nz']}.dir"
    nav_name = f"nav_{case}_{config['shots']}shots_{config['receiver_grid'] ** 2}rec.nav"
    input_name = f"input_{case}.in"

    vel = list(velocity_values(config["ny"], config["nx"], config["nz"], dz))
    nav = build_nav(config, dx, dy, dz)

    write_float32(case_dir / vel_name, vel)
    write_float32(case_dir / nav_name, nav)
    (case_dir / input_name).write_text(
        input_text(case, config, nav_name, vel_name, len(nav) // 6),
        encoding="utf-8",
    )

    manifest = [
        f"case={case}",
        f"ny={config['ny']}",
        f"nx={config['nx']}",
        f"nz={config['nz']}",
        f"nt={config['nt']}",
        f"shots={config['shots']}",
        f"receivers_per_shot={config['receiver_grid'] ** 2}",
        f"receiver_spacing={config['receiver_spacing']}",
        f"trace_count={len(nav) // 6}",
        f"gpu_count={config['gpu_count']}",
        f"xpad={config['xpad']}",
        f"velocity={vel_name}",
        f"nav={nav_name}",
        f"input={input_name}",
    ]
    (case_dir / "case_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return case_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*CASES.keys(), "all"], default="all")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cases = CASES.keys() if args.case == "all" else [args.case]
    for case in cases:
        case_dir = create_case(root, case)
        print(f"created {case}: {case_dir}")


if __name__ == "__main__":
    main()
