#!/usr/bin/env python3
import math
import struct
from pathlib import Path


def write_float32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "bench_smoke"
    (out / "d_obs").mkdir(parents=True, exist_ok=True)

    ny, nx, nz = 48, 48, 48
    dy = dx = dz = 0.025
    nt = 51
    dt = 0.002

    vel = []
    for iy in range(ny):
        for ix in range(nx):
            for iz in range(nz):
                depth = iz * dz
                lateral = 0.05 * math.sin(ix / 8.0) * math.cos(iy / 9.0)
                vel.append(2.0 + 0.45 * depth + lateral)

    vel_name = "vel_smoke_ny48_nx48_nz48.dir"
    write_float32(out / vel_name, vel)

    shots = [(0.30, 0.30), (0.60, 0.55), (0.90, 0.80)]
    offsets = [-0.10, -0.05, 0.0, 0.05, 0.10]
    nav = []
    for sx, sy in shots:
        for oy in offsets:
            for ox in offsets:
                rx = min(max(sx + ox, 0.05), (nx - 2) * dx)
                ry = min(max(sy + oy, 0.05), (ny - 2) * dy)
                nav.extend([sx, sy, dz, rx, ry, dz])

    nav_name = "nav_smoke_3shots_25rec.nav"
    write_float32(out / nav_name, nav)

    input_text = f"""./d_obs/d_obs_smoke_shot_
1.
noinput
0
./{nav_name}
ricker1_smoke
noinput
0
./tmut_zz/
0
./bmut_zz/
out.dir
{len(nav) // 6}
0.0
100.
0.0
2
{dt}
{nt}
./{vel_name}
0
3.0
{ny}
{nx}
{nz}
{dy}
{dx}
{dz}
8
0.1
0.05
0
6
1
"""
    (out / "input_smoke.in").write_text(input_text, encoding="utf-8")

    print(out)
    print(f"velocity={vel_name} floats={len(vel)}")
    print(f"nav={nav_name} traces={len(nav) // 6}")
    print("input=input_smoke.in")


if __name__ == "__main__":
    main()
