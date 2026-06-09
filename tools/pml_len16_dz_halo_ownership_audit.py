#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


PML_TILE_MASK_Z = 1
PML_TILE_MASK_X = 2
PML_TILE_MASK_Y = 4
PML_TILE_MASK_MIXED = 8


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def interval_len(lo: int, hi: int) -> int:
    return max(0, hi - lo)


def tile_inside(z0, x0, y0, bz, bx, by, n1, n2, n3, zlo, zhi, xlo, xhi, ylo, yhi):
    z1 = min(z0 + bz, n1)
    x1 = min(x0 + bx, n2)
    y1 = min(y0 + by, n3)
    return z0 >= zlo and z1 <= zhi and x0 >= xlo and x1 <= xhi and y0 >= ylo and y1 <= yhi


def tile_mask(z0, x0, y0, bz, bx, by, n1, n2, n3, npml):
    z1 = min(z0 + bz, n1)
    x1 = min(x0 + bx, n2)
    y1 = min(y0 + by, n3)
    axes = 0
    mask = 0
    if z0 < npml or z1 > n1 - npml:
        mask |= PML_TILE_MASK_Z
        axes += 1
    if x0 < npml or x1 > n2 - npml:
        mask |= PML_TILE_MASK_X
        axes += 1
    if y0 < npml or y1 > n3 - npml:
        mask |= PML_TILE_MASK_Y
        axes += 1
    if axes > 1:
        mask |= PML_TILE_MASK_MIXED
    return mask


def pml_z_state(gtid3, gtid2, gtid1, n3, n2, n1, npml):
    if gtid1 < 0 or gtid1 >= n1 or gtid2 < 0 or gtid2 >= n2 or gtid3 < 0 or gtid3 >= n3:
        return None
    if gtid1 < npml:
        return (gtid3, gtid2, gtid1)
    if gtid1 >= n1 - npml:
        return (gtid3, gtid2, npml + (gtid1 - (n1 - npml)))
    return None


def build_pressure_tiles(n3, n2, n1, npml, bz, bx, by, margin):
    grid1 = ceil_div(n1, bz)
    grid2 = ceil_div(n2, bx)
    grid3 = ceil_div(n3, by)
    core1_lo = npml + margin
    core2_lo = npml + margin
    core3_lo = npml + margin
    core1_hi = n1 - npml - margin
    core2_hi = n2 - npml - margin
    core3_hi = n3 - npml - margin
    tiles = []
    for yy in range(grid3):
        y0 = yy * by
        for xx in range(grid2):
            x0 = xx * bx
            for zz in range(grid1):
                z0 = zz * bz
                skip = (
                    core1_hi > core1_lo
                    and core2_hi > core2_lo
                    and core3_hi > core3_lo
                    and tile_inside(z0, x0, y0, bz, bx, by, n1, n2, n3, core1_lo, core1_hi, core2_lo, core2_hi, core3_lo, core3_hi)
                )
                if not skip:
                    tiles.append((z0, x0, y0, tile_mask(z0, x0, y0, bz, bx, by, n1, n2, n3, npml)))
    return tiles


def is_pressure_len16(tile, n3, n2, n1, npml, bz, bx, by, margin):
    z0, x0, y0, _mask = tile
    z1 = min(z0 + bz, n1)
    x1 = min(x0 + bx, n2)
    y1 = min(y0 + by, n3)
    core1_lo = npml + margin
    core2_lo = npml + margin
    core3_lo = npml + margin
    core1_hi = n1 - npml - margin
    core2_hi = n2 - npml - margin
    core3_hi = n3 - npml - margin
    if x1 - x0 != bx or y1 - y0 != by:
        return False
    if not (x0 >= core2_lo and x1 <= core2_hi and y0 >= core3_lo and y1 <= core3_hi):
        return False
    core_overlap = interval_len(max(z0, core1_lo), min(z1, core1_hi))
    active_z_len = (z1 - z0) - core_overlap
    return active_z_len == 16


def active_z_range_for_line(tile, gtid2, gtid3, n3, n2, n1, npml, bz, margin):
    z0, _x0, _y0, _mask = tile
    central_z0 = z0
    central_z1 = min(z0 + bz, n1)
    core1_lo = npml + margin
    core2_lo = npml + margin
    core3_lo = npml + margin
    core1_hi = n1 - npml - margin
    core2_hi = n2 - npml - margin
    core3_hi = n3 - npml - margin
    active_lo = central_z0
    active_hi = central_z1
    xy_in_domain = 0 <= gtid2 < n2 and 0 <= gtid3 < n3
    if not xy_in_domain or central_z1 <= central_z0:
        return active_lo, active_lo
    xy_in_core = core2_lo <= gtid2 < core2_hi and core3_lo <= gtid3 < core3_hi
    if xy_in_core and central_z0 < core1_lo:
        active_hi = min(central_z1, core1_lo)
    elif xy_in_core and central_z1 > core1_hi:
        active_lo = max(central_z0, core1_hi)
    elif xy_in_core:
        active_hi = active_lo
    return active_lo, active_hi


def pressure_tile_zmem_sets(tile, n3, n2, n1, npml, bz, bx, by, margin):
    reads = set()
    writes = set()
    z0, x0, y0, _mask = tile
    for ly in range(by):
        gtid3 = y0 + ly
        for lx in range(bx):
            gtid2 = x0 + lx
            active_lo, active_hi = active_z_range_for_line(tile, gtid2, gtid3, n3, n2, n1, npml, bz, margin)
            if active_hi <= active_lo:
                continue
            for cache_z in range(active_lo - 4, active_hi + 3):
                state = pml_z_state(gtid3, gtid2, cache_z, n3, n2, n1, npml)
                if state is None:
                    continue
                reads.add(state)
                if active_lo <= cache_z < active_hi:
                    writes.add(state)
    return reads, writes


def parse_case_dims(input_path: Path):
    rows = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        ny = int(rows[22])
        nx = int(rows[23])
        nz = int(rows[24])
        npml = int(rows[28])
    except (IndexError, ValueError) as exc:
        raise SystemExit(f"cannot parse dimensions from {input_path}: {exc}") from exc
    return ny, nx, nz, npml


def summarize_case(input_path: Path, bz: int, bx: int, by: int, margin: int):
    ny, nx, nz, npml = parse_case_dims(input_path)
    n3 = ny + 2 * npml
    n2 = nx + 2 * npml
    n1 = nz + 2 * npml
    tiles = build_pressure_tiles(n3, n2, n1, npml, bz, bx, by, margin)
    len16 = [t for t in tiles if is_pressure_len16(t, n3, n2, n1, npml, bz, bx, by, margin)]
    residual = [t for t in tiles if not is_pressure_len16(t, n3, n2, n1, npml, bz, bx, by, margin)]

    len16_reads = set()
    len16_writes = set()
    for tile in len16:
        reads, writes = pressure_tile_zmem_sets(tile, n3, n2, n1, npml, bz, bx, by, margin)
        len16_reads |= reads
        len16_writes |= writes

    residual_reads = set()
    residual_writes = set()
    for tile in residual:
        reads, writes = pressure_tile_zmem_sets(tile, n3, n2, n1, npml, bz, bx, by, margin)
        residual_reads |= reads
        residual_writes |= writes

    len16_halo_reads = len16_reads - len16_writes
    if len(len16) == 0:
        gate = "not_applicable_no_len16"
    elif len(len16_halo_reads) == 0 and not (residual_reads & len16_writes) and not (residual_writes & len16_writes):
        gate = "allow_compact_dz16_old_next_design"
    else:
        gate = "block_compact_dz_old_next_commit"

    return {
        "input": str(input_path),
        "dims": {"ny": ny, "nx": nx, "nz": nz, "npml": npml, "n3": n3, "n2": n2, "n1": n1},
        "tile_shape": {"z": bz, "x": bx, "y": by, "core_margin": margin},
        "pressure_tiles": len(tiles),
        "len16_tiles": len(len16),
        "residual_tiles": len(residual),
        "len16_zmem_read_states": len(len16_reads),
        "len16_zmem_write_states": len(len16_writes),
        "len16_halo_read_states_outside_writes": len(len16_halo_reads),
        "residual_zmem_read_states": len(residual_reads),
        "residual_zmem_write_states": len(residual_writes),
        "residual_reads_len16_written_states": len(residual_reads & len16_writes),
        "residual_writes_len16_written_states": len(residual_writes & len16_writes),
        "len16_reads_residual_written_states": len(len16_reads & residual_writes),
        "gate": gate,
    }


def write_md(results, out_path: Path):
    lines = [
        "# PML Len16 DZ Halo Ownership Audit",
        "",
        "This is a static exact-FP32 ownership audit. It mirrors the pressure PML tile-list split and z-cache state access ranges; it does not run CUDA kernels.",
        "",
        "A case with zero len16 pressure tiles is marked `not_applicable_no_len16`; it can still validate residual fallback, but it does not cover the compact len16 path.",
        "",
        "| case | len16 tiles | residual tiles | len16 write states | len16 halo reads outside writes | residual reads len16 writes | residual writes len16 writes | gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        lines.append(
            f"| `{Path(item['input']).name}` | `{item['len16_tiles']}` | `{item['residual_tiles']}` | "
            f"`{item['len16_zmem_write_states']}` | `{item['len16_halo_read_states_outside_writes']}` | "
            f"`{item['residual_reads_len16_written_states']}` | `{item['residual_writes_len16_written_states']}` | `{item['gate']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "A compact `memory_dz` old/next commit prototype is allowed only for cases with len16 coverage and zero len16 halo reads outside its compact write set plus zero residual overlap with len16-written z-state.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    parser.add_argument("--bz", type=int, default=32)
    parser.add_argument("--bx", type=int, default=4)
    parser.add_argument("--by", type=int, default=2)
    parser.add_argument("--margin", type=int, default=4)
    args = parser.parse_args()

    results = [summarize_case(Path(p), args.bz, args.bx, args.by, args.margin) for p in args.inputs]
    out_json = Path(args.json_out)
    out_md = Path(args.md_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    write_md(results, out_md)
    print(out_md)


if __name__ == "__main__":
    main()
