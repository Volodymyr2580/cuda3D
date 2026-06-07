#!/usr/bin/env python3
import argparse
import array
import json
import math
import re
from pathlib import Path


def read_float32(path):
    data = array.array("f")
    with open(path, "rb") as f:
        data.frombytes(f.read())
    return data


def parse_meta(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def compare_float_files(base, cand, rel_tol, abs_tol):
    if base.stat().st_size != cand.stat().st_size:
        return {
            "pass": False,
            "reason": "size_mismatch",
            "baseline_bytes": base.stat().st_size,
            "candidate_bytes": cand.stat().st_size,
        }

    a = read_float32(base)
    b = read_float32(cand)
    if len(a) != len(b):
        return {"pass": False, "reason": "count_mismatch", "baseline_count": len(a), "candidate_count": len(b)}

    finite = True
    sum_diff2 = 0.0
    sum_base2 = 0.0
    max_abs = 0.0
    max_rel = 0.0
    max_index = 0
    base_at_max = 0.0
    cand_at_max = 0.0

    for i, (x_raw, y_raw) in enumerate(zip(a, b)):
        x = float(x_raw)
        y = float(y_raw)
        if not (math.isfinite(x) and math.isfinite(y)):
            finite = False
            max_index = i
            base_at_max = x
            cand_at_max = y
            break
        diff = y - x
        adiff = abs(diff)
        sum_diff2 += diff * diff
        sum_base2 += x * x
        denom = max(abs(x), abs(y), 1.0e-30)
        rel = adiff / denom
        if adiff > max_abs:
            max_abs = adiff
            max_rel = rel
            max_index = i
            base_at_max = x
            cand_at_max = y

    if not finite:
        return {
            "pass": False,
            "reason": "non_finite",
            "count": len(a),
            "max_index": max_index,
            "baseline_at_max": base_at_max,
            "candidate_at_max": cand_at_max,
        }

    rms = math.sqrt(sum_diff2 / len(a)) if a else 0.0
    if sum_base2 == 0.0:
        rel_l2 = 0.0 if sum_diff2 == 0.0 else float("inf")
        passed = max_abs <= abs_tol
        criterion = "abs"
    else:
        rel_l2 = math.sqrt(sum_diff2 / sum_base2)
        passed = rel_l2 <= rel_tol
        criterion = "rel_l2"

    return {
        "pass": passed,
        "criterion": criterion,
        "count": len(a),
        "rel_l2": rel_l2,
        "rms": rms,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_index": max_index,
        "baseline_at_max": base_at_max,
        "candidate_at_max": cand_at_max,
        "rel_tol": rel_tol,
        "abs_tol": abs_tol,
    }


CORE_DUMP_RE = re.compile(r"^rank_(?P<rank>\d+)_shot_(?P<shot>\d+)_it_(?P<it>\d+)_(?P<field>p[012])_core\.bin$")


def parse_core_dump_name(path):
    match = CORE_DUMP_RE.match(path.name)
    if match is None:
        return None
    return (
        int(match.group("rank")),
        int(match.group("shot")),
        int(match.group("it")),
        match.group("field"),
    )


def collect_core_dumps(path):
    files = {}
    ignored = []
    for dump in sorted(path.glob("*_core.bin")):
        key = parse_core_dump_name(dump)
        if key is None:
            ignored.append(dump.name)
        else:
            files[key] = dump
    return files, ignored


def compare_same_name(baseline, candidate, rel_tol, abs_tol):
    base_files = {p.name: p for p in sorted(baseline.glob("*_core.bin"))}
    cand_files = {p.name: p for p in sorted(candidate.glob("*_core.bin"))}
    missing = sorted(set(base_files) - set(cand_files))
    extra = sorted(set(cand_files) - set(base_files))

    results = []
    all_pass = not missing and not extra
    for name in sorted(set(base_files) & set(cand_files)):
        result = compare_float_files(base_files[name], cand_files[name], rel_tol, abs_tol)
        result["file"] = name
        result["baseline_file"] = name
        result["candidate_file"] = name
        results.append(result)
        all_pass = all_pass and result.get("pass", False)

    return all_pass, missing, extra, [], results


def compare_p2_shift(baseline, candidate, rel_tol, abs_tol):
    base_files, base_ignored = collect_core_dumps(baseline)
    cand_files, cand_ignored = collect_core_dumps(candidate)

    p2_keys = sorted(key for key in cand_files if key[3] == "p2")
    targets = {}
    missing = []
    results = []
    all_pass = bool(p2_keys)

    for rank, shot, it, _field in p2_keys:
        target_key = (rank, shot, it + 1, "p0")
        cand_path = cand_files[(rank, shot, it, "p2")]
        base_path = base_files.get(target_key)
        label = f"rank_{rank}_shot_{shot}_it_{it}_p2_core.bin -> it_{it + 1}_p0_core.bin"
        if base_path is None:
            missing.append(f"rank_{rank}_shot_{shot}_it_{it + 1}_p0_core.bin")
            all_pass = False
            continue
        targets[target_key] = True
        result = compare_float_files(base_path, cand_path, rel_tol, abs_tol)
        result["file"] = label
        result["baseline_file"] = base_path.name
        result["candidate_file"] = cand_path.name
        results.append(result)
        all_pass = all_pass and result.get("pass", False)

    extra = []
    if not p2_keys:
        missing.append("candidate p2_core dumps")
    ignored = sorted(base_ignored + cand_ignored)
    return all_pass, sorted(missing), extra, ignored, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rel-tol", type=float, default=1.0e-5)
    parser.add_argument("--abs-tol", type=float, default=1.0e-7)
    parser.add_argument(
        "--mode",
        choices=("same-name", "p2-shift"),
        default="same-name",
        help="same-name compares matching dump names; p2-shift compares candidate p2(it) with baseline p0(it+1)",
    )
    args = parser.parse_args()

    baseline = Path(args.baseline)
    candidate = Path(args.candidate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "same-name":
        all_pass, missing, extra, ignored, results = compare_same_name(
            baseline, candidate, args.rel_tol, args.abs_tol
        )
    else:
        all_pass, missing, extra, ignored, results = compare_p2_shift(
            baseline, candidate, args.rel_tol, args.abs_tol
        )

    meta_files = sorted(baseline.glob("*_core_meta.txt"))
    meta_summary = [parse_meta(path) for path in meta_files[:8]]
    report = {
        "pass": all_pass,
        "mode": args.mode,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "missing": missing,
        "extra": extra,
        "ignored": ignored,
        "meta_samples": meta_summary,
        "results": results,
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Core Interior Dump Comparison",
        "",
        f"- Pass: `{all_pass}`",
        f"- Mode: `{args.mode}`",
        f"- Baseline: `{baseline}`",
        f"- Candidate: `{candidate}`",
        f"- Missing files: `{len(missing)}`",
        f"- Extra files: `{len(extra)}`",
        f"- Ignored files: `{len(ignored)}`",
        "",
        "| File | Pass | Criterion | Count | Rel L2 | Max Abs | Max Rel | RMS | Max Index |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['file']} | {r.get('pass')} | {r.get('criterion', r.get('reason'))} | "
            f"{r.get('count', 0)} | {r.get('rel_l2', 0.0):.6e} | "
            f"{r.get('max_abs', 0.0):.6e} | {r.get('max_rel', 0.0):.6e} | "
            f"{r.get('rms', 0.0):.6e} | {r.get('max_index', 0)} |"
        )
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "comparison.md")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
