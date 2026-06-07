#!/usr/bin/env python3
import argparse
import array
import json
import math
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rel-tol", type=float, default=1.0e-5)
    parser.add_argument("--abs-tol", type=float, default=1.0e-7)
    args = parser.parse_args()

    baseline = Path(args.baseline)
    candidate = Path(args.candidate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_files = {p.name: p for p in sorted(baseline.glob("*_core.bin"))}
    cand_files = {p.name: p for p in sorted(candidate.glob("*_core.bin"))}
    missing = sorted(set(base_files) - set(cand_files))
    extra = sorted(set(cand_files) - set(base_files))

    results = []
    all_pass = not missing and not extra
    for name in sorted(set(base_files) & set(cand_files)):
        result = compare_float_files(base_files[name], cand_files[name], args.rel_tol, args.abs_tol)
        result["file"] = name
        results.append(result)
        all_pass = all_pass and result.get("pass", False)

    meta_files = sorted(baseline.glob("*_core_meta.txt"))
    meta_summary = [parse_meta(path) for path in meta_files[:8]]
    report = {
        "pass": all_pass,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "missing": missing,
        "extra": extra,
        "meta_samples": meta_summary,
        "results": results,
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Core Interior Dump Comparison",
        "",
        f"- Pass: `{all_pass}`",
        f"- Baseline: `{baseline}`",
        f"- Candidate: `{candidate}`",
        f"- Missing files: `{len(missing)}`",
        f"- Extra files: `{len(extra)}`",
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
