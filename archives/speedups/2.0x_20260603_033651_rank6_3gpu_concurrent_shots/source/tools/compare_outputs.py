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


def compare_file(base, new, rel_tol, abs_tol):
    if base.stat().st_size != new.stat().st_size:
        return {
            "file": base.name,
            "pass": False,
            "reason": "size_mismatch",
            "baseline_bytes": base.stat().st_size,
            "candidate_bytes": new.stat().st_size,
        }

    a = read_float32(base)
    b = read_float32(new)
    sum_diff2 = 0.0
    sum_base2 = 0.0
    max_abs = 0.0
    max_rel = 0.0
    finite = True

    for x, y in zip(a, b):
        if not (math.isfinite(x) and math.isfinite(y)):
            finite = False
            break
        diff = float(y) - float(x)
        adiff = abs(diff)
        sum_diff2 += diff * diff
        sum_base2 += float(x) * float(x)
        max_abs = max(max_abs, adiff)
        denom = max(abs(float(x)), abs(float(y)), 1.0e-30)
        max_rel = max(max_rel, adiff / denom)

    if not finite:
        return {"file": base.name, "pass": False, "reason": "non_finite"}

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
        "file": base.name,
        "pass": passed,
        "criterion": criterion,
        "count": len(a),
        "rel_l2": rel_l2,
        "rms": rms,
        "max_abs": max_abs,
        "max_rel": max_rel,
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

    base_files = {p.name: p for p in baseline.glob("*.dir")}
    new_files = {p.name: p for p in candidate.glob("*.dir")}
    results = []

    missing = sorted(set(base_files) - set(new_files))
    extra = sorted(set(new_files) - set(base_files))
    for name in sorted(set(base_files) & set(new_files)):
        results.append(compare_file(base_files[name], new_files[name], args.rel_tol, args.abs_tol))

    passed = not missing and not extra and all(r.get("pass") for r in results)
    report = {
        "pass": passed,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "missing": missing,
        "extra": extra,
        "results": results,
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Output Comparison",
        "",
        f"- Pass: `{passed}`",
        f"- Baseline: `{baseline}`",
        f"- Candidate: `{candidate}`",
        f"- Missing files: `{len(missing)}`",
        f"- Extra files: `{len(extra)}`",
        "",
        "| File | Pass | Criterion | Rel L2 | Max Abs | Max Rel | RMS |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['file']} | {r.get('pass')} | {r.get('criterion', r.get('reason'))} | "
            f"{r.get('rel_l2', 0):.6e} | {r.get('max_abs', 0):.6e} | "
            f"{r.get('max_rel', 0):.6e} | {r.get('rms', 0):.6e} |"
        )
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out / "comparison.md")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
