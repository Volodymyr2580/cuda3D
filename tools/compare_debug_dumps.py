#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def compare_file(base_path: Path, cand_path: Path):
    base = np.fromfile(base_path, dtype=np.float32)
    cand = np.fromfile(cand_path, dtype=np.float32)
    if base.shape != cand.shape:
        return {
            "pass": False,
            "reason": f"shape mismatch {base.shape} vs {cand.shape}",
        }
    base_finite = np.isfinite(base).all()
    cand_finite = np.isfinite(cand).all()
    diff = cand - base
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    denom = float(np.linalg.norm(base.astype(np.float64)))
    rel_l2 = float(np.linalg.norm(diff.astype(np.float64)) / denom) if denom != 0.0 else float(np.linalg.norm(diff.astype(np.float64)))
    idx = int(abs_diff.argmax()) if abs_diff.size else 0
    max_rel = float(max_abs / max(abs(float(base[idx])), 1.0e-30)) if abs_diff.size else 0.0
    return {
        "pass": bool(base_finite and cand_finite and rel_l2 <= 1.0e-5),
        "base_finite": bool(base_finite),
        "cand_finite": bool(cand_finite),
        "count": int(base.size),
        "rel_l2": rel_l2,
        "max_abs": max_abs,
        "max_rel_at_max_abs": max_rel,
        "max_abs_index": idx,
        "base_at_max_abs": float(base[idx]) if abs_diff.size else 0.0,
        "cand_at_max_abs": float(cand[idx]) if abs_diff.size else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline = Path(args.baseline)
    candidate = Path(args.candidate)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_files = {path.name: path for path in sorted(baseline.glob("*.bin"))}
    cand_files = {path.name: path for path in sorted(candidate.glob("*.bin"))}
    missing = sorted(set(base_files) - set(cand_files))
    extra = sorted(set(cand_files) - set(base_files))
    rows = []
    all_pass = not missing and not extra

    for name in sorted(set(base_files) & set(cand_files)):
        result = compare_file(base_files[name], cand_files[name])
        result["name"] = name
        rows.append(result)
        all_pass = all_pass and result["pass"]

    lines = []
    lines.append("# PML Debug Dump Comparison\n")
    lines.append(f"- Pass: `{all_pass}`")
    lines.append(f"- Baseline: `{baseline}`")
    lines.append(f"- Candidate: `{candidate}`")
    lines.append(f"- Missing files: `{len(missing)}`")
    lines.append(f"- Extra files: `{len(extra)}`\n")
    if missing:
        lines.append("Missing:\n")
        lines.extend(f"- `{name}`" for name in missing)
        lines.append("")
    if extra:
        lines.append("Extra:\n")
        lines.extend(f"- `{name}`" for name in extra)
        lines.append("")

    lines.append("| File | Pass | Count | Rel L2 | Max Abs | Max Rel@Max | Max Index | Base | Candidate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if "reason" in row:
            lines.append(f"| {row['name']} | False | 0 | nan | nan | nan | 0 | nan | nan |")
        else:
            lines.append(
                f"| {row['name']} | {row['pass']} | {row['count']} | "
                f"{row['rel_l2']:.6e} | {row['max_abs']:.6e} | "
                f"{row['max_rel_at_max_abs']:.6e} | {row['max_abs_index']} | "
                f"{row['base_at_max_abs']:.6e} | {row['cand_at_max_abs']:.6e} |"
            )
    report = "\n".join(lines) + "\n"
    (out_dir / "comparison.md").write_text(report, encoding="utf-8")
    print(out_dir / "comparison.md")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
