#!/usr/bin/env python3
"""Archive a verified speedup milestone without deleting or overwriting files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


SOURCE_FILES = [
    "AGENTS.md",
    "AGENT_LOG.md",
    "src/main.cu",
    "src/rem_fd.cu",
    "src/single_solver.cu",
    "src/makefile.server",
    "include/inc3D/alloc.h",
    "include/inc3D/common.h",
    "include/inc3D/cu_common.h",
    "include/inc3D/single_solver.h",
    "tools/create_smoke_case.py",
    "tools/create_benchmark_cases.py",
    "tools/run_benchmark.py",
    "tools/compare_outputs.py",
    "tools/archive_speedup.py",
    "tools/sweep_blocksize.py",
]

REPORT_FILES = ["comparison.md", "comparison.json"]
RUN_FILES = ["manifest.json", "run.log", "nvidia-smi.txt"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_one(src: Path, dst: Path) -> dict:
    if not src.exists():
        return {"source": str(src), "copied": False, "reason": "missing"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive file: {dst}")
    shutil.copy2(src, dst)
    return {
        "source": str(src),
        "archive": str(dst),
        "copied": True,
        "size": dst.stat().st_size,
        "sha256": sha256_file(dst),
    }


def copy_named_files(src_dir: Path | None, dst_dir: Path, names: list[str]) -> list[dict]:
    if src_dir is None:
        return []
    records = []
    for name in names:
        records.append(copy_one(src_dir / name, dst_dir / name))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root. Default: current directory.")
    parser.add_argument("--threshold", required=True, help="Milestone threshold, for example 1.5.")
    parser.add_argument("--speedup", required=True, type=float, help="Measured speedup.")
    parser.add_argument("--baseline-time", required=True, type=float, help="Baseline WP computing time in seconds.")
    parser.add_argument("--candidate-time", required=True, type=float, help="Candidate WP computing time in seconds.")
    parser.add_argument("--tag", required=True, help="Short archive tag.")
    parser.add_argument("--correctness-report", help="Directory containing correctness comparison report.")
    parser.add_argument("--perf-report", help="Directory containing performance comparison report.")
    parser.add_argument("--candidate-run", help="Candidate benchmark run directory.")
    parser.add_argument("--baseline-run", help="Baseline benchmark run directory.")
    parser.add_argument("--notes", default="", help="Short notes written into manifest.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold_label = f"{float(args.threshold):.1f}x"
    safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in args.tag)
    archive_dir = root / "archives" / "speedups" / f"{threshold_label}_{timestamp}_{safe_tag}"

    if archive_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive directory: {archive_dir}")

    archive_dir.mkdir(parents=True, exist_ok=False)

    source_records = []
    for rel in SOURCE_FILES:
        source_records.append(copy_one(root / rel, archive_dir / "source" / rel))

    correctness_dir = Path(args.correctness_report).resolve() if args.correctness_report else None
    perf_dir = Path(args.perf_report).resolve() if args.perf_report else None
    candidate_dir = Path(args.candidate_run).resolve() if args.candidate_run else None
    baseline_dir = Path(args.baseline_run).resolve() if args.baseline_run else None

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": threshold_label,
        "speedup": args.speedup,
        "baseline_time_seconds": args.baseline_time,
        "candidate_time_seconds": args.candidate_time,
        "tag": args.tag,
        "notes": args.notes,
        "archive_dir": str(archive_dir),
        "sources": source_records,
        "reports": {
            "correctness": copy_named_files(correctness_dir, archive_dir / "reports" / "correctness", REPORT_FILES),
            "perf": copy_named_files(perf_dir, archive_dir / "reports" / "perf", REPORT_FILES),
        },
        "runs": {
            "candidate": copy_named_files(candidate_dir, archive_dir / "runs" / "candidate", RUN_FILES),
            "baseline": copy_named_files(baseline_dir, archive_dir / "runs" / "baseline", RUN_FILES),
        },
    }

    manifest_path = archive_dir / "archive_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"archive_dir": str(archive_dir), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
