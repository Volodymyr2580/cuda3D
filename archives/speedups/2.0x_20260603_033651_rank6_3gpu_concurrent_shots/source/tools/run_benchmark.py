#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


CASE_DEFAULTS = {
    "smoke_1gpu": {
        "case_dir": "bench_smoke",
        "input": "input_smoke.in",
        "np": 1,
        "gpus": "0",
        "timeout": 120,
    },
    "smoke_3gpu": {
        "case_dir": "bench_smoke",
        "input": "input_smoke_3gpu.in",
        "np": 3,
        "gpus": "0,1,2",
        "timeout": 120,
    },
    "correctness": {
        "case_dir": "benchmarks/cases/correctness",
        "input": "input_correctness.in",
        "np": 1,
        "gpus": "0",
        "timeout": 600,
    },
    "perf_3gpu": {
        "case_dir": "benchmarks/cases/perf_3gpu",
        "input": "input_perf_3gpu.in",
        "np": 3,
        "gpus": "0,1,2",
        "timeout": 1200,
    },
    "profile_1gpu": {
        "case_dir": "benchmarks/cases/profile_1gpu",
        "input": "input_profile_1gpu.in",
        "np": 1,
        "gpus": "0",
        "timeout": 900,
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_capture(command, cwd):
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def collect_hashes(root, case_dir, input_name):
    files = [
        root / "bin" / "cuda_3D_FM",
        case_dir / input_name,
    ]
    manifest_path = case_dir / "case_manifest.txt"
    if manifest_path.exists():
        files.append(manifest_path)
        manifest = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                manifest[key] = value
        for key in ["velocity", "nav", "input"]:
            if key in manifest:
                files.append(case_dir / manifest[key])
    return {str(path.relative_to(root)): sha256(path) for path in files if path.exists()}


def copy_outputs(case_dir, run_dir):
    out_dir = run_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    source_dir = case_dir / "d_obs"
    if source_dir.exists():
        for path in sorted(source_dir.glob("*.dir")):
            target = out_dir / path.name
            shutil.copy2(path, target)
            copied.append({"file": path.name, "bytes": target.stat().st_size, "sha256": sha256(target)})
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=CASE_DEFAULTS.keys())
    parser.add_argument("--tag", default="run")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--np", type=int)
    parser.add_argument("--gpus")
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    defaults = CASE_DEFAULTS[args.case]
    case_dir = root / defaults["case_dir"]
    input_name = defaults["input"]
    mpi_np = args.np or defaults["np"]
    gpus = args.gpus or defaults["gpus"]
    timeout = args.timeout or defaults["timeout"]

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = root / "benchmarks" / ("baselines/current_runnable" if args.baseline else "runs")
    run_dir = base_dir / f"{args.case}_{args.tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    env_info = {
        "timestamp": timestamp,
        "case": args.case,
        "tag": args.tag,
        "baseline": args.baseline,
        "case_dir": str(case_dir.relative_to(root)),
        "input": input_name,
        "np": mpi_np,
        "gpus": gpus,
        "timeout_seconds": timeout,
        "hashes": collect_hashes(root, case_dir, input_name),
    }
    env_info["nvidia_smi_before"] = run_capture(
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader",
        root,
    )
    env_info["tool_versions"] = run_capture(
        "/usr/local/cuda-12.2/bin/nvcc --version; /opt/intel/oneapi/mpi/latest/bin/mpirun --version | head -5; gcc --version | head -1",
        root,
    )

    cmd = (
        "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; "
        f"cd {case_dir}; "
        f"CUDA_VISIBLE_DEVICES={gpus} timeout {timeout}s "
        f"/opt/intel/oneapi/mpi/latest/bin/mpirun -np {mpi_np} "
        f"{root / 'bin' / 'cuda_3D_FM'} < {input_name}"
    )
    timed_cmd = f"/usr/bin/time -v bash -lc {json.dumps(cmd)}"
    result = subprocess.run(
        timed_cmd,
        cwd=root,
        text=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    (run_dir / "run.log").write_text(result.stdout, encoding="utf-8")
    outputs = copy_outputs(case_dir, run_dir)
    env_info["returncode"] = result.returncode
    env_info["outputs"] = outputs
    env_info["nvidia_smi_after"] = run_capture(
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader",
        root,
    )
    (run_dir / "manifest.json").write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    print(run_dir)
    print(f"returncode={result.returncode}")
    print(f"outputs={len(outputs)}")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
