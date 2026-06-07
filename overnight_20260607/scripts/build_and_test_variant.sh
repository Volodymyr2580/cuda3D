#!/usr/bin/env bash

set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OVERNIGHT_DIR="$ROOT/overnight_20260607"
LOG_DIR="$OVERNIGHT_DIR/logs"
REPORT_DIR="$OVERNIGHT_DIR/reports"
BUILD_DIR="$OVERNIGHT_DIR/builds"

TAG=""
NVFLAGS=""
RUN_DEBUG=0
RUN_CORRECTNESS=0
RUN_PERF1=0
RUN_PERF6=0
RUN_PERF6_REPEAT=0
COMPARE_CORRECTNESS=""
COMPARE_PERF1=""
COMPARE_PERF6=""
COMPARE_PERF6_REPEAT=""
DEBUG_BASELINE_PREFIX=""

usage() {
  cat <<'EOF'
usage: build_and_test_variant.sh --tag TAG --nvflags NVFLAGS [options]

Options:
  --debug-dump
  --correctness
  --perf-1gpu
  --perf-1gpu-6shots
  --perf-1gpu-6shots-repeat
  --compare-correctness-run RUN_DIR
  --compare-perf1-run RUN_DIR
  --compare-perf6-run RUN_DIR
  --compare-perf6-repeat-run RUN_DIR
  --debug-baseline-prefix PREFIX
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --nvflags) NVFLAGS="$2"; shift 2 ;;
    --debug-dump) RUN_DEBUG=1; shift ;;
    --correctness) RUN_CORRECTNESS=1; shift ;;
    --perf-1gpu) RUN_PERF1=1; shift ;;
    --perf-1gpu-6shots) RUN_PERF6=1; shift ;;
    --perf-1gpu-6shots-repeat) RUN_PERF6_REPEAT=1; shift ;;
    --compare-correctness-run) COMPARE_CORRECTNESS="$2"; shift 2 ;;
    --compare-perf1-run) COMPARE_PERF1="$2"; shift 2 ;;
    --compare-perf6-run) COMPARE_PERF6="$2"; shift 2 ;;
    --compare-perf6-repeat-run) COMPARE_PERF6_REPEAT="$2"; shift 2 ;;
    --debug-baseline-prefix) DEBUG_BASELINE_PREFIX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$TAG" ] || [ -z "$NVFLAGS" ]; then
  usage >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$BUILD_DIR"

BUILD_LOG="$LOG_DIR/${TAG}_build.log"
RUNS_TSV="$REPORT_DIR/${TAG}_runs.tsv"
COMPARES_TSV="$REPORT_DIR/${TAG}_compares.tsv"
SUMMARY_JSON="$REPORT_DIR/${TAG}_summary.json"
REPORT_MD="$REPORT_DIR/${TAG}.md"
STATUS="ok"
FAIL_REASON=""

: > "$RUNS_TSV"
: > "$COMPARES_TSV"

write_failure_summary() {
  python3 - "$SUMMARY_JSON" "$REPORT_MD" "$TAG" "$NVFLAGS" "$STATUS" "$FAIL_REASON" "$BUILD_LOG" <<'PY'
import json
import sys
from pathlib import Path

summary_json, report_md, tag, nvflags, status, fail_reason, build_log = sys.argv[1:]
data = {
    "tag": tag,
    "status": status,
    "fail_reason": fail_reason,
    "nvflags": nvflags,
    "build_log": build_log,
    "runs": [],
    "compares": [],
}
Path(summary_json).write_text(json.dumps(data, indent=2), encoding="utf-8")
Path(report_md).write_text(
    "\n".join([
        f"# Variant {tag}",
        "",
        f"- Status: `{status}`",
        f"- Fail reason: `{fail_reason}`",
        f"- Build log: `{build_log}`",
        "",
        "```text",
        nvflags,
        "```",
        "",
    ]),
    encoding="utf-8",
)
PY
}

echo "TAG=$TAG"
echo "NVFLAGS=$NVFLAGS"
echo "BUILD_LOG=$BUILD_LOG"

(cd "$ROOT/src" && make -B -f makefile.rtx5090 test NVFLAGS="$NVFLAGS") > "$BUILD_LOG" 2>&1
BUILD_RC=$?
if [ "$BUILD_RC" -ne 0 ]; then
  STATUS="build_failed"
  FAIL_REASON="build rc=$BUILD_RC"
  write_failure_summary
  echo "VARIANT_FAILED tag=$TAG reason=$FAIL_REASON"
  exit 0
fi

run_case() {
  local case_name="$1"
  local label="$2"
  local stdout_log="$LOG_DIR/${TAG}_${label}.stdout"
  local run_tag="${TAG}_${label}"
  python3 "$ROOT/tools/run_benchmark.py" --case "$case_name" --tag "$run_tag" > "$stdout_log" 2>&1
  local rc=$?
  local run_dir
  run_dir="$(head -n 1 "$stdout_log" || true)"
  echo -e "${label}\t${case_name}\t${rc}\t${run_dir}\t${stdout_log}" >> "$RUNS_TSV"
  cat "$stdout_log"
  return "$rc"
}

compare_run() {
  local label="$1"
  local baseline_run="$2"
  local candidate_run="$3"
  if [ -z "$baseline_run" ] || [ -z "$candidate_run" ]; then
    return 0
  fi
  local out_dir="$REPORT_DIR/${TAG}_compare_${label}"
  python3 "$ROOT/tools/compare_outputs.py" \
    --baseline "$baseline_run/outputs" \
    --candidate "$candidate_run/outputs" \
    --out "$out_dir" > "$LOG_DIR/${TAG}_compare_${label}.stdout" 2>&1
  local rc=$?
  echo -e "${label}\t${rc}\t${baseline_run}\t${candidate_run}\t${out_dir}" >> "$COMPARES_TSV"
  cat "$LOG_DIR/${TAG}_compare_${label}.stdout"
  if [ -f "$out_dir/comparison.md" ]; then
    cat "$out_dir/comparison.md"
  fi
  return "$rc"
}

debug_dump() {
  local step="$1"
  local dump_dir="$OVERNIGHT_DIR/profiles/${TAG}_debug_it${step}"
  local stdout_log="$LOG_DIR/${TAG}_debug_it${step}.stdout"
  mkdir -p "$dump_dir"
  CUDA3D_PML_DUMP_DIR="$dump_dir" CUDA3D_PML_DUMP_STEP="$step" \
    python3 "$ROOT/tools/run_benchmark.py" --case smoke_1gpu --tag "${TAG}_debug_it${step}" > "$stdout_log" 2>&1
  local rc=$?
  local run_dir
  run_dir="$(head -n 1 "$stdout_log" || true)"
  echo -e "debug_it${step}\tsmoke_1gpu\t${rc}\t${run_dir}\t${stdout_log}\t${dump_dir}" >> "$RUNS_TSV"
  cat "$stdout_log"
  if [ "$rc" -ne 0 ]; then
    return "$rc"
  fi
  if [ -n "$DEBUG_BASELINE_PREFIX" ]; then
    local out_dir="$REPORT_DIR/${TAG}_debug_compare_it${step}"
    python3 "$ROOT/tools/compare_debug_dumps.py" \
      --baseline "${DEBUG_BASELINE_PREFIX}_it${step}" \
      --candidate "$dump_dir" \
      --out "$out_dir" > "$LOG_DIR/${TAG}_debug_compare_it${step}.stdout" 2>&1
    rc=$?
    echo -e "debug_it${step}\t${rc}\t${DEBUG_BASELINE_PREFIX}_it${step}\t${dump_dir}\t${out_dir}" >> "$COMPARES_TSV"
    cat "$LOG_DIR/${TAG}_debug_compare_it${step}.stdout"
    if [ -f "$out_dir/comparison.md" ]; then
      cat "$out_dir/comparison.md"
    fi
  fi
  return "$rc"
}

if [ "$RUN_DEBUG" -eq 1 ]; then
  for step in 0 1 2; do
    if ! debug_dump "$step"; then
      STATUS="debug_failed"
      FAIL_REASON="debug dump step $step failed"
      write_failure_summary
      echo "VARIANT_FAILED tag=$TAG reason=$FAIL_REASON"
      exit 0
    fi
  done
fi

CORRECTNESS_RUN=""
PERF1_RUN=""
PERF6_RUN=""
PERF6_REPEAT_RUN=""

if [ "$RUN_CORRECTNESS" -eq 1 ]; then
  if ! run_case correctness correctness; then
    STATUS="correctness_failed"
    FAIL_REASON="correctness run failed"
    write_failure_summary
    echo "VARIANT_FAILED tag=$TAG reason=$FAIL_REASON"
    exit 0
  fi
  CORRECTNESS_RUN="$(awk -F '\t' '$1=="correctness"{print $4}' "$RUNS_TSV" | tail -n 1)"
  if ! compare_run correctness "$COMPARE_CORRECTNESS" "$CORRECTNESS_RUN"; then
    STATUS="correctness_compare_failed"
    FAIL_REASON="correctness compare failed"
    write_failure_summary
    echo "VARIANT_FAILED tag=$TAG reason=$FAIL_REASON"
    exit 0
  fi
fi

if [ "$RUN_PERF1" -eq 1 ]; then
  if ! run_case perf_1gpu perf1; then
    STATUS="perf1_failed"
    FAIL_REASON="perf_1gpu failed"
    write_failure_summary
    echo "VARIANT_FAILED tag=$TAG reason=$FAIL_REASON"
    exit 0
  fi
  PERF1_RUN="$(awk -F '\t' '$1=="perf1"{print $4}' "$RUNS_TSV" | tail -n 1)"
  if ! compare_run perf1 "$COMPARE_PERF1" "$PERF1_RUN"; then
    STATUS="perf1_compare_failed"
    FAIL_REASON="perf_1gpu compare failed"
    write_failure_summary
    echo "VARIANT_FAILED tag=$TAG reason=$FAIL_REASON"
    exit 0
  fi
fi

if [ "$RUN_PERF6" -eq 1 ]; then
  if ! run_case perf_1gpu_6shots perf6; then
    STATUS="perf6_failed"
    FAIL_REASON="perf_1gpu_6shots failed"
    write_failure_summary
    echo "VARIANT_FAILED tag=$TAG reason=$FAIL_REASON"
    exit 0
  fi
  PERF6_RUN="$(awk -F '\t' '$1=="perf6"{print $4}' "$RUNS_TSV" | tail -n 1)"
  if ! compare_run perf6 "$COMPARE_PERF6" "$PERF6_RUN"; then
    STATUS="perf6_compare_failed"
    FAIL_REASON="perf_1gpu_6shots compare failed"
    write_failure_summary
    echo "VARIANT_FAILED tag=$TAG reason=$FAIL_REASON"
    exit 0
  fi
fi

if [ "$RUN_PERF6_REPEAT" -eq 1 ]; then
  if ! run_case perf_1gpu_6shots perf6_repeat; then
    STATUS="perf6_repeat_failed"
    FAIL_REASON="perf_1gpu_6shots repeat failed"
    write_failure_summary
    echo "VARIANT_FAILED tag=$TAG reason=$FAIL_REASON"
    exit 0
  fi
  PERF6_REPEAT_RUN="$(awk -F '\t' '$1=="perf6_repeat"{print $4}' "$RUNS_TSV" | tail -n 1)"
  if ! compare_run perf6_repeat "$COMPARE_PERF6_REPEAT" "$PERF6_REPEAT_RUN"; then
    STATUS="perf6_repeat_compare_failed"
    FAIL_REASON="perf_1gpu_6shots repeat compare failed"
    write_failure_summary
    echo "VARIANT_FAILED tag=$TAG reason=$FAIL_REASON"
    exit 0
  fi
fi

python3 - "$SUMMARY_JSON" "$REPORT_MD" "$TAG" "$NVFLAGS" "$BUILD_LOG" "$RUNS_TSV" "$COMPARES_TSV" "$STATUS" <<'PY'
import json
import re
import sys
from pathlib import Path

summary_json, report_md, tag, nvflags, build_log, runs_tsv, compares_tsv, status = sys.argv[1:]

def parse_time(run_dir):
    path = Path(run_dir) / "run.log"
    data = {"wp": None, "gradient": None, "wall": None, "all_done": False}
    if not path.exists():
        return data
    text = path.read_text(encoding="utf-8", errors="replace")
    data["all_done"] = "ALL DONE" in text
    m = re.search(r"Gradient TIME all=\s*([0-9.]+)s,\s*WP computing time =\s*([0-9.]+)s", text)
    if m:
        data["gradient"] = float(m.group(1))
        data["wp"] = float(m.group(2))
    m = re.search(r"Elapsed \(wall clock\) time .*:\s*([0-9:.]+)", text)
    if m:
        data["wall"] = m.group(1)
    return data

runs = []
for line in Path(runs_tsv).read_text(encoding="utf-8").splitlines():
    parts = line.split("\t")
    if len(parts) < 5:
        continue
    row = {
        "label": parts[0],
        "case": parts[1],
        "returncode": int(parts[2]),
        "run_dir": parts[3],
        "stdout_log": parts[4],
    }
    if len(parts) > 5:
        row["dump_dir"] = parts[5]
    row.update(parse_time(parts[3]))
    runs.append(row)

compares = []
if Path(compares_tsv).exists():
    for line in Path(compares_tsv).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        compares.append({
            "label": parts[0],
            "returncode": int(parts[1]),
            "baseline": parts[2],
            "candidate": parts[3],
            "out_dir": parts[4],
        })

summary = {
    "tag": tag,
    "status": status,
    "nvflags": nvflags,
    "build_log": build_log,
    "runs": runs,
    "compares": compares,
}
Path(summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

lines = [
    f"# Variant {tag}",
    "",
    f"- Status: `{status}`",
    f"- Build log: `{build_log}`",
    "",
    "```text",
    nvflags,
    "```",
    "",
    "| Label | Case | RC | WP | Gradient | Wall | ALL DONE | Run dir |",
    "|---|---|---:|---:|---:|---|---:|---|",
]
for r in runs:
    lines.append(
        f"| {r['label']} | {r['case']} | {r['returncode']} | "
        f"{r.get('wp')} | {r.get('gradient')} | {r.get('wall')} | "
        f"{r.get('all_done')} | `{r['run_dir']}` |"
    )
lines.extend(["", "| Compare | RC | Baseline | Candidate | Report |", "|---|---:|---|---|---|"])
for c in compares:
    lines.append(
        f"| {c['label']} | {c['returncode']} | `{c['baseline']}` | "
        f"`{c['candidate']}` | `{c['out_dir']}` |"
    )
Path(report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

sha256sum "$ROOT/bin/cuda_3D_FM" > "$BUILD_DIR/${TAG}_binary.sha256" 2>/dev/null || true
echo "VARIANT_DONE tag=$TAG summary=$SUMMARY_JSON report=$REPORT_MD"
