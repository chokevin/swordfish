#!/usr/bin/env bash
# Profile Marlin (and FP16 baseline) across all P0 shapes on an A100.
#
# Produces, under docs/profiling/marlin/<timestamp>/:
#   env.txt              host/driver/CUDA/torch/triton snapshot
#   summary.csv          bench/run_bench.py results
#   manifest.json        same, with env header
#   trace.json           torch.profiler Chrome trace (Perfetto-loadable)
#   <shape>.nsys-rep     nsys timeline per shape
#   <shape>.sqlite       nsys SQLite (for trace-processor / Perfetto)
#   <shape>.ncu-rep      ncu deep-dive per shape
#   <shape>.ncu.csv      flat metric table for roofline.py
#
# All NVTX-annotated; nsys timeline regions match the per-impl/per-shape labels
# emitted by bench/run_bench.py.
#
# Usage (on the A100 box, from repo root):
#   bench/profile_marlin.sh            # default shape set: voice
#   SHAPES=full bench/profile_marlin.sh
#   IMPLS=fp16,marlin bench/profile_marlin.sh

set -euo pipefail

SHAPES="${SHAPES:-voice}"
IMPLS="${IMPLS:-fp16,marlin}"
REPEATS="${REPEATS:-5}"
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-10}"

if ! command -v nsys >/dev/null; then
  echo "ERROR: nsys not found. Install Nsight Systems (CUDA toolkit)." >&2
  exit 1
fi
if ! command -v ncu >/dev/null; then
  echo "ERROR: ncu not found. Install Nsight Compute (CUDA toolkit)." >&2
  exit 1
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="docs/profiling/marlin/${TS}"
mkdir -p "${OUT}"

# ---- env snapshot ------------------------------------------------------------
{
  echo "=== profile_marlin.sh @ ${TS} ==="
  echo "--- host ---"
  uname -a
  echo "--- nvidia-smi ---"
  nvidia-smi
  echo "--- nvcc ---"
  nvcc --version || true
  echo "--- nsys ---"
  nsys --version
  echo "--- ncu ---"
  ncu --version
  echo "--- python / torch / triton / marlin ---"
  python -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda); 
import triton; print('triton', triton.__version__)" 2>/dev/null || true
  python -c "import marlin; print('marlin', getattr(marlin, '__version__', 'unknown'))" 2>/dev/null \
      || echo "marlin: NOT INSTALLED — see docs/profiling/RUN_ME_ON_A100.md"
  echo "--- repo SHA ---"
  git rev-parse HEAD
} | tee "${OUT}/env.txt"

# ---- summary bench + Perfetto trace -----------------------------------------
echo
echo "[1/3] bench harness with --profile (Perfetto trace) ..."
python -m bench.run_bench \
  --shapes "${SHAPES}" \
  --impls "${IMPLS}" \
  --repeats "${REPEATS}" \
  --iters "${ITERS}" \
  --warmup "${WARMUP}" \
  --out "${OUT}" \
  --profile

# ---- per-shape nsys + ncu ---------------------------------------------------
# Use --shapes-one (single shape per invocation) so ncu doesn't trace 6 shapes
# into one giant report. We rely on a shape filter via $IMPLS unchanged but
# loop the shape names extracted from bench.shapes.
SHAPE_NAMES=$(python -c "from bench.shapes import resolve; print(' '.join(s.name for s in resolve('${SHAPES}')))")

# Single-shape runner that the profilers wrap.
RUN_ONE='python -m bench.run_bench --shapes %SHAPE% --impls '"${IMPLS}"' --repeats 1 --iters 5 --warmup 3'

# We need a shape set per single-shape name; hack: extend bench.shapes via env?
# Simpler: write a tiny inline runner that pulls one Shape and runs it directly.
cat > "${OUT}/_run_one.py" <<'PY'
"""Tiny driver: bench a single named shape across given impls. Used by the
profiler loop so each .nsys-rep / .ncu-rep covers exactly one shape."""
import sys
from bench.run_bench import bench_shape
from bench.shapes import ALL_SHAPES

name = sys.argv[1]
impls = sys.argv[2].split(",")
shape = next(s for s in ALL_SHAPES if s.name == name)
for r in bench_shape(shape, impls, repeats=1, warmup=3, iters=5):
    print(r)
PY

echo
echo "[2/3] nsys per-shape timelines ..."
for SH in ${SHAPE_NAMES}; do
  echo "  nsys ${SH}"
  # nsys 2024.2 (NGC 24.05) is strict about flag parsing; --sample=none was
  # ambiguous-matching against several `sampl*` options in iter-4. Drop the
  # noise-reduction flags — defaults add a small CPU sampling overhead but
  # don't pollute the GPU timeline, which is what we actually analyze.
  nsys profile \
    --trace=cuda,nvtx,osrt,cudnn \
    --output="${OUT}/${SH}" \
    --force-overwrite=true \
    -- python "${OUT}/_run_one.py" "${SH}" "${IMPLS}" >/dev/null \
    || { echo "WARN: nsys profile failed for ${SH}; continuing"; }
  # Export to SQLite (Perfetto's trace_processor can ingest this for unified view)
  nsys export --type=sqlite --output="${OUT}/${SH}.sqlite" --force-overwrite=true \
    "${OUT}/${SH}.nsys-rep" >/dev/null 2>&1 || true
done

echo
echo "[3/3] ncu per-shape deep dive ..."
# Sections that matter for our diagnosis:
#   SpeedOfLight + SpeedOfLight_RooflineChart -> roofline data
#   MemoryWorkloadAnalysis -> HBM, L2, SMEM bandwidth utilization
#   SchedulerStats + WarpStateStats -> stall reasons (LG throttle, MIO, MMA, etc.)
#   ComputeWorkloadAnalysis -> tensor-core pipe utilization
NCU_SECTIONS="SpeedOfLight,SpeedOfLight_RooflineChart,MemoryWorkloadAnalysis,SchedulerStats,WarpStateStats,ComputeWorkloadAnalysis"

# Roofline-relevant raw metrics for the CSV that bench/roofline.py consumes.
NCU_METRICS="\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__inst_executed_pipe_tensor.sum,\
sm__cycles_elapsed.avg,\
sm__cycles_elapsed.avg.per_second,\
dram__bytes.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
gpu__time_duration.sum"

for SH in ${SHAPE_NAMES}; do
  echo "  ncu ${SH}"
  # Full report (UI-loadable). Capture stderr to a sidecar log so failures
  # are diagnosable post-hoc instead of vanishing to /dev/null (iter-6 caught
  # ncu producing empty CSVs with the actual error message hidden).
  # IMPORTANT: do NOT quote the --section expansion — ncu wants each
  # --section as a separate arg; quoting collapses them into one literal
  # arg ("--section a --section b --section c") which ncu rejects with
  # "did not match any section" (iter-7 catch).
  ncu --set full \
    --section ${NCU_SECTIONS//,/ --section } \
    --target-processes all \
    --replay-mode kernel \
    --export "${OUT}/${SH}.ncu-rep" --force-overwrite \
    python "${OUT}/_run_one.py" "${SH}" "${IMPLS}" >"${OUT}/${SH}.ncu.log" 2>&1 || \
      echo "    (ncu full failed for ${SH} — see ${SH}.ncu.log)"

  # Flat CSV with only the roofline metrics (cheap second pass)
  ncu --csv \
    --metrics "${NCU_METRICS}" \
    --target-processes all \
    --replay-mode kernel \
    python "${OUT}/_run_one.py" "${SH}" "${IMPLS}" \
    > "${OUT}/${SH}.ncu.csv" 2>"${OUT}/${SH}.ncu-csv.log" || \
      echo "    (ncu csv failed for ${SH})"
done

rm -f "${OUT}/_run_one.py"

echo
echo "DONE. artifacts in ${OUT}"
echo "next: python -m bench.roofline ${OUT}    # generates roofline.png"
