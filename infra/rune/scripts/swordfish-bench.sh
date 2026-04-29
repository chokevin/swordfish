#!/usr/bin/env bash
# infra/rune/scripts/swordfish-bench.sh
#
# In-pod entrypoint for swordfish benchmarks dispatched via `rune submit`.
# Invoked by the rune profile's runtime.command (bash). All arguments are
# forwarded verbatim to `python -m swordfish.runner`.
#
# Profile-supplied environment variables consumed here:
#   SWORDFISH_SOURCE_DIR  — where the swordfish working tree lives on the
#                            shared PVC (default /data-nfs/swordfish/src/current).
#   SWORDFISH_RESULT_DIR  — where to land result JSONs (default
#                            /data-nfs/swordfish/week1).
#   SWORDFISH_ARCH_LABEL  — a100 / h100 / h200, used as the default
#                            --arch-label when the caller does not pass one.
#   REF                    — provenance string recorded in env.source_ref.
#
# Optional overrides set by the caller's --env flags on `rune submit`:
#   SWORDFISH_PROFILE_NCU=1   wrap the python invocation in `ncu`.
#   SWORDFISH_NCU_OUT          target path for the NCU CSV (default
#                              "${out_json%.json}.ncu.csv").
#
# Usage (forwarded by rune):
#   bash infra/rune/scripts/swordfish-bench.sh \
#       run-gemm --backend torch --m 4096 --n 4096 --k 4096 ...
#
#   bash infra/rune/scripts/swordfish-bench.sh \
#       liger-perkernel --kernel rmsnorm --hidden 4096 ...

set -euo pipefail

SOURCE_DIR="${SWORDFISH_SOURCE_DIR:-/data-nfs/swordfish/src/current}"
RESULT_DIR="${SWORDFISH_RESULT_DIR:-/data-nfs/swordfish/week1}"
ARCH_LABEL="${SWORDFISH_ARCH_LABEL:-}"
REF_VALUE="${REF:-rune-managed}"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "swordfish source dir not found: $SOURCE_DIR" >&2
  echo "ensure the runner image / volume has the swordfish tree at that path" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
cd "$SOURCE_DIR"

echo "== swordfish-bench =="
echo "host:       $(hostname)"
echo "source:     $SOURCE_DIR"
echo "result_dir: $RESULT_DIR"
echo "arch_label: ${ARCH_LABEL:-<unset>}"
echo "ref:        $REF_VALUE"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi
echo "args:       $*"
echo "=================="

# Defense in depth: the canonical swordfish-bench image bakes liger-kernel,
# but we may also be running on the nvcr.io/nvidia/pytorch base if the new
# image hasn't propagated yet, or on a stale tag. Check liger and install on
# the fly if missing — costs ~10s once and saves a failed run. Skipped when
# SWORDFISH_SKIP_LIGER_INSTALL=1 is set (e.g., for non-Liger GEMM jobs that
# want to keep the env stable).
if [[ "${SWORDFISH_SKIP_LIGER_INSTALL:-0}" != "1" ]]; then
  if ! python -c "import liger_kernel" >/dev/null 2>&1; then
    echo "liger_kernel missing; installing on the fly (set SWORDFISH_SKIP_LIGER_INSTALL=1 to disable)"
    pip install --no-cache-dir "liger-kernel==${SWORDFISH_LIGER_VERSION:-0.5.10}" >&2
  fi
  python -c "import liger_kernel; print(f'liger_kernel: {getattr(liger_kernel, \"__version__\", \"unknown\")}')" || true
fi

# Inject --arch-label only if the caller did not pass one. Avoids surprising
# overrides when the caller specifies a deliberate label.
inject_arch=""
if [[ -n "$ARCH_LABEL" ]]; then
  has_arch=0
  for a in "$@"; do
    [[ "$a" == "--arch-label" || "$a" == --arch-label=* ]] && has_arch=1
  done
  if (( has_arch == 0 )); then
    inject_arch="--arch-label $ARCH_LABEL"
  fi
fi

# Provenance: pass REF into the runner's env so it is recorded in env.source_ref.
export REF="$REF_VALUE"

PYTHON_CMD=(python -m swordfish.runner "$@" $inject_arch)

if [[ "${SWORDFISH_PROFILE_NCU:-0}" == "1" ]]; then
  if ! command -v ncu >/dev/null 2>&1; then
    echo "SWORDFISH_PROFILE_NCU=1 but ncu is not on PATH; install Nsight Compute" >&2
    exit 3
  fi
  ncu_out="${SWORDFISH_NCU_OUT:-/data-nfs/swordfish/week1/swordfish-bench.ncu.csv}"
  echo "wrapping in ncu -> $ncu_out"
  exec ncu \
      --csv \
      --log-file "$ncu_out" \
      --target-processes all \
      --section LaunchStats \
      --section Occupancy \
      --section SpeedOfLight \
      --section MemoryWorkloadAnalysis \
      "${PYTHON_CMD[@]}"
fi

exec "${PYTHON_CMD[@]}"
