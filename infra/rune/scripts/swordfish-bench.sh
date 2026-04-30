#!/usr/bin/env bash
# infra/rune/scripts/swordfish-bench.sh
#
# In-pod entrypoint for swordfish benchmarks dispatched via `rune submit`.
# Invoked by the rune profile's runtime.command (bash). All arguments are
# forwarded verbatim to `python -m swordfish.runner`.
#
# This script is the BASH path. For Python-first iteration (recommended
# day-to-day flow), researchers write a `.py` file with a
# `#!/usr/bin/env python3` shebang and pass it directly via
# `rune submit --script my_bench.py` — rune respects the shebang. See
# experiments/ for examples.
#
# The image bakes the swordfish source at /work/swordfish (pip-installed
# editable). PYTHONPATH/imports resolve without a PVC mount or pip-install
# preflight. The optional /data volume (training-nfs) is for result JSONs.
#
# NOTE on profiling: rune now natively supports `--profile-mode ncu|nsys`
# which wraps the entrypoint at the renderer level and writes to
# `$RUNE_PROFILE_OUT` (= /data/<job-name>/profile/profile.{ncu-rep|nsys-rep}).
# That is the day-to-day path. The SWORDFISH_PROFILE env-var path below is
# kept as a back-compat / advanced-override for when the caller wants the
# legacy ncu CSV format with section overrides; pass `--profile-mode` to
# rune for the binary .ncu-rep/.nsys-rep, or set SWORDFISH_PROFILE here.
#
# Profile-supplied environment variables consumed here:
#   RUNE_DATA_DIR        — V1 storage contract; durable mount (default /data).
#                           Result JSONs land under $RUNE_DATA_DIR/swordfish/...
#   SWORDFISH_ARCH_LABEL — a100/h100/h200, used as the default --arch-label
#                           when the caller does not pass one.
#   SWORDFISH_PROFILE    — when set, wraps the python invocation in a
#                           profiler. Values: ncu | nsys | none (default).
#                           Prefer rune's `--profile-mode` for new flows.
#   SWORDFISH_PROFILE_OUT — explicit path for profile output. When unset the
#                           script derives it from the --out flag in $@:
#                             ncu  -> ${out_json%.json}.ncu.csv
#                             nsys -> ${out_json%.json}.nsys-rep
#
# Usage (forwarded by rune):
#   bash infra/rune/scripts/swordfish-bench.sh \
#       run-gemm --backend torch --m 4096 --n 4096 --k 4096 ...
#
#   bash infra/rune/scripts/swordfish-bench.sh \
#       liger-perkernel --kernel rmsnorm --hidden 4096 ...

set -euo pipefail

DATA_DIR="${RUNE_DATA_DIR:-/data}"
ARCH_LABEL="${SWORDFISH_ARCH_LABEL:-}"
PROFILE="${SWORDFISH_PROFILE:-none}"

# The image has source baked at /work/swordfish. Cd there so relative result
# paths and `git`-style commands (rare) work as expected.
cd /work/swordfish

mkdir -p "${DATA_DIR}/swordfish"

echo "== swordfish-bench =="
echo "host:        $(hostname)"
echo "swordfish:   ${SWORDFISH_SHA:-unknown} (baked into image)"
echo "data_dir:    ${DATA_DIR}"
echo "arch_label:  ${ARCH_LABEL:-<unset>}"
echo "profile:     ${PROFILE}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi
echo "args:        $*"
echo "===================="

# Inject --arch-label only if the caller did not pass one.
inject_arch=()
if [[ -n "$ARCH_LABEL" ]]; then
  has_arch=0
  for a in "$@"; do
    [[ "$a" == "--arch-label" || "$a" == --arch-label=* ]] && has_arch=1
  done
  if (( has_arch == 0 )); then
    inject_arch=(--arch-label "$ARCH_LABEL")
  fi
fi

PYTHON_CMD=(python -m swordfish.runner "$@" "${inject_arch[@]}")

# Derive a default profile output path from the --out flag the caller passed
# to swordfish.runner. The profile output sits next to the result JSON so
# fetch_result(include_traces=True) can grab both with one path stem.
derive_profile_out() {
  local extension="$1"
  local out_json=""
  local prev=""
  for a in "$@"; do
    if [[ "$prev" == "--out" ]]; then
      out_json="$a"
      break
    fi
    if [[ "$a" == --out=* ]]; then
      out_json="${a#--out=}"
      break
    fi
    prev="$a"
  done
  if [[ -z "$out_json" ]]; then
    echo "${DATA_DIR}/swordfish/profile.${extension}"
  else
    echo "${out_json%.json}.${extension}"
  fi
}

case "$PROFILE" in
  none|"")
    exec "${PYTHON_CMD[@]}"
    ;;
  ncu)
    if ! command -v ncu >/dev/null 2>&1; then
      echo "SWORDFISH_PROFILE=ncu but ncu is not on PATH" >&2
      exit 3
    fi
    out="${SWORDFISH_PROFILE_OUT:-$(derive_profile_out csv "$@")}"
    out="${out%.ncu.csv}.ncu.csv"
    mkdir -p "$(dirname "$out")"
    echo "wrapping in ncu -> $out"
    exec ncu \
        --csv \
        --log-file "$out" \
        --target-processes all \
        --section LaunchStats \
        --section Occupancy \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        "${PYTHON_CMD[@]}"
    ;;
  nsys)
    if ! command -v nsys >/dev/null 2>&1; then
      echo "SWORDFISH_PROFILE=nsys but nsys is not on PATH" >&2
      exit 3
    fi
    out="${SWORDFISH_PROFILE_OUT:-$(derive_profile_out nsys-rep "$@")}"
    out="${out%.nsys-rep}"
    mkdir -p "$(dirname "$out")"
    echo "wrapping in nsys -> ${out}.nsys-rep"
    exec nsys profile \
        --output "$out" \
        --trace cuda,nvtx,osrt \
        --stats true \
        --force-overwrite true \
        "${PYTHON_CMD[@]}"
    ;;
  *)
    echo "unknown SWORDFISH_PROFILE=${PROFILE}; expected one of: none, ncu, nsys" >&2
    exit 3
    ;;
esac
