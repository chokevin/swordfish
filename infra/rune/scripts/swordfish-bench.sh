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
# Profile-supplied environment variables consumed here:
#   RUNE_DATA_DIR   — V1 storage contract; durable mount (default /data).
#                      Result JSONs land under $RUNE_DATA_DIR/swordfish/...
#   SWORDFISH_ARCH_LABEL — a100/h100/h200, used as the default --arch-label
#                      when the caller does not pass one.
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

# The image has source baked at /work/swordfish. Cd there so relative result
# paths and `git`-style commands (rare) work as expected.
cd /work/swordfish

mkdir -p "${DATA_DIR}/swordfish"

echo "== swordfish-bench =="
echo "host:        $(hostname)"
echo "swordfish:   ${SWORDFISH_SHA:-unknown} (baked into image)"
echo "data_dir:    ${DATA_DIR}"
echo "arch_label:  ${ARCH_LABEL:-<unset>}"
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

exec python -m swordfish.runner "$@" "${inject_arch[@]}"
