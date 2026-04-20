#!/usr/bin/env bash
# swordfish autoresearch inner entrypoint.
#
# Invoked by /usr/local/bin/swordfish-autoresearch (deploy/image/bootstrap.sh
# in the image) AFTER the repo has been cloned. CWD is the repo root, gh auth
# is already configured. Anything in this file can be changed without
# rebuilding the Docker image — push to main and re-run helm install.
#
# Inputs (env, all defaulted by bootstrap or chart):
#   SHAPES          shape set name (default voice)
#   IMPLS           comma-separated impls (default fp16,marlin)
#   REPEATS         repeats per impl (default 5)
#   REPO            owner/name (default chokevin/swordfish)
#   MARLIN_SHA      pin (default baked into image)
#   BRANCH_PREFIX   default "autoresearch/profile"
#   PR_DRAFT        "true"/"false" (default true)
#   GH_TOKEN        REQUIRED — set by bootstrap from secret

set -euo pipefail

SHAPES="${SHAPES:-voice}"
IMPLS="${IMPLS:-fp16,marlin}"
REPEATS="${REPEATS:-5}"
REPO="${REPO:-chokevin/swordfish}"
BRANCH_PREFIX="${BRANCH_PREFIX:-autoresearch/profile}"
PR_DRAFT="${PR_DRAFT:-true}"

SOURCE_SHA="$(git rev-parse HEAD)"
SOURCE_SHA_SHORT="$(git rev-parse --short=7 HEAD)"
echo "source SHA: ${SOURCE_SHA}"

# Use the container's pre-built torch (CUDA-matched). Install swordfish
# with --no-deps so pip doesn't yank torch/triton/numpy and replace them
# with PyPI wheels (which would break Marlin's ABI binding to NGC torch).
# Then install bench-only extras (tabulate/matplotlib/pandas) — none of
# these depend on torch, so they can't drag in a torch upgrade.
pip install --no-cache-dir --no-deps -e .
pip install --no-cache-dir tabulate matplotlib pandas

# Sanity: confirm we did NOT replace torch.
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} '
  f'device_count={torch.cuda.device_count()} '
  f'sm={torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None}')"
python -c "import marlin; print('marlin import OK')"

echo
echo "=== profile run: shapes=${SHAPES} impls=${IMPLS} repeats=${REPEATS} ==="
SHAPES="${SHAPES}" IMPLS="${IMPLS}" REPEATS="${REPEATS}" \
  bench/profile_marlin.sh

# Newest run dir under docs/profiling/marlin/.
RUN_DIR="$(ls -td docs/profiling/marlin/*/ | head -1)"
RUN_TS="$(basename "${RUN_DIR%/}")"
echo "run dir: ${RUN_DIR}"

echo
echo "=== roofline plot ==="
python -m bench.roofline "${RUN_DIR}" --gpu a100-80gb-sxm \
  || echo "warning: roofline failed — likely missing ncu CSVs (continuing)"

echo
echo "=== summary table ==="
SUMMARY_MD="${RUN_DIR}/SUMMARY.md"
python - "${RUN_DIR}" "${SOURCE_SHA_SHORT}" "${SHAPES}" "${IMPLS}" "${REPEATS}" "${RUN_TS}" "${MARLIN_SHA:-unknown}" > "${SUMMARY_MD}" <<'PY'
import csv, json, sys
from pathlib import Path
run_dir, sha_short, shapes, impls, repeats, run_ts, marlin_sha = sys.argv[1:8]
run_dir = Path(run_dir)
rows = list(csv.DictReader((run_dir / "results.csv").open()))
manifest = json.loads((run_dir / "manifest.json").read_text())
env = manifest["env"]

print(f"# Autoresearch run `{run_ts}`\n")
print(f"- **source SHA:** `{sha_short}`")
print(f"- **GPU:** {env.get('gpu_name','?')} (cc {env.get('gpu_cc','?')}, "
      f"{env.get('gpu_mem_gb','?')} GB)")
print(f"- **CUDA / torch / triton:** {env.get('torch_cuda','?')} / "
      f"{env.get('torch','?')} / {env.get('triton','?')}")
print(f"- **shapes:** `{shapes}`  **impls:** `{impls}`  **repeats:** {repeats}")
print(f"- **marlin SHA:** `{marlin_sha}`\n")
print("## Results\n")
print("| shape | impl | ms_mean | ms_p95 | TFLOPS | speedup vs fp16 | error |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    if r.get("error"):
        print(f"| {r['name']} | {r['impl']} | — | — | — | — | {r['error']} |")
    else:
        ms = float(r['ms_mean']); p95 = float(r['ms_p95'])
        tf = float(r['tflops_mean']); sp = float(r['speedup_vs_fp16'])
        print(f"| {r['name']} | {r['impl']} | {ms:.3f} | {p95:.3f} | {tf:.1f} | x{sp:.2f} | |")
print("\n![roofline](./roofline.png)")
PY

echo
echo "=== update docs/profiling/INDEX.md ==="
INDEX="docs/profiling/INDEX.md"
if [[ ! -f "${INDEX}" ]]; then
  cat > "${INDEX}" <<'EOF'
# Autoresearch run index

One row per profiling run produced by the swordfish-autoresearch chart.
Newest first. PR column links to the draft PR carrying the artifacts.

| timestamp (UTC) | source SHA | shapes | impls | GPU | 8b-b1 marlin TFLOPS | run dir | PR |
|---|---|---|---|---|---|---|---|
EOF
fi

HEADLINE_TF="$(python - "${RUN_DIR}/results.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
hit = next((r for r in rows if r['name']=='8b-b1' and r['impl']=='marlin' and not r.get('error')), None)
print(f"{float(hit['tflops_mean']):.1f}" if hit else "n/a")
PY
)"
GPU_NAME="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['env'].get('gpu_name','?'))" "${RUN_DIR}/manifest.json")"
NEW_ROW="| ${RUN_TS} | \`${SOURCE_SHA_SHORT}\` | ${SHAPES} | ${IMPLS} | ${GPU_NAME} | ${HEADLINE_TF} | [\`${RUN_TS}/\`](./marlin/${RUN_TS}/) | _pending_ |"
awk -v row="${NEW_ROW}" '
  /^\|---\|---\|---\|/ && !done { print; print row; done=1; next }
  { print }
' "${INDEX}" > "${INDEX}.tmp" && mv "${INDEX}.tmp" "${INDEX}"

echo
echo "=== archive run dir to NFS (long-term, off-repo) ==="
NFS_DIR="/data-nfs/swordfish-autoresearch/${RUN_TS}"
if [[ -d /data-nfs ]]; then
  mkdir -p "${NFS_DIR}"
  cp -r "${RUN_DIR}/." "${NFS_DIR}/" || echo "warning: NFS archive failed (continuing)"
  echo "archived to ${NFS_DIR}"
else
  echo "skipped: /data-nfs not mounted"
fi

echo
echo "=== commit + push branch ==="
BRANCH="${BRANCH_PREFIX}-${RUN_TS}-${SOURCE_SHA_SHORT}"
git checkout -b "${BRANCH}"
git add docs/profiling/
git commit -m "autoresearch: profile run ${RUN_TS}

source SHA: ${SOURCE_SHA}
shapes:     ${SHAPES}
impls:      ${IMPLS}
repeats:    ${REPEATS}
GPU:        ${GPU_NAME}
marlin SHA: ${MARLIN_SHA:-unknown}

Headline (8b-b1 marlin): ${HEADLINE_TF} TFLOPS
"
git push origin "${BRANCH}"

echo
echo "=== open ${PR_DRAFT:+draft }PR ==="
DRAFT_FLAG=""
[[ "${PR_DRAFT}" == "true" ]] && DRAFT_FLAG="--draft"
PR_URL="$(gh pr create \
  --repo "${REPO}" \
  --base main \
  --head "${BRANCH}" \
  --title "autoresearch: profile run ${RUN_TS} (${SOURCE_SHA_SHORT})" \
  --body-file "${SUMMARY_MD}" \
  --label autoresearch \
  ${DRAFT_FLAG} || echo '')"
echo "PR: ${PR_URL}"

# Backfill the INDEX row's "_pending_" cell with the actual PR URL.
if [[ -n "${PR_URL}" ]]; then
  sed -i "0,/_pending_/{s|_pending_|[link](${PR_URL})|}" "${INDEX}"
  git add "${INDEX}"
  git commit -m "autoresearch: link PR in INDEX.md"
  git push origin "${BRANCH}"
fi

echo
echo "DONE. PR: ${PR_URL:-<not opened>}"
echo "Branch: ${BRANCH}"
echo "Run dir on branch: ${BRANCH}:${RUN_DIR}"
