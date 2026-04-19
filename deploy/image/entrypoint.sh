#!/usr/bin/env bash
# swordfish autoresearch entrypoint.
#
# What it does, in order:
#   1. Clones swordfish at $REF.
#   2. Installs swordfish into the pre-baked Python env (NGC torch, not uv venv).
#   3. Runs bench/profile_marlin.sh — produces nsys/ncu/perfetto artifacts.
#   4. Generates roofline.png from the ncu CSV.
#   5. Renders SUMMARY.md from results.csv (PR body).
#   6. Updates docs/profiling/INDEX.md with one new row.
#   7. Archives the run dir to /data-nfs (long-term, off-repo).
#   8. Commits to a per-run branch and opens a draft PR.
#
# Inputs (all env, all defaulted):
#   REF             git ref of swordfish (default main)
#   SHAPES          shape set name (default voice)
#   IMPLS           comma-separated impls (default fp16,marlin)
#   REPEATS         repeats per impl (default 5)
#   REPO            owner/name (default chokevin/swordfish)
#   MARLIN_SHA      pin (default baked into image)
#   BRANCH_PREFIX   default "autoresearch/profile"
#   PR_DRAFT        "true"/"false" (default true)
#   GH_TOKEN        REQUIRED — repo:contents + pull-requests write
#   GIT_USER_EMAIL  default autoresearch@swordfish.bot
#   GIT_USER_NAME   default swordfish-autoresearch

set -euo pipefail

REF="${REF:-main}"
SHAPES="${SHAPES:-voice}"
IMPLS="${IMPLS:-fp16,marlin}"
REPEATS="${REPEATS:-5}"
REPO="${REPO:-chokevin/swordfish}"
BRANCH_PREFIX="${BRANCH_PREFIX:-autoresearch/profile}"
PR_DRAFT="${PR_DRAFT:-true}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-autoresearch@swordfish.bot}"
GIT_USER_NAME="${GIT_USER_NAME:-swordfish-autoresearch}"

if [[ -z "${GH_TOKEN:-}" ]]; then
  echo "FATAL: GH_TOKEN not set. Mount the gh-token secret." >&2
  exit 1
fi

export GH_TOKEN
git config --global user.email "${GIT_USER_EMAIL}"
git config --global user.name "${GIT_USER_NAME}"
git config --global credential.helper '!f() { echo "username=x-access-token"; echo "password=${GH_TOKEN}"; }; f'

cd /work
echo "=== clone swordfish @ ${REF} ==="
git clone --depth 50 "https://github.com/${REPO}.git" swordfish
cd swordfish
git fetch --depth 50 origin "${REF}"
git checkout "${REF}"
SOURCE_SHA="$(git rev-parse HEAD)"
SOURCE_SHA_SHORT="$(git rev-parse --short=7 HEAD)"
echo "source SHA: ${SOURCE_SHA}"

# Use the container's pre-built torch (CUDA-matched). Editable install adds
# swordfish + bench extras without disturbing torch/triton.
pip install --no-cache-dir -e ".[bench]"

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
