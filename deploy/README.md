# swordfish autoresearch

A one-shot Helm chart that runs `bench/profile_marlin.sh` on an A100 in the
**voice-agent-flex** cluster and lands the resulting nsys/ncu/Perfetto/roofline
artifacts on a per-run branch + draft PR. The PR is the iteration surface:
diff two runs, comment on a kernel, merge to land the writeup.

## Layout

```
deploy/
├── charts/swordfish-autoresearch/   # the Helm chart (Job + DRA + Kueue)
├── image/                           # Dockerfile + entrypoint + build.sh
└── values/voice-agent-flex.yaml     # cluster-specific overlay
```

## What one run produces

On a successful run the pod commits, to branch `autoresearch/profile-<ts>-<sha>`:

- `docs/profiling/marlin/<ts>/results.csv` + `manifest.json` (env header)
- `docs/profiling/marlin/<ts>/trace.json` (Perfetto Chrome trace)
- `docs/profiling/marlin/<ts>/<shape>.nsys-rep` + `.sqlite`
- `docs/profiling/marlin/<ts>/<shape>.ncu-rep` + `.ncu.csv`
- `docs/profiling/marlin/<ts>/roofline.png`
- `docs/profiling/marlin/<ts>/SUMMARY.md` (becomes the PR body)
- `docs/profiling/INDEX.md` (catalog row appended)

…and opens a draft PR titled `autoresearch: profile run <ts> (<sha>)` against
`main`, labeled `autoresearch`.

## Iteration loop

1. Land a code change on `main` (or a branch).
2. Trigger a profile run — the Job re-clones the ref and re-profiles.
3. Read the draft PR's diff against the previous run's PR — same files,
   different numbers. nsys/ncu/Perfetto traces all open from the PR file view.
4. Update `docs/profiling/marlin-bottlenecks.md` based on findings; merge.

`docs/profiling/INDEX.md` is the catalog: timestamp, source SHA, GPU,
headline 8b-b1 TFLOPS, link to the PR. Sortable by hand; sortable by tooling.

## Build the image

```bash
cd deploy/image
REGISTRY=ghcr.io/chokevin TAG=$(git -C ../.. rev-parse --short HEAD) \
  ./build.sh push
```

The image bakes Marlin at the SHA pinned in the Dockerfile so cold-start
doesn't pay the build cost. Bump `MARLIN_SHA` in `deploy/image/Dockerfile`
**and** `docs/profiling/RUN_ME_ON_A100.md` together.

## Trigger a profile run

```bash
helm install \
  swordfish-profile-$(date -u +%Y%m%dT%H%M%SZ) \
  ./deploy/charts/swordfish-autoresearch \
  -n ray \
  -f ./deploy/values/voice-agent-flex.yaml \
  --set run.ref=main \
  --set run.shapes=voice \
  --set run.impls=fp16,marlin
```

Override anything from `values.yaml` via `--set`. Common overrides:

| flag | what | when |
|---|---|---|
| `--set run.ref=<sha-or-branch>` | which swordfish to profile | every run |
| `--set run.shapes=full` | wider shape sweep | weekly sweep |
| `--set run.impls=fp16,marlin,swordfish` | once Triton kernel is in | W2+ |
| `--set run.image=...:<tag>` | use a non-`latest` image | reproducibility |
| `--set run.prDraft=false` | open as ready PR | for landed runs |
| `--set run.priorityClass=lora-priority` | bump priority | when contended |

## Watch + debug

```bash
# what's queued / admitted
kubectl get jobs,workloads -n ray -l app=swordfish-autoresearch

# logs
kubectl logs -n ray -l app=swordfish-autoresearch -f --tail=200

# per-step output: clone -> install -> profile -> commit -> PR
# the entrypoint emits clearly delimited section headers ("=== ... ===")
```

If the Job hangs in the queue: check the `Workload` object — most often
`AdmissionCheck` is failing because no worker cluster has a free A100, or
the resource flavor is missing. See voice-agent-flex MultiKueue runbook.

## Cluster prerequisites

(All present on voice-agent-flex per `voice-agent/deploy/values/manager.yaml`;
listed here so a fresh cluster operator knows the contract.)

- Kueue `LocalQueue` named `training-queue` with admission to a
  `ClusterQueue` that has A100 GPU quota.
- DRA `ResourceClaimTemplate` named `full-gpu` (one full A100).
- PriorityClass `eval-priority` (or `lora-priority`).
- PVC `blob-training` mounted at `/data` (Azure Blob CSI).
- PVC `training-nfs` mounted at `/data-nfs` (long-term archive).
- Secret `gh-token` with key `token` — GitHub PAT scoped to
  `repo:contents:write` + `pull-requests:write` on `chokevin/swordfish`.
- nodeSelector `agentpool=gpu` and toleration `sku=gpu:NoSchedule`
  match the existing GPU pool taints.

## Why not RayJob?

Marlin profiling is a **single-node, single-GPU, run-to-completion** workload;
nothing distributed about it. RayJob adds a Ray head + worker pod, two more
pods to debug, and the per-shape work is already trivially serial. Plain Job
keeps the failure modes visible and the YAML one screen long.

If we ever want to fan shapes out in parallel (one A100 per shape), the
right move is N parallel Jobs from a small driver script — still no Ray.
