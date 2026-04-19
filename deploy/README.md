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

## How airun-aligned is this?

Mapped against the airun 5-layer stack (see kstack `airun-triage` skill):

| Layer | Our use | Notes |
|---|---|---|
| L1 Ray | **Skipped** | Single-GPU run-to-completion; plain Job is the right shape. |
| L2 Kueue | `training-queue` → `gpu-cluster-queue`, `priorityClass: eval-priority` | Lowest priority — preemptible by every real workload. Diagnostic profiling never blocks training. |
| L3 k8s scheduler | `nodeSelector: agentpool=gpu` + toleration `sku=gpu` | Exactly matches the `gpu-a100-full` ResourceFlavor's nodeLabels + tolerations. |
| L4 GPU/MIG | DRA `resourceClaimTemplate: full-gpu` | Full A100, **not** MIG. Profiling on a MIG slice would distort roofline (different SM/L2/HBM ratios per slice) and invalidate comparison against Marlin's published numbers. |
| L5 Node pool | Single region (manager) | Could federate via MultiKueue if A100 contended — see "scaling out" below. |

When something goes wrong, `/kstack-airun-triage` walks this same ladder
top-down. The mapping above tells you which layer is opinionated for *us*
vs which is just convention.

## Kueue quota: how much pain is "set up another tenant"?

Three artifacts, two if you're piggybacking on someone else's flavor.

**Existing on voice-agent-flex** (you can read these as templates):

```bash
kubectl get resourceflavors                         # gpu-a100-full, gpu-h100-full, gpu-mig-*
kubectl get clusterqueue gpu-cluster-queue -o yaml  # 16 A100 quota; cohort=none
kubectl get localqueue   -n ray training-queue      # → gpu-cluster-queue
```

**To carve out a quota slice for a new tenant (~10 min of YAML):**

```yaml
# 1. ClusterQueue — quota lives here. Reuse an existing ResourceFlavor.
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata: { name: swordfish-cq }
spec:
  cohort: research                       # share borrowable headroom with fauna/gura
  namespaceSelector:
    matchExpressions:
      - { key: kubernetes.io/metadata.name, operator: In, values: [ray] }
  queueingStrategy: BestEffortFIFO
  preemption:
    withinClusterQueue: LowerPriority
    reclaimWithinCohort: Any
    borrowWithinCohort: { policy: Never }
  resourceGroups:
    - coveredResources: [cpu, memory, nvidia.com/gpu]
      flavors:
        - name: gpu-a100-full
          resources:
            - { name: cpu,             nominalQuota: "8" }
            - { name: memory,          nominalQuota: "48Gi" }
            - { name: nvidia.com/gpu,  nominalQuota: "1" }
---
# 2. LocalQueue — namespace-scoped pointer to the CQ.
apiVersion: kueue.x-k8s.io/v1beta2
kind: LocalQueue
metadata: { name: swordfish-queue, namespace: ray }
spec: { clusterQueue: swordfish-cq }
```

That's it: two objects (~30 lines) and you have an isolated, preemption-policy'd
slice of the cluster. **Easy.**

**The actually-painful parts** (have already been paid for you here):

| Setup task | Pain | Notes |
|---|---|---|
| Install Kueue controller | Low — one helm chart | `oci://registry.k8s.io/kueue/charts/kueue` v0.17+ for DRA-aware admission. |
| Define `ResourceFlavor`s | **Medium** — must match real node labels and any GPU partitioning scheme (MIG profiles, DRA device classes, hyperscaler taints). Get this wrong and *nothing* admits. | Already done: `gpu-a100-full`, `gpu-h100-full`, `gpu-mig-{1g10gb,2g20gb,3g40gb}`, `gpu-multikueue`. |
| Decide cohort topology | **High** — preemption boundaries are organizational, not technical. "Can serving evict training to reclaim a GPU?" is a people question. | Already decided: serving CQ is *cohort-less* (never preemptible from training); training CQs share `research` cohort and may borrow each other's headroom. |
| DRA + Kueue accounting | **High** — Kueue v0.17 adds DRAExtendedResources beta but you still need `deviceClassMappings` config to gate admission on DRA claims. Without it, Kueue admits DRA pods on CPU/memory only and the scheduler does the GPU placement. | Voice-agent-flex deferred this; admission is CPU/mem-only. We work fine in this mode. |
| MultiKueue federation (cross-region/cross-cluster) | **High** — separate manager ↔ worker Kueue installs, secret distribution, ResourceFlavor naming agreement, network reachability. | Already wired: `gpu-multikueue` flavor exists. We don't *use* it (single-region is enough for one A100), but the path is there. |
| PriorityClasses | Low — one YAML, four classes | Already exist: `serving=300`, `deepspeed=200`, `lora=100`, `eval=50`. |

**Bottom line for swordfish autoresearch:** since voice-agent-flex already has
ResourceFlavors, a CQ with A100 quota, priority classes, DRA templates, the
gh-token secret, and the blob/nfs PVCs, our marginal setup cost is **zero new
cluster objects**. We piggyback on `training-queue` + `eval-priority` and pay
only the polite-citizen tax of small CPU/memory requests.

If/when this collides with real training, the right escalation is a dedicated
`swordfish-cq` (the YAML above), not arguing about quota in a shared queue.

## Scaling out (when one A100 isn't enough)

The Helm chart is a single Job today — one shape sweep at a time. Two natural
fan-outs:

1. **Many shapes in parallel.** Loop over shapes in a driver shell script,
   `helm install` once per shape with `--set run.shapes=<one>`. N concurrent
   Jobs, each holding one A100 claim, all admit through `training-queue`
   subject to the CQ's 16-GPU cap.
2. **Many GPUs across regions.** Switch the LocalQueue to one whose
   ClusterQueue uses the `gpu-multikueue` ResourceFlavor — Kueue federates
   the workload to a worker cluster with capacity. No chart change needed;
   only `--set run.queue=<multikueue-queue>`.

Neither requires Ray. Neither requires touching this chart's templates.

## Why not RayJob?

Marlin profiling is a **single-node, single-GPU, run-to-completion** workload;
nothing distributed about it. RayJob adds a Ray head + worker pod, two more
pods to debug, and the per-shape work is already trivially serial. Plain Job
keeps the failure modes visible and the YAML one screen long.

If we ever want to fan shapes out in parallel (one A100 per shape), the
right move is N parallel Jobs from a small driver script — still no Ray.
