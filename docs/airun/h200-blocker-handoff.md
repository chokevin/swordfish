# H200 blocker handoff

## Current state

H200 capacity returned and the single-arch H200 GEMM rerun completed
successfully. This handoff is now historical for the capacity outage and
documents the remaining tombstoned orphan pod.

- Fresh H200 result: `runs/airun/week1/torch-gemm-h200.json`
- GPU: `NVIDIA H200`
- Mean latency: `0.182761 ms`
- Throughput: `752.014 TFLOP/s`
- Correctness: passing
- NCU completeness: `true`
- Run: `swordfish-gemm-180223-h200`
- Node: `flex-h200-zjj8s`

The strict completion gate now passes for A100/H100/H200. The later A100 NCU
unblock is recorded in [`docs/airun/a100-ncu-blocker.md`](./a100-ncu-blocker.md).

## Historical blocker

Known blocker:

- Context: `voice-agent-flex`
- Namespace: `ray`
- NodePool: `flex-h200`
- NodeClass: `h200-eastus2`
- Instance type: `Standard_ND96isr_H200_v5`
- Resource group in failing NodeClaim events: `voice-agent-flex-h200-rg`
- Orphan pod: `sf-gemm-133050-h200-8wr7p`

Previously observed behavior:

- `flex-h200` was Ready but had 0 nodes.
- Fresh `flex-h200-*` NodeClaims repeatedly failed with Azure
  `InsufficientCapacityError`.
- The orphan pod still had `batch.kubernetes.io/job-tracking` after deletion.
- Normal and admin `kubectl patch pod ... finalizers` dry-runs fail API
  validation with `spec.tolerations: Forbidden`.
- The pod `finalize` subresource is not served for this resource.
- After H200 nodes returned, the orphan pod was still present and deletion
  marked, but H200 preflight now treats a `Failed` pod with a deletion timestamp
  as a warning once live Ready H200 nodes exist.

## Fast local check

Run:

```bash
make airun-h200-preflight
```

Expected now: exit code 0 when live Ready H200 nodes exist. If H200 capacity
regresses, the preflight should still fail before submission.

Historical blocked output included:

- `ERROR: no live nodes match selector gpu=h200`
- `ERROR: known blocker pod sf-gemm-133050-h200-8wr7p still exists`

If the preflight fails, do not submit H200 Jobs.

After the unblock criteria are met, submit H200 through:

```bash
make airun-h200-apply
```

This target runs the same preflight first and only submits the H200 manifest if
the preflight exits 0.

## Evidence commands

```bash
kubectl --context voice-agent-flex -n ray get localqueue fauna-train-queue -o wide
kubectl --context voice-agent-flex get clusterqueue gpu-cluster-queue -o wide
kubectl --context voice-agent-flex get nodepool flex-h200 -o wide
kubectl --context voice-agent-flex get nodes -l gpu=h200 \
  -L gpu,nvidia.com/gpu.product,kubernetes.azure.com/agentpool
kubectl --context voice-agent-flex get nodeclaims -o wide \
  --sort-by=.metadata.creationTimestamp
kubectl --context voice-agent-flex get events -A --sort-by=.lastTimestamp \
  | grep -Ei 'flex-h200|h200|InsufficientCapacity|nodeclaim'
kubectl --context voice-agent-flex -n ray get pod sf-gemm-133050-h200-8wr7p \
  -o jsonpath='{.status.phase}{"\n"}{.metadata.deletionTimestamp}{"\n"}{.metadata.finalizers}{"\n"}'
```

## Admin/control-plane cleanup still needed

This is no longer blocking the H200 benchmark while live H200 nodes exist, but
the old pod should still be cleaned up outside this normal CLI session:

1. The orphan pod `sf-gemm-133050-h200-8wr7p` is removed through a
   control-plane/admin path that bypasses the current pod spec validation issue.
2. If H200 capacity regresses again, Azure capacity must be restored for
   `Standard_ND96isr_H200_v5` or `flex-h200` must move to an available region/SKU.

Do not use destructive broad cleanup commands. Target only the known orphan pod
or the known H200 NodeClaims/NodePool after confirming ownership.

## Regressed-state unblock criteria

If H200 capacity disappears again, the H200 path is safe to retry only when all
are true:

1. `make airun-h200-preflight` exits 0.
2. `kubectl --context voice-agent-flex get nodes -l gpu=h200` returns at least
   one Ready node.
3. The known orphan pod is either gone, or is `Failed` with a deletion timestamp
   and live Ready H200 nodes exist.
4. Recent `flex-h200-*` NodeClaim events no longer show Azure
   `InsufficientCapacityError`.

After that, rerun only the H200 GEMM job first. The matrix is complete only when
`make airun-validate-results` exits 0 with complete NCU metrics for every
requested architecture.
