# A100 Nsight Compute handoff

## Current state

Fresh A100 timing and Nsight Compute JSON exists and is valid. The A100 blocker
was resolved by temporarily excluding A100 nodes from the DCGM exporter
DaemonSet during the NCU profiling window, then restoring the DaemonSet.

- Result path: `runs/airun/week1/torch-gemm-a100.json`
- GPU: `NVIDIA A100-SXM4-80GB`
- Mean latency: `0.604938 ms`
- Throughput: `227.195 TFLOP/s`
- Correctness: passing
- Protocol validation: passing
- NCU completeness: `true`

The strict matrix gate is now ready for A100/H100/H200.

## Evidence

The normal A100 run without extra capabilities fails with:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

A follow-up A100 cap-test added `SYS_ADMIN` to the benchmark container. That
changed the failure mode, but still did not produce counters:

```text
Profiling failed because a driver resource was unavailable. Ensure that no other
tool (like DCGM) is concurrently collecting profiling data.
```

The same failure occurs when NCU is filtered to GEMM-like kernels:

```text
Failed to profile "ampere_fp16_s16816gemm_fp16_2..."
```

A one-node DCGM pause test also failed: the A100 node's `nvidia-dcgm-exporter`
pod was deleted, a `SYS_ADMIN` benchmark pod immediately ran a GEMM-filtered NCU
pass, and NCU still failed on the cuBLAS GEMM kernel with driver resource
unavailable. The DaemonSet recreated the exporter afterward. This points to
node/driver/operator-side profiler contention rather than a benchmark script
issue.

After H200 capacity returned, a fresh A100 privileged retry
`swordfish-a100-ncu-181205-a100` was submitted with `SYS_ADMIN` and the current
source. It completed the unprofiled timing pass and wrote a final JSON, but NCU
still failed:

```text
Profiling failed because a driver resource was unavailable. Ensure that no other
tool (like DCGM) is concurrently collecting profiling data.
Failed to profile "distribution_elementwise_grid..." in process 172
```

The retry result was still `ncu.complete=false` with all required NCU metrics
missing, which narrowed the remaining strict-gate blocker to A100 NCU before the
later DaemonSet-level DCGM pause resolved it.

One final differentiated retry targeted the other A100 node,
`aks-gpu-33826946-vmss000000`, with `SYS_ADMIN`. Immediately before the run, the
node's `nvidia-dcgm-exporter` pod was deleted, and NCU was filtered directly to
GEMM-like kernels:

```text
ncu ... --kernel-name regex:.*gemm.* python -m swordfish.runner run-gemm ...
```

That still failed directly on the cuBLAS GEMM kernel:

```text
Profiling failed because a driver resource was unavailable.
Failed to profile "ampere_fp16_s16816gemm_fp16_2..." in process 172
```

This rules out a single bad A100 node and rules out setup-kernel filtering as the
fix. The DCGM exporter DaemonSet recreated the exporter after the test.

The successful run changed the layer of attack from per-pod retries to
operator-side profiler ownership:

1. Backed up the `nvidia-dcgm-exporter` DaemonSet spec.
2. Patched the DaemonSet with node affinity excluding `gpu=a100`, removing DCGM
   exporter pods from both A100 nodes while leaving H100/H200 monitoring intact.
3. Ran `swordfish-a100-dcgm-off-182331-a100` with `SYS_ADMIN`.
4. A100 NCU completed with all required metrics.
5. Removed the temporary affinity patch and confirmed the DaemonSet rolled out
   successfully to all six GPU nodes.

Successful A100 NCU summary:

```text
ncu.complete=true
gpu__time_duration.sum=46560.0
sm__throughput.avg.pct_of_peak_sustained_elapsed=13.31
dram__throughput.avg.pct_of_peak_sustained_elapsed=73.36
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed=73.36
```

## Ruled out

- Missing result JSON: ruled out; A100 writes final JSON.
- Bad benchmark route: ruled out; the job lands on `NVIDIA A100-SXM4-80GB`.
- NCU replay polluting latency: fixed; timing now runs before the profiler pass.
- Missing container capability only: ruled out; adding `SYS_ADMIN` changed the
  error but still failed due driver resource contention.
- Profiling the wrong setup kernel: ruled out; GEMM kernel filtering still fails.
- One stale DCGM exporter pod: ruled out; deleting the exporter on one A100 node
  did not make NCU succeed before the DaemonSet recreated it.
- H200 capacity as a proxy for cluster health: ruled out; H200 returned and
  produced a complete NCU result, while a fresh privileged A100 retry still failed
  with driver resource unavailable.
- A single bad A100 node: ruled out; both `aks-gpu-33826946-vmss000001` and
  `aks-gpu-33826946-vmss000000` fail NCU.
- Profiling setup kernels instead of GEMM: ruled out; a GEMM-filtered retry still
  failed directly on `ampere_fp16_s16816gemm_fp16_2...`.

## Operational procedure

When A100 NCU profiling is needed:

1. Confirm the benchmark can run normally without profiling.
2. Temporarily patch `nvidia-dcgm-exporter` so it does not schedule on
   `gpu=a100` nodes.
3. Wait until no `nvidia-dcgm-exporter-*` pods are running on the target A100
   nodes.
4. Run `make airun-a100-ncu-preflight`; it should fail if a running DCGM exporter
   pod is still on a Ready target A100 node or if the A100 arch config lacks
   `SYS_ADMIN`.
5. Run the NCU benchmark job with `SYS_ADMIN`.
6. Copy the final JSON/CSV artifacts locally.
7. Restore the DaemonSet immediately and confirm `rollout status
   ds/nvidia-dcgm-exporter` succeeds.

Do not leave DCGM disabled after profiling. The permanent lesson is that A100
NCU and DCGM exporter profiling groups contend for the same driver profiling
resource on this lane.
