# airun Week 1 protocol

This is the Week 1 smoke loop: run one standard torch/cuBLAS GEMM on A100,
H100, and H200, then persist one JSON result per architecture. The goal is
benchmark plumbing, not custom-kernel performance. In the current airun state,
A100/H100 are the safe default path; H200 must pass the preflight before any
H200 job is submitted.

## Monday: local schema smoke

Use CPU only to verify the JSON schema and CLI on a laptop. This does not claim
any performance number; it only proves the runner writes the right shape of
artifact.

```bash
uv run python -m swordfish.runner run-gemm \
  --backend torch \
  --m 32 --n 32 --k 32 \
  --dtype fp32 \
  --repeats 1 --warmup 0 --iters 1 \
  --device cpu --allow-cpu \
  --arch-label a100 \
  --out /tmp/swordfish-gemm-smoke.json
```

## Tuesday: cluster discovery

Before submitting jobs, record the routing contract for each architecture.

```bash
kubectl config current-context
kubectl get resourceclaimtemplate -n ray
kubectl get resourceflavors
kubectl get localqueue -A
kubectl get clusterqueue
kubectl get nodes --show-labels | grep -Ei 'a100|h100|h200|gpu|nvidia'
```

For each of A100, H100, and H200, capture:

- namespace
- queue / cluster queue
- DRA template or legacy GPU resource request
- node selector
- taints and tolerations
- priority class
- output path for JSON results

The discovered `voice-agent-flex` routing is captured in
`infra/airun/airun-gemm.voice-agent-flex.json`:

- context: `voice-agent-flex`
- namespace: `ray`
- queue: `fauna-train-queue`
- output PVC: `training-nfs`, mounted at `/data-nfs`
- A100 selector: `agentpool=gpu`, `nvidia.com/mig.config=all-disabled`
- H100 selector: `gpu=h100`
- H200 selector: `gpu=h200`

Copy the current source tree to the shared PVC, replace the source/ref
placeholders, then render the safe A100/H100 Kueue `Job`s:

```bash
uv run python -m swordfish.runner render-airun-gemm \
  --config infra/airun/airun-gemm.voice-agent-flex.json \
  --manifest-dir infra/airun/generated/week1 \
  --arch-labels a100 h100 \
  --dry-run-client
```

Review the manifests, then submit them with the same safe arch filter plus
`--apply`:

```bash
uv run python -m swordfish.runner render-airun-gemm \
  --config infra/airun/airun-gemm.voice-agent-flex.json \
  --manifest-dir infra/airun/generated/week1 \
  --arch-labels a100 h100 \
  --apply
```

The same flow is available as:

```bash
make airun-dry-run
make airun-apply
```

By default the Make targets render/apply only A100 and H100 while H200 is
blocked:

```bash
make airun-dry-run AIRUN_ARCH_LABELS="a100 h100"
```

Only include H200 after the H200 preflight passes:

```bash
make airun-h200-apply
```

Run the H200 preflight before submitting any H200 benchmark work:

```bash
make airun-h200-preflight
```

This renders and runs a fail-fast script that checks the configured Kueue route,
verifies the DRA `ResourceClaimTemplate` when one is configured, requires at
least one live node matching the H200 selector, and fails if the known orphan
blocker pod still exists. If it fails, do not submit more H200 jobs; fix capacity
or cleanup first. The current cleanup/capacity handoff is recorded in
[`docs/airun/h200-blocker-handoff.md`](../../docs/airun/h200-blocker-handoff.md).

`infra/airun/Dockerfile` is still available if you want an immutable image later,
but the default config runs from source on `training-nfs` using NVIDIA's PyTorch
container so early benchmark jobs do not wait on image publishing.

The generated jobs use the Kueue `kueue.x-k8s.io/queue-name` label, request one
GPU, run the same `4096^3 fp16` benchmark contract without profiler overhead,
run a separate Nsight Compute pass, and attach the NCU SOL summary back into the
final JSON.

### A100 Nsight Compute caveat

On the current `voice-agent-flex` lane, A100 Nsight Compute profiling contends
with `nvidia-dcgm-exporter` when the exporter is collecting DCGM profiling metric
groups. `SYS_ADMIN` is necessary for the benchmark pod, but not sufficient by
itself: NCU still fails with driver resource unavailable while DCGM owns the
profiling resource.

For A100 NCU runs, use a controlled profiling window:

1. Temporarily patch the `nvidia-dcgm-exporter` DaemonSet so it excludes
   `gpu=a100` nodes.
2. Wait until no exporter pods are running on the target A100 nodes.
3. Run `make airun-a100-ncu-preflight`; it fails if a running DCGM exporter pod
   is still on a Ready target A100 node or if the A100 arch config lacks
   `SYS_ADMIN`.
4. Run the A100 NCU benchmark job.
5. Restore the DaemonSet immediately.
6. Confirm `kubectl --context voice-agent-flex -n gpu-operator rollout status
   ds/nvidia-dcgm-exporter --timeout=300s` succeeds.

`make airun-apply` also runs this A100 preflight automatically whenever
`AIRUN_ARCH_LABELS` includes `a100`, so routine submissions fail before Kueue if
DCGM has not been paused for the profiling window.

Do not leave DCGM disabled after profiling. The detailed handoff is in
[`docs/airun/a100-ncu-blocker.md`](../../docs/airun/a100-ncu-blocker.md).

## Wednesday: GPU run command

Inside the scheduled GPU job/container, run:

```bash
uv run python -m swordfish.runner run-gemm \
  --backend torch \
  --m 4096 --n 4096 --k 4096 \
  --dtype fp16 \
  --repeats 5 --warmup 10 --iters 50 \
  --device auto \
  --arch-label a100 \
  --out /data-nfs/swordfish/week1/torch-gemm-a100.json
```

Repeat with `--arch-label h100` and `--arch-label h200`. The runner fails
loudly if the requested arch label does not match the CUDA device name.

## Thursday: attach Nsight Compute metrics

```bash
uv run python -m swordfish.runner run-gemm \
  --backend torch \
  --m 4096 --n 4096 --k 4096 \
  --dtype fp16 \
  --repeats 5 --warmup 10 --iters 50 \
  --device auto \
  --arch-label a100 \
  --out /data-nfs/swordfish/week1/torch-gemm-a100.raw.json

ncu --csv \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
  --target-processes all \
  --replay-mode kernel \
  uv run python -m swordfish.runner run-gemm \
    --backend torch \
    --m 4096 --n 4096 --k 4096 \
    --dtype fp16 \
    --repeats 5 --warmup 10 --iters 50 \
    --device auto \
    --arch-label a100 \
    --out /data-nfs/swordfish/week1/torch-gemm-a100.profile.raw.json \
  > /data-nfs/swordfish/week1/torch-gemm-a100.ncu.csv

uv run python -m swordfish.runner attach-ncu \
  --result /data-nfs/swordfish/week1/torch-gemm-a100.raw.json \
  --ncu-csv /data-nfs/swordfish/week1/torch-gemm-a100.ncu.csv \
  --out /data-nfs/swordfish/week1/torch-gemm-a100.json
```

Each final JSON records shape, dtype, timing config, host, commit SHA,
torch/CUDA versions, GPU metadata, latency samples, achieved TFLOP/s, estimated
bandwidth, rough SOL percentages, finite-output checksum, torch-reference error
fields, and NCU SOL metrics when attached.

For a local schema-only matrix smoke, generate all three arch-labeled JSON files
on CPU:

```bash
uv run python -m swordfish.runner run-gemm-matrix \
  --backend torch \
  --m 8 --n 8 --k 8 \
  --dtype fp32 \
  --repeats 1 --warmup 0 --iters 1 \
  --device cpu --allow-cpu \
  --out-dir /tmp/swordfish-week1-smoke
```

For the real airun run, validate the strict completion gate after A100/H100/H200
jobs finish:

```bash
make airun-validate-results
```

This checks copied local artifacts under
`runs/airun/week1/torch-gemm-{a100,h100,h200}.json` by default, searches
recursively for timestamped run subdirectories, and fails if any result is
missing or ambiguous, has the wrong arch/backend/shape, fails correctness, or
lacks complete NCU metrics. Override `AIRUN_RESULT_DIR=/data-nfs/swordfish/week1`
if you are running the validation from an environment with the NFS mount.

## Friday: publishable note

Write 10 lines:

1. GPU and driver/CUDA/torch versions.
2. Exact command.
3. Mean/p50/p95 latency.
4. TFLOP/s and rough compute SOL.
5. Estimated bandwidth and rough HBM SOL.
6. NCU SM throughput.
7. NCU DRAM throughput.
8. Whether output was finite.
9. Biggest caveat.
10. Next measurement.
