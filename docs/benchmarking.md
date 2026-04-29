# Benchmark result protocol

Every benchmark JSON must make cross-GPU comparisons explicit. A result is not
comparable across A100/H100/H200 unless it records:

- `config.scope`: the measured unit, such as `gemm`, `block`, or `model`.
- `config.backend`: the implementation that actually ran, such as `torch`,
  `triton`, or `cutlass`.
- `config.shape`: the shape that defines the work.
- `config.dtype`: the operand/model dtype.
- `env.git_sha` and `env.git_dirty`: exact source provenance.
- `env.torch`, `env.torch_cuda`, `env.cuda_driver`, `env.nvidia_driver`,
  `env.triton`, and `env.ncu`: toolchain provenance.
- `env.gpu_name`, `env.gpu_class`, and `env.gpu_cc`: actual device identity.
- `correctness`: finite-output status plus checksum and reference-error fields
  where applicable.
- `metrics.latency`: latency samples and summary statistics.
- `ncu.complete` and `ncu.missing_metrics`, when an Nsight Compute CSV is
  attached, so partial profiler output cannot be mistaken for a complete SOL
  summary.

`swordfish.runner.schema.validate_result_protocol` enforces the common fields.
Individual benchmarks can add richer metrics, but they should not omit these
fields or change their meaning.

Latency and profiler metrics should come from separate passes. The airun job
script first writes the timed `.raw.json` without Nsight Compute, then runs a
second NCU pass into a `.ncu.csv`, and finally attaches the profiler summary to
the unprofiled timing JSON. This keeps NCU replay overhead out of
`metrics.latency`.

If Nsight Compute reports `ERR_NVGPUCTRPERM`, the node driver is restricting
performance counters. Airun arch configs can add a per-architecture
`container_security_context` such as `capabilities.add: ["SYS_ADMIN"]` for an
explicit test run, if cluster policy allows it. Do not mark NCU complete unless
the attached CSV contains every required metric.

If Nsight Compute then changes to `Profiling failed because a driver resource
was unavailable`, check for DCGM/profiler contention. The current A100 airun lane
requires temporarily excluding `nvidia-dcgm-exporter` from A100 nodes during the
NCU profiling window, then restoring the DaemonSet and confirming rollout. This
is an operational profiling window, not a permanent monitoring change. Use
`make airun-a100-ncu-preflight` before submission; `make airun-apply` invokes it
automatically when `AIRUN_ARCH_LABELS` includes `a100`.

Use the matrix validator as the completion gate for cross-GPU GEMM runs:

```bash
uv run python -m swordfish.runner validate-gemm-matrix \
  --result-dir runs/airun/week1 \
  --prefix torch-gemm \
  --backend torch \
  --dtype fp16 \
  --m 4096 --n 4096 --k 4096 \
  --arch-labels a100 h100 h200 \
  --recursive \
  --require-ncu
```

`make airun-validate-results` runs the same strict check against the local copied
artifact directory, `runs/airun/week1`, by default. Override `AIRUN_RESULT_DIR`
when validating directly against an NFS-mounted result directory. The gate fails
until every requested architecture has a matching JSON result, correct arch
provenance, passing correctness fields, and complete NCU metrics.

To compare completed JSON artifacts side by side:

```bash
uv run python -m swordfish.runner compare-results \
  --result runs/torch-gemm-a100.json runs/triton-gemm-a100.json \
  --out /tmp/swordfish-comparison.md
```

The comparison table includes benchmark/backend/GPU/dtype/shape, mean latency,
speedup versus the first result, TFLOP/s when present, correctness flags, NCU
completeness, and common protocol validation status.

To build a machine-readable index for dashboards or static publishing:

```bash
uv run python -m swordfish.runner index-results \
  --result-dir runs/airun/week1 \
  --recursive \
  --out docs/dashboard/results-index.json
```

`make dashboard-index` runs the same path using the configured
`AIRUN_RESULT_DIR`, recursive setting, and dashboard output path.

The index skips non-result JSON files and `.raw.json` intermediate benchmark
outputs by default. It records one compact row per final benchmark result,
including provenance, latency, throughput, correctness, NCU completeness, and
protocol-validation errors.

To render a human-readable completion report without hand-copying validator
errors:

```bash
make completion-report
```

The report writes `docs/dashboard/completion-report.md` by default, includes the
same strict matrix gate used by `make airun-validate-results`, and summarizes the
indexed artifacts found under `AIRUN_RESULT_DIR`.
