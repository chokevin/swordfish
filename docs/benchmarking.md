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

Latency and profiler metrics should come from separate passes. The bench script
first writes the timed `.raw.json` without Nsight Compute, then runs a second
NCU pass into a `.ncu.csv` (legacy CSV path triggered by `SWORDFISH_PROFILE=ncu`)
or rune's native `--profile-mode=ncu` (binary `.ncu-rep`), and finally attaches
the profiler summary to the unprofiled timing JSON. This keeps NCU replay
overhead out of `metrics.latency`.

If Nsight Compute reports `ERR_NVGPUCTRPERM`, the node driver is restricting
performance counters. NCU on A100 needs container `SYS_ADMIN`, which rune
profiles cannot currently request — this is a known limitation tracked in
`docs/airun/a100-ncu-blocker.md`. Do not mark NCU complete unless the attached
CSV/rep contains every required metric.

If Nsight Compute then changes to `Profiling failed because a driver resource
was unavailable`, check for DCGM/profiler contention. The A100 NCU lane requires
temporarily excluding `nvidia-dcgm-exporter` from A100 nodes during the NCU
profiling window, then restoring the DaemonSet and confirming rollout. This is
an operational profiling window, not a permanent monitoring change.

Use the matrix validator as the completion gate for cross-GPU GEMM runs:

```bash
uv run python -m swordfish.runner validate-gemm-matrix \
  --result-dir runs/rune/week1 \
  --prefix torch-gemm \
  --backend torch \
  --dtype fp16 \
  --m 4096 --n 4096 --k 4096 \
  --arch-labels a100 h100 h200 \
  --recursive \
  --require-ncu
```

`make validate-results` runs the same strict check against the local copied
artifact directory, `runs/rune/week1`, by default. Override `RESULT_DIR`
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
  --result-dir runs/rune/week1 \
  --recursive \
  --out docs/dashboard/results-index.json
```

`make dashboard-index` runs the same path using the configured
`RESULT_DIR`, recursive setting, and dashboard output path.

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
same strict matrix gate used by `make validate-results`, and summarizes the
indexed artifacts found under `RESULT_DIR`.

## Inspecting a finished run locally (Mac)

The day-to-day kernel-tuning loop is: edit kernel → submit to cluster → fetch
artifacts → open the trace in `ncu-ui` / `nsys-ui` on the Mac. The `inspect-run`
subcommand collapses fetch-and-open into one step.

```bash
# Submit with a profile mode (only ncu / nsys produce a binary trace; torch
# produces a chrome-trace JSON via swordfish.runner.profile_torch).
uv run python -m swordfish.runner submit-bench \
  --workload liger-rmsnorm --arch h100 --profile-mode ncu \
  --name sf-liger-rmsnorm-h100-tune-001

# ... wait for the job to finish ...

# Pull the result JSON + the .ncu-rep into runs/inspect/<name>/ and open the
# trace in ncu-ui (file association on macOS):
uv run python -m swordfish.runner inspect-run \
  sf-liger-rmsnorm-h100-tune-001 --profile-mode ncu
```

`inspect-run` is idempotent: re-running with the same name reuses cached files
unless `--overwrite` is passed. Use `--no-open` to fetch without launching the
GUI (useful in CI or when iterating on the artifact path). Without
`--profile-mode`, it fetches only the result JSON.

The `ncu-ui` and `nsys-ui` Mac clients are free downloads from
[developer.nvidia.com/tools-overview](https://developer.nvidia.com/tools-overview).
Once installed, double-clicking a `.ncu-rep` (or `open file.ncu-rep`) launches
the full Speed-of-Light dashboard with occupancy, source-attributed SASS, and
baseline comparison sets — the same view kernel engineers use to drive Liger /
Triton kernel work.
