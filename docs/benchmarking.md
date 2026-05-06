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

## TriMul outgoing tuning handoff

The GPUMODE outgoing TriMul work lives on PR
[`#7`](https://github.com/chokevin/swordfish/pull/7), branch
`chokevin/trimul-outgoing-20260506`. The current fused-tail iteration landed in
commit `26c543b` (`trimul: fuse tail norm gate`).

### What changed

- `submission.py` now splits the packed triangle path into explicit pack,
  packed-matmul, and unpack stages so `--profile-ops` reports
  `triangle_pack_left`, `triangle_pack_right`, `triangle_matmul`, and
  `triangle_unpack` instead of one opaque `triangle` phase.
- `--gate-pack-backend {auto,torch,triton}` exists for the fused
  mask/gate/pack prototype. It is intentionally **not** selected by `auto`
  because the measured Triton prototype regressed.
- `--tail-backend {auto,torch,triton}` controls a fused Triton output
  layernorm plus output-gate kernel. `auto` promotes it for CUDA
  `H=128,N<=256` and keeps larger shapes on the previous PyTorch path.

### H200 evidence

Default `auto` with fused tail, all reference checks passing:

| Shape | Mean latency | Max abs error |
| --- | ---: | ---: |
| `B=2,N=256,C=128,H=128,normal,nomask` | `1.1111 ms` | `0.01318` |
| `B=2,N=256,C=128,H=128,cauchy,nomask` | `1.1085 ms` | `0.01000` |
| `B=2,N=256,C=384,H=128,normal,masked` | `1.4953 ms` | `0.01095` |
| `B=1,N=512,C=128,H=128,normal,nomask` | `3.7624 ms` | `0.00626` |

Isolated first-shape comparisons:

| Variant | Mean latency | Takeaway |
| --- | ---: | --- |
| no new fusion | `1.3298 ms` | baseline after BF16 projection policy |
| tail-only | `1.1087 ms` | promoted winner |
| gate-pack-only | `1.4178 ms` | correct but slower |
| gate-pack + tail | `1.1920 ms` | tail win partly offset by gate-pack regression |

The packed triangle matmul itself is not the dominant H200 N=256 cost anymore:
`triangle_matmul` measured about `0.054 ms`; pack-left, pack-right, and unpack
each measured around `0.11 ms`.

### Do not repeat

Do not continue the current channel-at-a-time `gate_mask_pack` design. It reads
`projected [B,N,N,5H]` one hidden channel per program, which loses coalescing
over the contiguous hidden lanes. A second attempt should either coalesce hidden
lanes in the projection layout or move the fusion into a projection epilogue.

### Next fresh-eyes target

For a real push toward `0.5 ms`, attack materialization and layout traffic
around input layernorm, stacked projection, gate/mask, pack/unpack, and final
projection. Another standalone elementwise Triton pass is unlikely to be enough;
a projection epilogue or custom CUDA/CUTLASS-style path is the more plausible
next step.

Evidence artifacts from this iteration are under `runs/trimul/` in the local
worktree, with names starting `sf-trimul-tailauto-*` and
`sf-trimul-fuse-*`.

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

### Reading kernel-level detail without ncu-ui (`ncu-summary`)

The `.ncu-rep` binary format is proprietary and only readable with NVIDIA's
`ncu-ui`. For agent-friendly inspection (or quick CLI scanning), there's a
companion subcommand that summarizes the *CSV* form of an NCU profile:

```bash
# Convert a fetched .ncu-rep to CSV (requires Nsight Compute installed locally
# OR you can have the cluster do it — see swordfish-bench.sh's legacy ncu mode):
ncu --import runs/inspect/<name>/<name>.ncu-rep --csv \
  > runs/inspect/<name>/<name>.ncu.csv

# Then summarize per-kernel:
uv run python -m swordfish.runner ncu-summary \
  runs/inspect/<name>/<name>.ncu.csv --top 10
```

The output is a sorted per-kernel table: invocations, total/mean/max time,
% of wall, and the headline Speed-of-Light metrics (SM%, MEM%, DRAM%). It's
the agent-readable substitute for the ncu-ui kernel list.

The week-1 GEMM CSVs are checked in as canonical examples — try
`ncu-summary runs/airun/week1/torch-gemm-h100.ncu.csv` for a feel.

When `inspect-run --profile-mode ncu` finds a `.ncu.csv` companion file in the
local cache directory (e.g. one you converted manually with `ncu --import`, or
one the cluster wrote alongside the .ncu-rep), it auto-prints the same
per-kernel summary on stdout. Otherwise it prints a stderr hint pointing you
at the conversion command above.

**What `ncu-summary` is NOT a substitute for:** source-line attribution, SASS
view, occupancy widget, the full Speed-of-Light radial chart, or any of the
metrics not in NCU's default `--csv` output set. For those, `ncu-ui` against
the original `.ncu-rep` remains the right tool.
