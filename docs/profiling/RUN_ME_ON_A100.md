# RUN_ME_ON_A100.md

This Week-1 work needs an A100 to produce its actual deliverables. The Mac
dev box can author code; only the A100 box can run nsys/ncu and reproduce
Marlin's published numbers.

## 0. Prereqs on the A100 box

- CUDA 12.1+ toolkit (provides `nsys` and `ncu`)
- A100-40GB or A100-80GB SXM (this is what the kernel targets)
- Python 3.10+ with `uv` installed
- Git access to this repo

Confirm with:
```bash
nvidia-smi
nvcc --version
nsys --version
ncu --version
```

## 1. Sync swordfish

```bash
git clone https://github.com/chokevin/swordfish && cd swordfish
uv sync --extra bench
uv run pytest -q          # baseline; should pass on CPU and CUDA
```

## 2. Install Marlin (pinned)

We pin upstream Marlin to a specific SHA so the published numbers are
reproducible. **Do not** add Marlin to `pyproject.toml` — it does not build
on macOS, and we want `uv sync` to keep working for laptop dev.

```bash
# Pinned to current main as of 2024-09-04 (README update; code state from
# 2024-04-04). If the upstream API changes, bump this here AND in
# docs/profiling/marlin-bottlenecks.md so the runbook stays honest.
MARLIN_SHA=1f25790bdd49fba53106164a24666dade68d7c90

uv pip install "git+https://github.com/IST-DASLab/marlin.git@${MARLIN_SHA}"
uv run python -c "import marlin; print('marlin OK at', '${MARLIN_SHA}')"
```

If the install fails because Marlin's `setup.py` is picky about CUDA arch,
override:
```bash
TORCH_CUDA_ARCH_LIST="8.0" uv pip install \
  "git+https://github.com/IST-DASLab/marlin.git@${MARLIN_SHA}"
```

## 3. Reproduce Marlin's headline number (`8b-b1`)

```bash
uv run python -m bench.run_bench \
  --shapes voice \
  --impls fp16,marlin \
  --repeats 5 --iters 50 \
  --out bench_results/repro/
```

**Acceptance gate:** `8b-b1` (M=1, N=4096, K=4096, g=128) Marlin TFLOPS
should land within ±10% of the published number from the Marlin paper
(roughly 4–5x speedup over FP16 baseline at this shape on A100). If not:
- Check `nvidia-smi` for clock throttling.
- Verify GPU is at default boost clocks (no power cap).
- Confirm you actually pinned the SHA above; HEAD-of-main may have drifted.
- Re-read `bench_results/repro/manifest.json` env header — driver/torch CUDA
  mismatch is the most common culprit.

## 4. Capture profiles

```bash
SHAPES=voice IMPLS=fp16,marlin bench/profile_marlin.sh
```

This drops everything under `docs/profiling/marlin/<UTC-timestamp>/`:
- `env.txt` — full env snapshot
- `results.csv` + `manifest.json` — bench output (with NVTX-tagged regions)
- `trace.json` — torch.profiler Chrome trace, **load at https://ui.perfetto.dev**
- `<shape>.nsys-rep` — nsys timeline (open with `nsys-ui` or convert to
  Perfetto via the `.sqlite` export below)
- `<shape>.sqlite` — nsys SQLite, can be ingested by Perfetto's
  trace_processor for unified Python+CUDA view
- `<shape>.ncu-rep` — Nsight Compute deep dive (open with `ncu-ui`)
- `<shape>.ncu.csv` — flat metric table consumed by `bench/roofline.py`

## 5. Roofline plot

```bash
LATEST=$(ls -td docs/profiling/marlin/*/ | head -1)
uv run python -m bench.roofline "${LATEST}" --gpu a100-80gb-sxm
# writes ${LATEST}/roofline.png
```

(Use `--gpu a100-40gb` if you're on the 40GB SKU.)

## 6. Fill in the bottleneck writeup

Edit `docs/profiling/marlin-bottlenecks.md`. For each P0 shape:
- Read its `.ncu-rep` Speed-of-Light section.
- Pick the dominant bottleneck (HBM bw / TC feed / scheduler stall / ...)
- Cite the metric + value that proves it.
- State the hypothesis for batch 4–16 underutilization that `swordfish` will attack.

Commit:
```bash
git add docs/profiling/marlin/ docs/profiling/marlin-bottlenecks.md
git commit -m "W1: Marlin profiling on A100 — $(date -u +%Y-%m-%d)"
```

That closes Week 1.

## Appendix: opening traces in Perfetto

- **torch.profiler trace** (`trace.json`): drag-and-drop into
  https://ui.perfetto.dev — works directly.
- **nsys timeline** (`<shape>.nsys-rep` or `<shape>.sqlite`): open the
  `.sqlite` with Perfetto's `trace_processor`:
  ```bash
  trace_processor --httpd <shape>.sqlite
  # then point ui.perfetto.dev at http://localhost:9001
  ```
  Or use the standalone Nsight Systems UI (`nsys-ui <shape>.nsys-rep`).
- **ncu deep dive** (`<shape>.ncu-rep`): only the Nsight Compute UI
  (`ncu-ui`) renders these correctly; no Perfetto path.
