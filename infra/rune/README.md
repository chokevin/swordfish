# swordfish + rune

This directory wires `swordfish` benchmarks into [`rune`][rune], the k8s-native AI
runtime CLI used to dispatch jobs to the `voice-agent-flex` cluster.

The shape is:

```
  rune submit <name> --profile <swordfish-bench-arch> --script scripts/swordfish-bench.sh \
                     --output <abs-path-on-PVC>
                                |                              |
                                |                              +-- runs inside the pod
                                +-- defines cluster wiring (queue/DRA/selectors/PVC)
                                    on top of the swordfish-bench image
```

[rune]: https://github.com/azure-management-and-platforms/aks-ai-runtime/

## Status

| What | Status |
| --- | --- |
| `infra/rune/image/Dockerfile` + `build-acr.sh` | **Working.** Builds on `voiceagentcr.azurecr.io/airun/autoresearch-pytorch-ray:dev` (in-region ACR, prewarmed on every GPU node). Bakes liger-kernel + swordfish source. Canonical build path is `az acr build` (publishes to `voiceagentcr.azurecr.io/airun/swordfish-bench`). The legacy `build.sh` (podman → ghcr.io) still works for local iteration. |
| `infra/rune/scripts/swordfish-bench.sh` | **Working.** In-pod entrypoint, on-the-fly liger install self-heal. |
| `infra/rune/profiles/swordfish-pack.yaml` | **Working.** Eight profiles: three normal one-GPU `swordfish-bench-*` profiles, three normal 8-GPU `swordfish-fsdp-*` profiles, plus A100-only `*-a100-ncu` variants that add `SYS_ADMIN` for Nsight Compute. Queue=`team-kernel-mode-reserved-cq`, image=`voiceagentcr.azurecr.io/airun/swordfish-bench:dev`, PVC `training-nfs` mounted at `/data` (rune storage contract). Self-contained spec (no `extends:` — embedded core profiles were removed in `airun-zero`). Generated from `swordfish/dispatch/profiles.py` — edit the Python, then `make rune-profiles`. A sync test in `tests/test_dispatch.py` enforces the invariant. |
| `make rune-install-profiles` | **Working.** Verifies the YAML is in sync with the Python source, then symlinks the pack into `~/.config/rune/profiles/`. The core parents are embedded in the rune binary; no extra symlink needed. |
| Makefile targets `rune-submit-*` | **Working.** Targets shell out to `python -m swordfish.runner submit-bench` for the GEMM matrix, A100 Liger per-kernel rows, and the A100 FSDP baseline/Liger catch-up pair; the dispatch SDK builds and invokes the right `rune submit` argv. Each sets `--output` so `rune submit get` can fetch results. |
| `rune submit --dry-run=client` GPU + DRA + PVC | **Working** — one-GPU profiles render DRA `full-gpu`; FSDP profiles render `ds-8gpus`; container `requests`/`claims` render, queue label correct, PVC mounted at `/data` with hot `/mnt` scratch added by rune. |
| `rune submit --profile-mode ncu` | **Working** — output lands at `/data/<job-name>/profile/profile.ncu-rep`; image must have `ncu` on PATH (the swordfish-bench image does — Nsight Compute 2025.1.0.0 from the autoresearch-pytorch-ray base). A100 submits use `swordfish-bench-a100-ncu` / `swordfish-fsdp-a100-ncu` so Rune renders `securityContext.capabilities.add: [SYS_ADMIN]`; A100 still needs the temporary DCGM exporter pause below. |
| `rune submit --profile-mode nsys` | **Working** — output lands at `/data/<job-name>/profile/profile.nsys-rep`; image must have `nsys` on PATH (the swordfish-bench image installs Nsight Systems 2024.6.2 via `cuda-nsight-systems-12-8` apt package). |
| `--profile-mode torch` (swordfish-side, in-process) | **Working** — output is a Chrome trace JSON at `/data/<job-name>/profile/profile.json` (open in [Perfetto](https://ui.perfetto.dev/) or chrome://tracing). Bypasses rune's external-wrapper path: the dispatch SDK injects `SWORDFISH_PROFILE=torch` + `SWORDFISH_PROFILE_OUT=...` env vars and `swordfish.runner.profile_torch.torch_profiler_context` wraps the bench main in `torch.profiler`. No SYS_ADMIN needed (unlike NCU on A100), and works on any image that has `torch` (i.e. all of them) — useful when nsys/ncu image dependencies or cluster mirrors are broken. |
| `rune submit --output /data/...` + `rune submit get NAME` | **Working** — annotations recorded; `rune submit get NAME --output raw` cats the file via a one-shot helper Pod. Use `--artifact NAME` for items inside a directory output. |
| `submit-bench --workload liger-fsdp` | **Dry-runable / dispatchable.** Uses generated 8-GPU `swordfish-fsdp-*` profiles, injects `--gpu-class`, and the bench script launches `torchrun --standalone --nproc-per-node 8` before running `liger-fsdp-step`. |

## A100 + Nsight Compute procedure

A100 NCU requires both container `SYS_ADMIN` and exclusive ownership of the
driver profiling resource. Rune now renders `spec.runtime.securityContext` from
profiles, and Swordfish's A100 NCU profiles request:

```yaml
runtime:
  securityContext:
    capabilities:
      add:
        - SYS_ADMIN
```

Before running A100 NCU, temporarily patch `nvidia-dcgm-exporter` so it does not
schedule on `gpu=a100` nodes, run the short profiling job, then restore the
DaemonSet immediately. The full operational checklist and evidence live in
[`docs/airun/a100-ncu-blocker.md`](../../docs/airun/a100-ncu-blocker.md).

H100 NVL and H200 NCU work with no extra privileges and run fine through `rune`.

## Image: `voiceagentcr.azurecr.io/airun/swordfish-bench`

Built locally via `infra/rune/image/build-acr.sh` (uses Azure ACR Tasks —
remote build on ACR build agents, ~6min vs ~25min on a GH ubuntu runner).
The legacy `infra/rune/image/build.sh` (podman → ghcr.io) still works for
Dockerfile iteration if you don't have ACR access.

```bash
az acr login -n voiceagentcr        # one-time per laptop session
infra/rune/image/build-acr.sh        # publishes :<sha> + :dev
```

Manual triggers via `gh workflow run build-swordfish-image.yml` accept
`liger_version` and `liger_ref` build-arg overrides plus an optional extra
`tag`. (CI publishes to ghcr.io; the in-cluster path uses the ACR tag.)

| Layer | What | Why |
| --- | --- | --- |
| Base | `voiceagentcr.azurecr.io/airun/autoresearch-pytorch-ray:dev` | In-region ACR, prewarmed on every GPU node by the cluster-wide `baked-image-prewarm-v2` daemonset (see `ai2:applications/airun-zero/deploy/umbrella/airun-core/values.yaml`). Ships torch 2.7+cu128, Triton 3.3.0, NCU 2025.1.0.0, Ray 2.40, transformers 5.5.4, peft, trl, accelerate, datasets, bitsandbytes. Switching to it gets us ~10x faster cold-start (in-region vs ghcr.io) and ~250x layer-dedup win on prewarmed nodes (kubelet skips the ~10GB base-layer download). |
| Profiler | `cuda-nsight-systems-12-8` (nsys 2024.6.2) | The base ships ncu but not nsys. Pulled from the NVIDIA CUDA apt repo the base already configures. Pinned to 12-8 to track the base's CUDA 12.8 toolchain. |
| Kernel lib | `liger-kernel==0.5.10` (override via `LIGER_VERSION` or `LIGER_REF`) | The Week 1 first upstream touchpoint. Baked so cold-start does not pay pip install for every job. Verified compatible with the base's transformers 5.5.4. |
| Source | swordfish package, `pip install -e . --no-deps` | Researchers iterate on Python files in experiments/ without rebuilding this layer; only when the swordfish/ package itself changes does this layer rebuild. |

The image deliberately does **not** carry the swordfish source tree — by
convention the working copy lives on the `training-nfs` PVC at
`/data/swordfish/src/current` so live edits land in the next pod
without a rebuild. The in-pod script `cd`s there at startup.

### Building locally (optional)

`build-acr.sh` is the canonical builder (Azure ACR Tasks: remote, ~6min).
Local builds via `build.sh` are only needed to iterate on the Dockerfile
itself — and they push to ghcr.io rather than the ACR location the cluster
profile expects, so they're for iteration only.

```bash
# CANONICAL — remote build via ACR Tasks, publishes to voiceagentcr ACR.
az acr login -n voiceagentcr
infra/rune/image/build-acr.sh

# LOCAL ITERATION — podman → ghcr.io (does not match the in-cluster image tag).
infra/rune/image/build.sh

# force amd64 local build (slow on arm via QEMU)
PLATFORM=linux/amd64 infra/rune/image/build.sh

# push the local build to ghcr (gh auth required)
PUSH=1 infra/rune/image/build.sh
```

Docker Desktop on macOS often gives the VM only ~3.5GB RAM, which is tight
for the ~12GB base. Podman with libkrun is what the swordfish dev box
uses; pass `CONTAINER_CMD=docker` if you prefer.

## Profiles

`infra/rune/profiles/swordfish-pack.yaml` defines one-GPU benchmark profiles,
8-GPU FSDP profiles, and A100-only NCU variants. The pack is
**generated from `swordfish/dispatch/profiles.py`** — edit the Python
constants and run `make rune-profiles` to regenerate; CI fails the build
if the two drift via `tests/test_dispatch.py::test_swordfish_pack_yaml_in_sync_with_python_source`.

| Profile | Arch (intended) | Queue | Notes |
| --- | --- | --- | --- |
| `swordfish-bench-a100` | A100 SXM4-80GB | `kernel-mode-training` | One-GPU microbenchmark profile for normal runs |
| `swordfish-bench-a100-ncu` | A100 SXM4-80GB | `kernel-mode-training` | One-GPU NCU profile; adds `SYS_ADMIN` and still requires the short DCGM pause |
| `swordfish-bench-h100` | H100 NVL | `kernel-mode-training` | One-GPU microbenchmark profile. NCU works without extra caps |
| `swordfish-bench-h200` | H200 | `kernel-mode-large-memory` | One-GPU microbenchmark profile. NCU works without extra caps |
| `swordfish-fsdp-a100` | 8x A100 SXM4-80GB | `kernel-mode-training` | Thursday Liger FSDP parity profile; gpu.size=xl, claimTemplate=ds-8gpus |
| `swordfish-fsdp-a100-ncu` | 8x A100 SXM4-80GB | `kernel-mode-training` | A100 FSDP NCU variant; adds `SYS_ADMIN` and uses claimTemplate=ds-8gpus |
| `swordfish-fsdp-h100` | 8x H100 NVL | `kernel-mode-training` | Capacity-permitting extension profile; gpu.size=xl, claimTemplate=ds-8gpus |
| `swordfish-fsdp-h200` | 8x H200 | `kernel-mode-large-memory` | Capacity-permitting extension profile; gpu.size=xl, claimTemplate=ds-8gpus |

All profiles pin `nvidia.com/gpu.product` so A100/H100/H200 runs cannot land on
the wrong node class. The FSDP profiles differ from the one-GPU profiles by
requesting gpu.size=xl, the `ds-8gpus` DRA claim template, and higher
CPU/memory, which maps to a full 8-GPU node.

## Installing the profiles into rune's search path

Rune searches for profiles in (in order):

1. `$RUNE_PROFILES_DIR` (if set)
2. `$XDG_CONFIG_HOME/rune/profiles` (if set)
3. `$HOME/.config/rune/profiles`
4. `$HOME/.airun/profiles` (legacy XDG path; rune still reads it)
5. The in-tree fallback `applications/airun-zero/profiles` (for ai2 dev work)
6. **Embedded `core/*`** (built into the rune binary; supplies `ai-train-gpu-l` etc.)

To make the swordfish pack discoverable on a workstation:

```bash
make rune-bootstrap           # installs rune-py + matching rune CLI
make rune-install-profiles      # symlinks infra/rune/profiles -> ~/.config/rune/profiles/
```

`make rune-bootstrap` uses the private `aks-ai-runtime` `rune-cli-v0.2.0`
release by default. It configures `gh` git auth, installs `rune-py` into this
repo's uv environment, downloads the matching CLI with `rune-py bootstrap`, and
runs `rune-py doctor`.

After profile install, `rune profile list` should show the four swordfish-bench
profiles and the four swordfish-fsdp profiles. To validate a profile against
the cluster:

```bash
rune profile doctor swordfish-bench-a100 --context voice-agent-flex-admin -n ray
```

## Day-to-day flows

The day-to-day entrypoint is the experiment-grounded Python dispatch shim. It
builds the right `rune submit` argv from repo-registered experiment intent, so
callers choose an experiment and an arch instead of remembering
`--profile`/`--script`/`--output`/`--volume` details.

```bash
# discover and explain repo-approved experiment intents
uv run python -m swordfish.runner list-experiments
uv run python -m swordfish.runner explain-experiment liger-fsdp --arch a100

# preferred: dispatch by experiment intent (python wraps rune)
uv run python -m swordfish.runner submit-experiment gemm --arch h100

# the same workload from a Python script:
python -c "from swordfish.dispatch import build_run_for_experiment; print(build_run_for_experiment('gemm', 'h100').submit().name)"

# preview the rendered Job manifest without submitting
uv run python -m swordfish.runner submit-experiment gemm --arch a100 \
  --dry-run client --print-yaml
```

The local registry lives in `swordfish.dispatch.experiments` until upstream Rune
grows project-scoped experiment aliases (tracked in
https://github.com/azure-management-and-platforms/aks-ai-runtime/issues/285).
It maps experiment IDs such as `gemm` and `liger-fsdp` to generated profile
families (`swordfish-bench-*`, `swordfish-fsdp-*`) while queue placement stays
owned by the generated profile YAML.

If you need a one-off submission with arbitrary args, the underlying
`rune submit` is still available:

```bash
# preview the rendered Job manifest (no cluster contact)
rune submit my-bench --profile swordfish-bench-a100 \
  --script infra/rune/scripts/swordfish-bench.sh \
  --output /data/swordfish/week1/my-bench.json \
  --dry-run=client

# real submission
rune submit my-bench --profile swordfish-bench-a100 \
  --script infra/rune/scripts/swordfish-bench.sh \
  --output /data/swordfish/week1/my-bench.json \
  --env SWORDFISH_ARCH_LABEL=a100 \
  -n ray

# fetch the result JSON written to /data on the PVC
rune submit get my-bench -n ray --output raw > my-bench.json

# wrap the entrypoint in NCU; artifact at /data/my-bench/profile/profile.ncu-rep
rune submit my-bench-ncu --profile swordfish-bench-h100 \
  --script infra/rune/scripts/swordfish-bench.sh \
  --output /data/swordfish/week1/my-bench-ncu.json \
  --profile-mode ncu \
  -n ray

# fetch the .ncu-rep file (the recorded result-path is the JSON; pass --path
# explicitly because the trace lives at rune's hardcoded /data/<name>/profile/)
rune submit get my-bench-ncu -n ray \
  --path /data/my-bench-ncu/profile --pvc training-nfs \
  --artifact profile.ncu-rep --output raw > my-bench-ncu.ncu-rep
```

The Python SDK in `swordfish/dispatch/` wraps all of this: see
`swordfish/dispatch/runs.py` (`LigerPerkernelRun`, `TorchGemmRun`) for the
typed-dataclass shape that compiles to the equivalent `rune submit`
invocation, and `swordfish/dispatch/profiles.py` for the profile pack.
