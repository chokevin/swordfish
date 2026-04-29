# swordfish + rune

This directory wires `swordfish` benchmarks into [`rune`][rune], the k8s-native AI
runtime CLI used to dispatch jobs to the `voice-agent-flex` cluster.

The shape is:

```
  rune submit <name> --profile <swordfish-bench-arch> --script scripts/swordfish-bench.sh
                                |                              |
                                |                              +-- runs inside the pod
                                +-- defines cluster wiring (queue/DRA/selectors/PVC/SYS_ADMIN)
                                    on top of the swordfish-bench image
```

`infra/airun/airun-gemm.voice-agent-flex.json` remains the source of truth for
the cluster routing values (context, namespace, queue, ResourceClaimTemplate,
node selectors, tolerations, NCU permission window). The rune profiles here
mirror that JSON so dispatching via either path lands on the same pod shape.

[rune]: https://github.com/azure-management-and-platforms/aks-ai-runtime/

## Status — read this first

| What | Status |
| --- | --- |
| `infra/rune/image/Dockerfile` + `build.sh` | **Working source.** Builds on `nvcr.io/nvidia/pytorch:25.03-py3`, bakes liger-kernel + uv + gh + git + jq. CI (`.github/workflows/build-swordfish-image.yml`) is the canonical builder; pushes to `ghcr.io/chokevin/swordfish-bench`. Local `build.sh` defaults to podman, supports `CONTAINER_CMD=docker` override. |
| `infra/rune/scripts/swordfish-bench.sh` | **Working.** In-pod entrypoint that runs `python -m swordfish.runner` with optional NCU wrap. Has on-the-fly `pip install liger-kernel` self-heal so it works whether or not the image baked it (`SWORDFISH_SKIP_LIGER_INSTALL=1` to disable). |
| `Makefile` targets `rune-submit-*` | **Working invocation shape** (verified via `make -n`). Will succeed once the rune profile YAMLs validate. |
| `infra/rune/profiles/*.yaml` | **DRAFT.** Built from rune binary strings + the cluster routing JSON. `rune profile list` currently rejects them silently. The canonical schema lives in private `applications/airun-zero/profiles/`; **diff one canonical profile against these drafts and fix the YAML before first dispatch.** Tracked as `rune-profile-schema-validation`. |

## Image: `ghcr.io/chokevin/swordfish-bench`

Built by CI on push to `infra/rune/image/**`. Manual triggers via
`gh workflow run build-swordfish-image.yml` accept `liger_version` and
`liger_ref` build-arg overrides plus an optional extra `tag`.

| Layer | What | Why |
| --- | --- | --- |
| Base | `nvcr.io/nvidia/pytorch:25.03-py3` | torch 2.7+nv25.03, CUDA 12.8, Triton, NCU, Nsys built against each other. The Ray-flavored cluster images do not ship this stack in a known-good combination. Don't reinvent it. |
| OS | git + jq + gh CLI | gh used by Friday-publish flows that file Discussions/PRs from inside the runner. |
| Tooling | uv 0.5.7 | swordfish uses `uv run` for dependency management. |
| Kernel lib | `liger-kernel==0.5.10` (override via `LIGER_VERSION` or `LIGER_REF`) | The Week 1 first upstream touchpoint. Baked so cold-start does not pay pip install for every job. |

The image deliberately does **not** carry the swordfish source tree — by
convention the working copy lives on the `training-nfs` PVC at
`/data-nfs/swordfish/src/current` so live edits land in the next pod
without a rebuild. The in-pod script `cd`s there at startup.

### Building locally (optional)

CI is the canonical builder. Local builds are only needed to iterate on the
Dockerfile itself.

```bash
# default: podman, host arch (arm64 on Apple silicon)
infra/rune/image/build.sh

# force amd64 (slow on arm via QEMU; same arch CI uses)
PLATFORM=linux/amd64 infra/rune/image/build.sh

# push (gh auth required)
PUSH=1 infra/rune/image/build.sh
```

Docker Desktop on macOS often gives the VM only ~3.5GB RAM, which is tight
for the ~12GB nvcr base. Podman with libkrun is what the swordfish dev box
uses; pass `CONTAINER_CMD=docker` if you prefer.

## Profiles

| Profile | Arch | Queue | Notes |
| --- | --- | --- | --- |
| `swordfish-bench-base.yaml` | (shared base) | `fauna-train-queue` | image, namespace, PVC, env, common resources |
| `swordfish-bench-a100.yaml` | A100 SXM4-80GB | inherits | adds `SYS_ADMIN` cap for NCU; preflight required to pause DCGM exporter on target nodes |
| `swordfish-bench-h100.yaml` | H100 NVL | inherits | NCU works without extra caps |
| `swordfish-bench-h200.yaml` | H200 | inherits | capacity-gated; preflight before submit |

## Dispatching

From the swordfish repo root, once the profile YAMLs validate:

```bash
make rune-submit-gemm-a100      # one fp16 4096^3 GEMM job on A100
make rune-submit-gemm-h100      # same on H100 NVL
make rune-submit-gemm-h200      # same on H200 (preflight gated)
make rune-submit-gemm-matrix    # all three, named with a shared run-id

make rune-submit-liger-rmsnorm-a100   # Liger RMSNorm sweep on A100
make rune-submit-liger-swiglu-a100    # Liger SwiGLU sweep on A100
```

Or directly with rune:

```bash
rune submit swordfish-gemm-$(date +%H%M%S)-a100 \
    --profile swordfish-bench-a100 \
    --script infra/rune/scripts/swordfish-bench.sh \
    -- run-gemm --backend torch \
       --m 4096 --n 4096 --k 4096 --dtype fp16 \
       --device auto --arch-label a100 \
       --out /data-nfs/swordfish/week1/torch-gemm-a100.json
```

The script forwards everything after `--` to `python -m swordfish.runner`.

## Installing the profiles into rune's search path

Rune searches for profiles in (in order):

1. `$HOME/.config/rune/profiles`
2. `$HOME/.airun/profiles`
3. `applications/airun-zero/profiles` (canonical, in-repo)

To make these profiles discoverable on a workstation:

```bash
make rune-install-profiles      # symlinks infra/rune/profiles -> ~/.config/rune/profiles/
```

After install, `rune profile list` should show `swordfish-bench-{base,a100,h100,h200}`.
If it still says "no profiles found" the YAMLs are failing schema validation —
see the `rune-profile-schema-validation` todo.

## Plan B: dispatch via the airun render path (no rune profile required)

If the rune profile schema becomes a blocker, the existing
`swordfish.runner render-airun-gemm` command writes a Kueue `Job` YAML
directly from `infra/airun/airun-gemm.voice-agent-flex.json` and is what the
strict completion gate documents. Use that path until the rune profile
schema is sorted:

```bash
make airun-render
make airun-apply
```

Note: the airun JSON still pins `nvcr.io/nvidia/pytorch:25.03-py3` for
backward compatibility with the Week 1 GEMM matrix; the in-pod script's
on-the-fly `pip install liger-kernel` covers the Liger gap until that
config is updated to the new image.

## Why both rune and `swordfish.runner render-airun-gemm`?

`render-airun-gemm` writes a Kueue `Job` manifest directly. It is the lowest
common denominator and is what the dashboard's strict completion gate
documents. `rune submit` is a thinner UX on top of the same Kueue stack with
profile-based defaults, post-admission status (`rune status`), log streaming
(`rune logs`), cost reporting (`rune cost`), and cancellation (`rune cancel`).

Use `render-airun-gemm` when reviewing the exact YAML; use `rune submit` for
day-to-day dispatch.
