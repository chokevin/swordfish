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
| `infra/rune/profiles/swordfish-pack.yaml` | **Working.** Three profiles (`swordfish-bench-{a100,h100,h200}`) extending `ai-train-gpu-l`, routing to `kernel-mode-training` LocalQueue, image=`ghcr.io/chokevin/swordfish-bench:latest`, persistence on `training-nfs` PVC. `rune profile list` finds them, `rune profile show` resolves the extends chain cleanly, `rune submit --dry-run=client` produces a valid Kueue Job. |
| `make rune-install-profiles` | **Working.** Symlinks `infra/rune/profiles/` into `~/.config/rune/profiles/`, plus copies the airun core profiles (`ai-train-gpu-l`, etc.) so `extends:` resolves locally. |
| `Makefile` targets `rune-submit-*` | **Working invocation shape** (verified via `make -n`). |
| GPU + PVC translation in V0 rune | **Known V0 gap.** The current rune binary renders `apiVersion: airun.aks.io/v1alpha1` Profiles but does **not** translate `spec.resources.persistence` to volumeMounts or `spec.resources.gpu`/`spec.resources.dra` to a DRA ResourceClaim. The job lands on Kueue with the right queue, image, and scheduling but currently no GPU. **Plan B (`make airun-render` + `make airun-apply`) is the path that produces a complete cluster-ready manifest today.** Tracked as `rune-v0-gpu-pvc-translation`. |

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

`infra/rune/profiles/swordfish-pack.yaml` defines three profiles, all of
which extend the airun-core `ai-train-gpu-l`:

| Profile | Arch (intended) | Queue | Notes |
| --- | --- | --- | --- |
| `swordfish-bench-a100` | A100 SXM4-80GB | `kernel-mode-training` | NCU on A100 needs SYS_ADMIN — fall back to `make airun-apply` for that case |
| `swordfish-bench-h100` | H100 NVL | `kernel-mode-training` | NCU works without extra caps |
| `swordfish-bench-h200` | H200 | `kernel-mode-training` | NCU works without extra caps |

Today the three profiles are functionally identical at the rune resolver
level — Kueue routes to whichever ResourceFlavor in
`team-kernel-mode-reserved-cq` has capacity. Per-arch routing via
`spec.resources.dra.claimTemplate` is the next iteration once the V1
rune webhook lands; until then, treat the arch suffix as a label hint.

## Installing the profiles into rune's search path

Rune searches for profiles in (in order):

1. `$HOME/.config/rune/profiles`
2. `$HOME/.airun/profiles`
3. `applications/airun-zero/profiles` (canonical, in-repo)

To make these profiles discoverable on a workstation:

```bash
make rune-install-profiles      # symlinks infra/rune/profiles -> ~/.config/rune/profiles/
                                # also copies the airun-core profiles so extends: resolves
```

After install, `rune profile list` should show three swordfish-bench
profiles plus the eight airun-core profiles they inherit from. If it
still says "no profiles found" the YAMLs are failing schema validation —
re-check `apiVersion: airun.aks.io/v1alpha1` and `kind: Profile` at the
top of each doc.

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
