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
| `infra/rune/image/Dockerfile` + `build.sh` | **Working.** Builds on `nvcr.io/nvidia/pytorch:25.03-py3`, bakes liger-kernel + uv + gh + git + jq. CI publishes to `ghcr.io/chokevin/swordfish-bench`. |
| `infra/rune/scripts/swordfish-bench.sh` | **Working.** In-pod entrypoint, on-the-fly liger install self-heal. |
| `infra/rune/profiles/swordfish-pack.yaml` | **Working.** Three profiles extending `ai-train-gpu-l`, queue=`kernel-mode-training`, image=`swordfish-bench:latest`. `rune profile show` resolves cleanly with parent's GPU/DRA/scheduling preserved (after re-declaring the parent's `resources.gpu/dra/requests` in each child to work around V0 rune's lack of deep-merge — see "PVC and merge gotchas" below). |
| `make rune-install-profiles` | **Working.** Symlinks profiles + airun-core into `~/.config/rune/profiles/`. |
| `Makefile` targets `rune-submit-*` | **Working invocation.** |
| `rune submit --dry-run=client` GPU + DRA | **Working** for our profiles (DRA `full-gpu` claim renders, container `requests`/`claims` renders, queue label correct). |
| `rune submit` PVC mount (`spec.resources.persistence`) | **NOT YET — V0 limitation.** The persistence array is a schema-valid hint but V0 rune `submit` does not translate it to volumeMounts/volumes. The `--volume` flag is in the binary but not yet exposed on `submit` (only `--pvc` on `shell`). For full PVC-mounted runs today, use Plan B (`make airun-render` + `make airun-apply`) which renders volumes explicitly from `infra/airun/airun-gemm.voice-agent-flex.json`. |

## PVC and merge gotchas (V0 rune)

Two V0 quirks that bit our first profile drafts; documenting so the next
time we touch the profile we don't repeat them:

### `spec.resources` is replaced wholesale, not deep-merged

If a child profile sets *any* field under `spec.resources` (e.g.
`persistence`), V0 rune's resolver replaces the entire `resources`
block, dropping the parent's `gpu`/`dra`/`requests` blocks. Workaround:
re-declare the parent's whole `resources` body in the child, then append
the override.

Our pack does this by repeating `gpu.size: l`, `gpu.memoryGiBMin: 60`,
`dra.{deviceClass, claimTemplate: full-gpu}`, and
`requests.{cpu: "16", memory: 64Gi}` from `ai-train-gpu-l` before
adding `persistence`. The redundancy is intentional — it survives V0's
shallow merge. V1 webhook resolution is documented to deep-merge, at
which point the duplicates can be removed.

### `spec.resources.persistence` is not honored by `rune submit` on V0

V0 `rune submit` recognizes `spec.resources.gpu`, `spec.resources.dra`,
and `spec.resources.requests` but **silently ignores
`spec.resources.persistence`**. The rendered Job has no volumeMounts and
no PVC volume. The binary contains a `--volume name=pvc:claimName` flag
internally, but it is not exposed on `submit` (only `--pvc` on `shell`,
which mounts the named PVC at `/workspace`).

Hardcoded default in the rune binary for finetune workflows: `blob-training`
PVC, expected to be pre-provisioned in the target namespace.

For swordfish today, this means:
- `rune submit` produces a Job that admits to Kueue but cannot read or
  write `/data-nfs` at runtime. The benchmarks that need the PVC will
  fail.
- Plan B (`make airun-render` + `make airun-apply`) renders explicit
  volumes/volumeMounts from `infra/airun/airun-gemm.voice-agent-flex.json`
  and is the dispatch path to use until V1 rune lands.
- `/data-nfs` mount expectations in `infra/rune/scripts/swordfish-bench.sh`
  are honest but currently only met under Plan B.

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
