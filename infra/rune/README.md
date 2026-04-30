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

`infra/airun/airun-gemm.voice-agent-flex.json` remains the source of truth for
the **Plan B** path (`make airun-apply`) used for A100 NCU runs that need
SYS_ADMIN. Profiles here mirror the same routing values so dispatching via
either path lands on the same kind of pod.

[rune]: https://github.com/azure-management-and-platforms/aks-ai-runtime/

## Status

| What | Status |
| --- | --- |
| `infra/rune/image/Dockerfile` + `build.sh` | **Working.** Builds on `nvcr.io/nvidia/pytorch:25.03-py3`, bakes liger-kernel + uv + gh + git + jq. CI publishes to `ghcr.io/chokevin/swordfish-bench`. |
| `infra/rune/scripts/swordfish-bench.sh` | **Working.** In-pod entrypoint, on-the-fly liger install self-heal. |
| `infra/rune/profiles/swordfish-pack.yaml` | **Working.** Three profiles extending `ai-train-gpu-l`, queue=`kernel-mode-training`, image=`swordfish-bench:latest`, PVC `training-nfs` mounted at `/data` (rune storage contract). Deep-merge of `spec.resources` and `spec.runtime` from the parent now happens in rune itself, so the pack only declares deltas. |
| `make rune-install-profiles` | **Working.** Symlinks the pack into `~/.config/rune/profiles/`. The airun-core parents are now embedded in the rune binary; no symlink needed. |
| Makefile targets `rune-submit-*` | **Working invocation.** Each target sets `--output` so `rune submit get` can fetch results. |
| `rune submit --dry-run=client` GPU + DRA + PVC | **Working** — DRA `full-gpu` claim renders, container `requests`/`claims` renders, queue label correct, PVC mounted at `/data` with hot `/mnt` scratch added by rune. |
| `rune submit --profile-mode ncu\|nsys` | **Working** — output lands at `/data/<job-name>/profile/profile.{ncu-rep\|nsys-rep}`; image must have `ncu` / `nsys` on PATH (the swordfish-bench image does). |
| `rune submit --output /data/...` + `rune submit get NAME` | **Working** — annotations recorded; `rune submit get NAME --output raw` cats the file via a one-shot helper Pod. Use `--artifact NAME` for items inside a directory output. |

## A100 + Nsight Compute caveat

The canonical airun-core profiles do not expose container `SYS_ADMIN` through the
Profile spec, which Nsight Compute requires on A100 to read GPU performance
counters. For A100 NCU runs, fall back to `make airun-apply` (which renders a
raw Kueue Job and adds the cap explicitly via
`infra/airun/airun-gemm.voice-agent-flex.json`).

H100 NVL and H200 NCU work with no extra privileges and run fine through `rune`.

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
`/data/swordfish/src/current` so live edits land in the next pod
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
`spec.resources.dra.claimTemplate` is the next iteration once arch-specific
flavors land; until then, treat the arch suffix as a label hint.

## Installing the profiles into rune's search path

Rune searches for profiles in (in order):

1. `$RUNE_PROFILES_DIR` (if set)
2. `$XDG_CONFIG_HOME/rune/profiles` (if set)
3. `$HOME/.config/rune/profiles`
4. `$HOME/.airun/profiles`
5. The in-tree fallback `applications/airun-zero/profiles` (for ai2 dev work)
6. **Embedded `core/*`** (built into the rune binary; supplies `ai-train-gpu-l` etc.)

To make the swordfish pack discoverable on a workstation:

```bash
make rune-install-profiles      # symlinks infra/rune/profiles -> ~/.config/rune/profiles/
```

After install, `rune profile list` should show the three swordfish-bench
profiles plus the embedded core profiles they inherit from. To validate a
profile against the cluster:

```bash
rune profile doctor swordfish-bench-a100 --context voice-agent-flex-admin -n ray
```

## Day-to-day flows

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
`swordfish/dispatch/runs.py` (`LigerPerkernelRun`) for the typed-dataclass
shape that compiles to the equivalent `rune submit` invocation.

## Plan B: dispatch via the airun render path (no rune profile required)

If the rune profile path becomes a blocker (the A100 SYS_ADMIN case is the
remaining one), the existing `swordfish.runner render-airun-gemm` command writes
a Kueue `Job` YAML directly from `infra/airun/airun-gemm.voice-agent-flex.json`
and is what the strict completion gate documents. Plan B mounts the same
`training-nfs` PVC at `/data-nfs` for backward compatibility with previously
written results.

```bash
make airun-render
make airun-apply
```

## Why both rune and `swordfish.runner render-airun-gemm`?

`render-airun-gemm` writes a Kueue `Job` manifest directly. It is the lowest
common denominator and is what the dashboard's strict completion gate
documents. `rune submit` is a thinner UX on top of the same Kueue stack with
profile-based defaults, post-admission status (`rune status`), log streaming
(`rune logs`), cost reporting (`rune cost`), result fetch (`rune submit get`),
and cancellation (`rune cancel`).

Use `render-airun-gemm` when reviewing the exact YAML or when you need the
A100 SYS_ADMIN profile; use `rune submit` for day-to-day dispatch.
