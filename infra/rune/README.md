# swordfish + rune

This directory wires `swordfish` benchmarks into [`rune`][rune], the k8s-native AI
runtime CLI used to dispatch jobs to the `voice-agent-flex` cluster.

The shape is:

```
  rune submit <name> --profile <swordfish-bench-arch> --script scripts/swordfish-bench.sh
                                |                              |
                                |                              +-- runs inside the pod
                                +-- defines cluster wiring (queue/DRA/selectors/PVC/SYS_ADMIN)
```

`infra/airun/airun-gemm.voice-agent-flex.json` remains the source of truth for
the cluster routing values (context, namespace, queue, ResourceClaimTemplate,
node selectors, tolerations, NCU permission window). The rune profiles here
mirror that JSON so dispatching via either path lands on the same pod shape.

[rune]: https://github.com/azure-management-and-platforms/aks-ai-runtime/

## Status — read this first

| What | Status |
| --- | --- |
| `infra/rune/scripts/swordfish-bench.sh` | **Working.** In-pod entrypoint that runs `python -m swordfish.runner` with optional NCU wrapping. Independent of the profile YAML schema. |
| `Makefile` targets `rune-submit-*` | **Working.** Invocation shape verified via `make -n`. Will succeed once the profile YAMLs validate. |
| `infra/rune/profiles/*.yaml` | **DRAFT.** Built from rune binary strings + the cluster routing JSON. `rune profile list` currently rejects them silently (likely a schema mismatch). The canonical schema lives in private `applications/airun-zero/profiles/`; **diff one canonical profile against these drafts and fix the YAML before first dispatch.** Tracked as `rune-profile-schema-validation`. |
| `make rune-install-profiles` | **Working install path.** Symlinks the YAMLs into `$HOME/.config/rune/profiles/`. Will start working as soon as the YAMLs validate. |

The profiles being draft-status does not block the cluster routing knowledge —
all queue/DRA/selector/toleration/SYS_ADMIN values are pinned in the airun JSON
and the per-arch profile YAMLs. The only unknown is the rune-specific YAML
*structure* around those values.

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

## Why both rune and `swordfish.runner render-airun-gemm`?

`render-airun-gemm` writes a Kueue `Job` manifest directly. It is the lowest
common denominator and is what the dashboard's strict completion gate
documents. `rune submit` is a thinner UX on top of the same Kueue stack with
profile-based defaults, post-admission status (`rune status`), log streaming
(`rune logs`), cost reporting (`rune cost`), and cancellation (`rune cancel`).

Use `render-airun-gemm` when reviewing the exact YAML; use `rune submit` for
day-to-day dispatch.
