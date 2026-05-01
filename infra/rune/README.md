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
| `infra/rune/profiles/swordfish-pack.yaml` | **Working.** Three profiles, queue=`team-kernel-mode-reserved-cq`, image=`voiceagentcr.azurecr.io/airun/swordfish-bench:dev`, PVC `training-nfs` mounted at `/data` (rune storage contract). Self-contained spec (no `extends:` — embedded core profiles were removed in `airun-zero`). Generated from `swordfish/dispatch/profiles.py` — edit the Python, then `make rune-profiles`. A sync test in `tests/test_dispatch.py` enforces the invariant. |
| `make rune-install-profiles` | **Working.** Verifies the YAML is in sync with the Python source, then symlinks the pack into `~/.config/rune/profiles/`. The core parents are embedded in the rune binary; no extra symlink needed. |
| Makefile targets `rune-submit-*` | **Working.** Each target shells out to `python -m swordfish.runner submit-bench --workload {gemm,liger-rmsnorm,liger-swiglu} --arch {a100,h100,h200}`; the dispatch SDK builds and invokes the right `rune submit` argv. Each sets `--output` so `rune submit get` can fetch results. |
| `rune submit --dry-run=client` GPU + DRA + PVC | **Working** — DRA `full-gpu` claim renders, container `requests`/`claims` renders, queue label correct, PVC mounted at `/data` with hot `/mnt` scratch added by rune. |
| `rune submit --profile-mode ncu` | **Working** — output lands at `/data/<job-name>/profile/profile.ncu-rep`; image must have `ncu` on PATH (the swordfish-bench image does — Nsight Compute 2025.1.0.0 is in the autoresearch-pytorch-ray base). |
| `rune submit --profile-mode nsys` | **Broken in current image** — the `autoresearch-pytorch-ray` base does not ship Nsight Systems. `SWORDFISH_PROFILE=nsys` errors with a clear "nsys not on PATH" message. Re-add nsys to the base (or to our derived layer via apt) if it becomes load-bearing. |
| `rune submit --output /data/...` + `rune submit get NAME` | **Working** — annotations recorded; `rune submit get NAME --output raw` cats the file via a one-shot helper Pod. Use `--artifact NAME` for items inside a directory output. |

## A100 + Nsight Compute caveat

Rune profiles cannot currently expose container `SYS_ADMIN` through the Profile
spec, which Nsight Compute requires on A100 to read GPU performance counters.
A100 NCU is a **known limitation** tracked in
[`docs/airun/a100-ncu-blocker.md`](../../docs/airun/a100-ncu-blocker.md); the fix
lives in rune (let profiles request `securityContext.capabilities.add: SYS_ADMIN`
under a kueue-gated allowlist), not swordfish.

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

`infra/rune/profiles/swordfish-pack.yaml` defines three profiles, all of
which extend the embedded core profile `ai-train-gpu-l`. The pack is
**generated from `swordfish/dispatch/profiles.py`** — edit the Python
constants and run `make rune-profiles` to regenerate; CI fails the build
if the two drift via `tests/test_dispatch.py::test_swordfish_pack_yaml_in_sync_with_python_source`.

| Profile | Arch (intended) | Queue | Notes |
| --- | --- | --- | --- |
| `swordfish-bench-a100` | A100 SXM4-80GB | `kernel-mode-training` | NCU on A100 needs SYS_ADMIN — currently a known rune limitation, see caveat above |
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
4. `$HOME/.airun/profiles` (legacy XDG path; rune still reads it)
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

The day-to-day entrypoint is the Python dispatch SDK; it builds the right
`rune submit` argv from typed dataclasses so callers don't have to remember
the `--profile`/`--script`/`--output`/`--volume` shape.

```bash
# preferred: dispatch via the swordfish runner CLI (python wraps rune)
uv run python -m swordfish.runner submit-bench \
  --workload gemm --arch h100 --m 4096 --n 4096 --k 4096 --dtype fp16

# the same workload from a Python script:
python -c "from swordfish.dispatch import TorchGemmRun; print(TorchGemmRun(arch='h100').submit().name)"

# preview the rendered Job manifest without submitting
uv run python -m swordfish.runner submit-bench --workload gemm --arch a100 \
  --dry-run client --print-yaml
```

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
