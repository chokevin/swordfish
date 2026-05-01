"""Source-of-truth Python definitions for the swordfish rune profile pack.

The on-disk file `infra/rune/profiles/swordfish-pack.yaml` is what rune
actually reads from `~/.config/rune/profiles/`, but it is generated from
this module. The sync is enforced by `tests/test_profiles_yaml_in_sync`.

To regenerate after editing the constants below, run:

    make rune-profiles

which calls `python -m swordfish.runner generate-rune-profiles`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

PACK_YAML_PATH = "infra/rune/profiles/swordfish-pack.yaml"

# Shared across all three profiles. Edit here, regenerate, single diff.
PARENT_PROFILE = "ai-train-gpu-l"
QUEUE_CLUSTER = "team-kernel-mode-reserved-cq"
PVC_NAME = "training-nfs"
PVC_MOUNT = "/data"
IMAGE = "ghcr.io/chokevin/swordfish-bench:latest"
IMAGE_PULL_POLICY = "IfNotPresent"
IMAGE_FAMILY = "swordfish-bench"

# Per-arch lane + local queue overrides. The rune topology validator (added in
# rune `chokevin/airun-zero-into-repo` after 2f566de) refuses
# gpuClass=h200-nvlink-141gb when lane != large-memory, so H200 routes through
# the kernel-mode-large-memory queue instead of kernel-mode-training. A100 and
# H100 stay on the training lane/queue.
LANE_TRAINING = "training"
LANE_LARGE_MEMORY = "large-memory"
QUEUE_TRAINING = "kernel-mode-training"
QUEUE_LARGE_MEMORY = "kernel-mode-large-memory"

ARCH_TO_LANE = {
    "a100": LANE_TRAINING,
    "h100": LANE_TRAINING,
    "h200": LANE_LARGE_MEMORY,
}
ARCH_TO_LOCAL_QUEUE = {
    "a100": QUEUE_TRAINING,
    "h100": QUEUE_TRAINING,
    "h200": QUEUE_LARGE_MEMORY,
}

ARCHES = ("a100", "h100", "h200")


@dataclass(frozen=True)
class ProfileSpec:
    arch: str

    @property
    def name(self) -> str:
        return f"swordfish-bench-{self.arch}"

    @property
    def lane(self) -> str:
        return ARCH_TO_LANE[self.arch]

    @property
    def local_queue(self) -> str:
        return ARCH_TO_LOCAL_QUEUE[self.arch]


def all_profiles() -> list[ProfileSpec]:
    return [ProfileSpec(arch=a) for a in ARCHES]


_HEADER = """\
# swordfish-pack — kernel-mode benchmark profiles
#
# GENERATED FROM swordfish/dispatch/profiles.py — do not edit by hand.
# Regenerate with: make rune-profiles
#
# Each profile extends the embedded core profile `ai-train-gpu-l` (built into
# the rune binary — no symlink needed). The pack overrides spec.lane and the
# Kueue local queue per arch so the rune topology validator routes H200 to
# the kernel-mode-large-memory lane (required for gpuClass=h200-nvlink-141gb)
# while A100 and H100 continue to use the kernel-mode-training lane.
#
# rune resolves spec.extends with deep-merge of spec.resources and spec.runtime
# (see applications/rune/internal/profile/loader.go), so we only declare the
# deltas. The parent ai-train-gpu-l supplies gpu/dra/requests defaults.
#
# A100 + Nsight Compute caveat: rune profiles cannot currently expose container
# SYS_ADMIN, which Nsight Compute requires on A100 to read GPU performance
# counters. A100 NCU is a known limitation tracked in docs/airun/a100-ncu-blocker.md;
# H100 NVL and H200 NCU work with no extra privileges and run fine through rune.
"""


def _render_one(profile: ProfileSpec) -> str:
    return (
        "---\n"
        "apiVersion: airun.aks.io/v1alpha1\n"
        "kind: Profile\n"
        "metadata:\n"
        f"  name: {profile.name}\n"
        "  labels:\n"
        "    airun.aks.io/catalog: pack\n"
        "    airun.aks.io/pack: swordfish\n"
        f"    airun.aks.io/lane: {profile.lane}\n"
        f"    swordfish.dev/arch: {profile.arch}\n"
        "spec:\n"
        f"  extends: {PARENT_PROFILE}\n"
        f"  lane: {profile.lane}\n"
        "  queue:\n"
        f"    clusterQueue: {QUEUE_CLUSTER}\n"
        f"    localQueue: {profile.local_queue}\n"
        "  resources:\n"
        "    persistence:\n"
        f"      - pvcName: {PVC_NAME}\n"
        f"        mountPath: {PVC_MOUNT}\n"
        "  runtime:\n"
        f"    image: {IMAGE}\n"
        f"    imagePullPolicy: {IMAGE_PULL_POLICY}\n"
        f"    imageFamily: {IMAGE_FAMILY}\n"
    )


def render_pack_yaml() -> str:
    """Render the full multi-document YAML for the pack."""
    out = io.StringIO()
    out.write(_HEADER)
    for profile in all_profiles():
        out.write("\n")
        out.write(_render_one(profile))
    return out.getvalue()
