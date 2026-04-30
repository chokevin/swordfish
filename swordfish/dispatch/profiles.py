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
QUEUE_LOCAL = "kernel-mode-training"
PVC_NAME = "training-nfs"
PVC_MOUNT = "/data"
IMAGE = "ghcr.io/chokevin/swordfish-bench:latest"
IMAGE_PULL_POLICY = "IfNotPresent"
IMAGE_FAMILY = "swordfish-bench"

ARCHES = ("a100", "h100", "h200")


@dataclass(frozen=True)
class ProfileSpec:
    arch: str

    @property
    def name(self) -> str:
        return f"swordfish-bench-{self.arch}"


def all_profiles() -> list[ProfileSpec]:
    return [ProfileSpec(arch=a) for a in ARCHES]


_HEADER = """\
# swordfish-pack — kernel-mode benchmark profiles
#
# GENERATED FROM swordfish/dispatch/profiles.py — do not edit by hand.
# Regenerate with: make rune-profiles
#
# Each profile extends the embedded core profile `ai-train-gpu-l` (built into
# the rune binary — no symlink needed). All three profiles share image, queue,
# persistence, and DRA claim; they differ only in the arch label so dashboards
# and `rune profile list` can tell them apart.
#
# rune resolves spec.extends with deep-merge of spec.resources and spec.runtime
# (see applications/rune/internal/profile/loader.go), so we only need to declare
# the deltas. The parent ai-train-gpu-l supplies gpu/dra/requests defaults.
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
        "    airun.aks.io/lane: train\n"
        f"    swordfish.dev/arch: {profile.arch}\n"
        "spec:\n"
        f"  extends: {PARENT_PROFILE}\n"
        "  queue:\n"
        f"    clusterQueue: {QUEUE_CLUSTER}\n"
        f"    localQueue: {QUEUE_LOCAL}\n"
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
