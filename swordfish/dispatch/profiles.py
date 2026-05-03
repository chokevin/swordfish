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
QUEUE_CLUSTER = "team-kernel-mode-reserved-cq"
PVC_NAME = "training-nfs"
PVC_MOUNT = "/data"
IMAGE = "voiceagentcr.azurecr.io/airun/swordfish-bench:dev"
IMAGE_PULL_POLICY = "IfNotPresent"
IMAGE_FAMILY = "swordfish-bench"

# Node placement. The actual node label on every GPU node in the
# voice-agent-flex cluster is `airun.aks.io/lane=train` (manually applied; not
# yet in the NodePool template). Keep this distinct from `spec.lane` (which is
# the rune topology validator's vocabulary: training | large-memory | eval |
# elastic) — they are two different concepts that historically shared a string.
NODE_LANE_LABEL = "airun.aks.io/lane"
NODE_LANE_VALUE = "train"
PRIORITY_CLASS = "airun-train-default"

# Pod resource requests + DRA claim. `full-gpu` is the right claim template for
# one-GPU kernel benches. The FSDP profile uses the cluster's single-node
# 8-GPU DRA claim template so torchrun's eight local ranks each see a device.
DRA_DEVICE_CLASS = "gpu.nvidia.com"
BENCH_DRA_CLAIM_TEMPLATE = "full-gpu"
FSDP_DRA_CLAIM_TEMPLATE = "ds-8gpus"
GPU_MEMORY_GIB_MIN = 60
BENCH_GPU_SIZE = "l"
BENCH_GPUS_PER_NODE = 1
BENCH_CPU_REQUEST = "16"
BENCH_MEMORY_REQUEST = "64Gi"
FSDP_GPU_SIZE = "xl"
FSDP_GPUS_PER_NODE = 8
FSDP_CPU_REQUEST = "64"
FSDP_MEMORY_REQUEST = "512Gi"

# Policy + cost. Cost is informational only (not enforced by Kueue admission)
# but the runtime preempt knobs do matter for the scheduler.
PREEMPTABLE = True
MAX_QUEUE_WAIT_SECONDS = 3600
CHECKPOINT_ON_PREEMPT = True

ARCH_TO_GPU_TYPE = {
    "a100": "a100",
    "h100": "h100",
    "h200": "h200",
}

# Physical GPU SKU per arch — used as an extra nodeSelector predicate so the
# Kueue scheduler can't place an a100-targeted job on an H200 node (or vice
# versa). Without this, every arch profile shares the same lane=train selector
# and Kueue picks whichever node has a free DRA claim, which is wrong.
ARCH_TO_GPU_PRODUCT = {
    "a100": "NVIDIA-A100-SXM4-80GB",
    "h100": "NVIDIA-H100-NVL",
    "h200": "NVIDIA-H200",
}

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
    workload: str = "bench"
    sys_admin: bool = False

    @property
    def name(self) -> str:
        base = f"swordfish-{self.workload}-{self.arch}"
        return f"{base}-ncu" if self.sys_admin else base

    @property
    def lane(self) -> str:
        return ARCH_TO_LANE[self.arch]

    @property
    def local_queue(self) -> str:
        return ARCH_TO_LOCAL_QUEUE[self.arch]

    @property
    def gpu_size(self) -> str:
        return FSDP_GPU_SIZE if self.workload == "fsdp" else BENCH_GPU_SIZE

    @property
    def gpus_per_node(self) -> int:
        return FSDP_GPUS_PER_NODE if self.workload == "fsdp" else BENCH_GPUS_PER_NODE

    @property
    def cpu_request(self) -> str:
        return FSDP_CPU_REQUEST if self.workload == "fsdp" else BENCH_CPU_REQUEST

    @property
    def memory_request(self) -> str:
        return FSDP_MEMORY_REQUEST if self.workload == "fsdp" else BENCH_MEMORY_REQUEST

    @property
    def claim_template(self) -> str:
        return FSDP_DRA_CLAIM_TEMPLATE if self.workload == "fsdp" else BENCH_DRA_CLAIM_TEMPLATE


def all_profiles() -> list[ProfileSpec]:
    return [
        *(ProfileSpec(arch=a, workload="bench") for a in ARCHES),
        *(ProfileSpec(arch=a, workload="fsdp") for a in ARCHES),
        ProfileSpec(arch="a100", workload="bench", sys_admin=True),
        ProfileSpec(arch="a100", workload="fsdp", sys_admin=True),
    ]


_HEADER = """\
# swordfish-pack — kernel-mode benchmark profiles
#
# GENERATED FROM swordfish/dispatch/profiles.py — do not edit by hand.
# Regenerate with: make rune-profiles
#
# Self-contained: every profile inlines the full spec (no `extends:`). The
# rune binary's embedded core profiles were removed in the
# chokevin/airun-zero-into-repo branch, so any pack that depended on
# `ai-train-gpu-l` as a parent now fails to load. Inlining keeps swordfish
# resilient to ai2 churn at the cost of having to mirror future parent
# changes by hand.
#
# Per-arch deltas:
#   * spec.lane          — rune topology validator vocabulary
#                          (training for A100/H100, large-memory for H200,
#                          required by the gpuClass=h200-nvlink-141gb check
#                          in applications/rune/internal/topology/topology.go).
#   * queue.localQueue   — kernel-mode-training vs kernel-mode-large-memory.
#   * cost.gpuType       — informational.
#
# Shared across all profiles:
#   * scheduling.nodeSelector keys on `airun.aks.io/lane=train` (the actual
#     label on every GPU node in voice-agent-flex today; not the validator's
#     `lane` value) AND on `nvidia.com/gpu.product` so a100-targeted jobs
#     can't land on H100/H200 nodes (and vice versa).
#   * scheduling.tolerations match the lane=train taint.
#   * resources.dra uses the `full-gpu` claim template (one device per pod;
#     correct for kernel benches). `swordfish-bench-*` profiles use gpu.size=l
#     (one GPU); `swordfish-fsdp-*` profiles use gpu.size=xl (one 8-GPU node).
#   * resources.persistence mounts the training-nfs PVC at /data.
#
# A100 + Nsight Compute caveat: A100 NCU requires container SYS_ADMIN to read
# GPU performance counters. The normal profiles do not request elevated
# capability; the dedicated `*-a100-ncu` profiles add it only for profiler runs.
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
        f"    swordfish.dev/workload: {profile.workload}\n"
        "spec:\n"
        f"  lane: {profile.lane}\n"
        "  queue:\n"
        f"    clusterQueue: {QUEUE_CLUSTER}\n"
        f"    localQueue: {profile.local_queue}\n"
        "  scheduling:\n"
        "    nodeSelector:\n"
        f"      {NODE_LANE_LABEL}: {NODE_LANE_VALUE}\n"
        f"      nvidia.com/gpu.product: {ARCH_TO_GPU_PRODUCT[profile.arch]}\n"
        "    tolerations:\n"
        f"      - key: {NODE_LANE_LABEL}\n"
        "        operator: Equal\n"
        f"        value: {NODE_LANE_VALUE}\n"
        "        effect: NoSchedule\n"
        f"    priorityClassName: {PRIORITY_CLASS}\n"
        "  resources:\n"
        "    gpu:\n"
        f"      size: {profile.gpu_size}\n"
        f"      memoryGiBMin: {GPU_MEMORY_GIB_MIN}\n"
        "    dra:\n"
        f"      deviceClass: {DRA_DEVICE_CLASS}\n"
        f"      claimTemplate: {profile.claim_template}\n"
        "    requests:\n"
        f'      cpu: "{profile.cpu_request}"\n'
        f"      memory: {profile.memory_request}\n"
        "    persistence:\n"
        f"      - pvcName: {PVC_NAME}\n"
        f"        mountPath: {PVC_MOUNT}\n"
        "  cost:\n"
        f"    gpuType: {ARCH_TO_GPU_TYPE[profile.arch]}\n"
        f"    gpusPerNode: {profile.gpus_per_node}\n"
        "  runtime:\n"
        f"    image: {IMAGE}\n"
        f"    imagePullPolicy: {IMAGE_PULL_POLICY}\n"
        f"    imageFamily: {IMAGE_FAMILY}\n"
        + (
            "    securityContext:\n      capabilities:\n        add:\n          - SYS_ADMIN\n"
            if profile.sys_admin
            else ""
        )
        + "  policy:\n"
        f"    preemptable: {str(PREEMPTABLE).lower()}\n"
        f"    maxQueueWaitSeconds: {MAX_QUEUE_WAIT_SECONDS}\n"
        f"    checkpointOnPreempt: {str(CHECKPOINT_ON_PREEMPT).lower()}\n"
    )


def render_pack_yaml() -> str:
    """Render the full multi-document YAML for the pack."""
    out = io.StringIO()
    out.write(_HEADER)
    for profile in all_profiles():
        out.write("\n")
        out.write(_render_one(profile))
    return out.getvalue()
