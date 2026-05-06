"""A100 Nsight Compute profiling window helpers for airun."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass


class DcgmWindowError(RuntimeError):
    pass


@dataclass(frozen=True)
class DcgmPod:
    name: str
    node: str
    phase: str
    ready: bool


@dataclass(frozen=True)
class DcgmWindowStatus:
    a100_nodes: tuple[str, ...]
    a100_exporter_pods: tuple[DcgmPod, ...]
    desired: int
    ready: int
    updated: int
    available: int

    @property
    def a100_clear(self) -> bool:
        return not self.a100_exporter_pods


def dcgm_window_status(
    *,
    context: str | None = None,
    namespace: str = "gpu-operator",
    daemonset: str = "nvidia-dcgm-exporter",
    app_label: str = "app=nvidia-dcgm-exporter",
    a100_selector: str = "nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB",
) -> DcgmWindowStatus:
    a100_nodes = _a100_nodes(context=context, selector=a100_selector)
    pods = _dcgm_pods(context=context, namespace=namespace, app_label=app_label)
    ds = _kubectl_json(
        ["-n", namespace, "get", "ds", daemonset, "-o", "json"],
        context=context,
    )
    status = ds.get("status", {})
    a100_set = set(a100_nodes)
    return DcgmWindowStatus(
        a100_nodes=tuple(a100_nodes),
        a100_exporter_pods=tuple(pod for pod in pods if pod.node in a100_set),
        desired=int(status.get("desiredNumberScheduled", 0)),
        ready=int(status.get("numberReady", 0)),
        updated=int(status.get("updatedNumberScheduled", 0)),
        available=int(status.get("numberAvailable", 0)),
    )


def pause_a100_dcgm(
    *,
    context: str | None = None,
    namespace: str = "gpu-operator",
    daemonset: str = "nvidia-dcgm-exporter",
    app_label: str = "app=nvidia-dcgm-exporter",
    a100_selector: str = "nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB",
    timeout_seconds: int = 300,
    poll_interval_seconds: float = 5.0,
) -> DcgmWindowStatus:
    """Exclude A100 nodes from DCGM exporter and delete existing A100 exporter pods."""
    patch = {
        "spec": {
            "template": {
                "spec": {
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [
                                    {
                                        "matchExpressions": [
                                            {
                                                "key": "nvidia.com/gpu.product",
                                                "operator": "NotIn",
                                                "values": ["NVIDIA-A100-SXM4-80GB"],
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
    }
    _kubectl(
        ["-n", namespace, "patch", "ds", daemonset, "--type", "merge", "-p", json.dumps(patch)],
        context=context,
    )
    status = dcgm_window_status(
        context=context,
        namespace=namespace,
        daemonset=daemonset,
        app_label=app_label,
        a100_selector=a100_selector,
    )
    if status.a100_exporter_pods:
        _kubectl(
            [
                "-n",
                namespace,
                "delete",
                "pod",
                *(pod.name for pod in status.a100_exporter_pods),
                "--wait=true",
                "--timeout=120s",
            ],
            context=context,
        )
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = dcgm_window_status(
            context=context,
            namespace=namespace,
            daemonset=daemonset,
            app_label=app_label,
            a100_selector=a100_selector,
        )
        if status.a100_clear:
            return status
        if time.monotonic() >= deadline:
            names = ", ".join(pod.name for pod in status.a100_exporter_pods)
            raise DcgmWindowError(f"DCGM exporter still running on A100 nodes: {names}")
        time.sleep(poll_interval_seconds)


def restore_dcgm(
    *,
    context: str | None = None,
    namespace: str = "gpu-operator",
    daemonset: str = "nvidia-dcgm-exporter",
    timeout_seconds: int = 600,
) -> DcgmWindowStatus:
    patch = {"spec": {"template": {"spec": {"affinity": None}}}}
    _kubectl(
        ["-n", namespace, "patch", "ds", daemonset, "--type", "merge", "-p", json.dumps(patch)],
        context=context,
    )
    _kubectl(
        ["-n", namespace, "rollout", "status", f"ds/{daemonset}", f"--timeout={timeout_seconds}s"],
        context=context,
    )
    return dcgm_window_status(context=context, namespace=namespace, daemonset=daemonset)


def format_dcgm_status(status: DcgmWindowStatus) -> str:
    pods = ", ".join(f"{p.name}@{p.node}:{p.phase}" for p in status.a100_exporter_pods) or "none"
    return "\n".join(
        [
            f"A100 nodes: {', '.join(status.a100_nodes) or 'none'}",
            f"A100 exporter pods: {pods}",
            "DaemonSet: "
            f"desired={status.desired} ready={status.ready} "
            f"updated={status.updated} available={status.available}",
        ]
    )


def _a100_nodes(*, context: str | None, selector: str) -> list[str]:
    payload = _kubectl_json(["get", "nodes", "-l", selector, "-o", "json"], context=context)
    return [item["metadata"]["name"] for item in payload.get("items", [])]


def _dcgm_pods(*, context: str | None, namespace: str, app_label: str) -> list[DcgmPod]:
    payload = _kubectl_json(
        ["-n", namespace, "get", "pods", "-l", app_label, "-o", "json"],
        context=context,
    )
    out: list[DcgmPod] = []
    for item in payload.get("items", []):
        statuses = item.get("status", {}).get("containerStatuses", [])
        ready = bool(statuses and all(s.get("ready") for s in statuses))
        out.append(
            DcgmPod(
                name=item["metadata"]["name"],
                node=item.get("spec", {}).get("nodeName", ""),
                phase=item.get("status", {}).get("phase", ""),
                ready=ready,
            )
        )
    return out


def _kubectl_json(args: list[str], *, context: str | None) -> dict:
    proc = _kubectl(args, context=context)
    return json.loads(proc.stdout)


def _kubectl(args: list[str], *, context: str | None) -> subprocess.CompletedProcess[str]:
    argv = ["kubectl"]
    if context:
        argv += ["--context", context]
    argv += args
    proc = subprocess.run(
        argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if proc.returncode != 0:
        raise DcgmWindowError(
            f"kubectl failed ({proc.returncode}): {' '.join(argv)}\n"
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc
