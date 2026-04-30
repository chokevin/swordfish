"""Result-fetch helper for level-1 (submit-only) swordfish runs.

V0/V1 rune does not have `rune submit get` to read back result JSONs the way
`rune eval get` does for level-2 managed evals. Until swordfish promotes to
level-2 (a swordfish-bench harness with `rune eval --harness ...`), the
result JSON sits on the training-nfs PVC at the path the in-pod entrypoint
wrote it to.

This helper grabs it back via `kubectl cp` from any pod that has the PVC
mounted. Strategy:

1. Look for the actual benchmark pod (still around if --grace-period is
   long enough or the Job hasn't been GC'd).
2. Otherwise, look for any "shell" or "helper" pod in the namespace with the
   same PVC mounted.
3. Otherwise, instruct the user to start one via `rune shell --pvc ...`.

The function is a best-effort convenience. For audited result transfer,
promote to level-2 and use `rune eval get`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ResultFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchedResult:
    name: str
    pod: str
    remote_path: str
    local_path: Path

    @property
    def parsed(self) -> dict:
        return json.loads(self.local_path.read_text())


def fetch_result(
    *,
    job_name: str,
    remote_path: str,
    local_path: Path | str,
    namespace: str = "ray",
    context: str | None = None,
    pod: str | None = None,
    pod_label_selector: str | None = None,
    kubectl_bin: str = "kubectl",
) -> FetchedResult:
    """Copy a result JSON from a cluster PVC back to the local filesystem.

    Args:
        job_name: rune job name (used as a label fallback to find the pod).
        remote_path: absolute path inside the pod (e.g. /data/swordfish/.../x.json).
        local_path: where to write the file locally.
        pod: explicit pod name override. Auto-discovered if None.
        pod_label_selector: explicit label selector to find a fallback helper
            pod (e.g. "app=research-shell"). Defaults to looking for a pod
            with kueue.x-k8s.io/job-name=<job_name>.
    """
    if shutil.which(kubectl_bin) is None:
        raise ResultFetchError(f"{kubectl_bin} not on PATH; install kubectl or pass kubectl_bin=")

    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)

    target_pod = pod or _find_pod(
        namespace=namespace,
        context=context,
        kubectl_bin=kubectl_bin,
        primary_selector=f"kueue.x-k8s.io/job-name={job_name}",
        fallback_selector=pod_label_selector,
    )
    if not target_pod:
        raise ResultFetchError(
            f"could not find a pod for job {job_name!r} or matching {pod_label_selector!r}; "
            "start one with `rune shell --pvc training-nfs` or pass pod="
        )

    args = [kubectl_bin]
    if context:
        args += ["--context", context]
    args += ["-n", namespace, "cp", f"{target_pod}:{remote_path}", str(local)]
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise ResultFetchError(
            f"kubectl cp failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    return FetchedResult(
        name=job_name,
        pod=target_pod,
        remote_path=remote_path,
        local_path=local,
    )


def _find_pod(
    *,
    namespace: str,
    context: str | None,
    kubectl_bin: str,
    primary_selector: str,
    fallback_selector: str | None,
) -> str | None:
    for selector in (primary_selector, fallback_selector):
        if not selector:
            continue
        args = [kubectl_bin]
        if context:
            args += ["--context", context]
        args += [
            "-n",
            namespace,
            "get",
            "pods",
            "-l",
            selector,
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ]
        try:
            proc = subprocess.run(
                args, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
            )
        except FileNotFoundError:
            return None
        name = proc.stdout.strip()
        if name:
            return name
    return None
