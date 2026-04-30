"""Result-fetch helpers for level-1 (submit-only) swordfish runs.

Two paths exist:

1. **`fetch_via_rune_submit_get`** (preferred): shells out to
   `rune submit get NAME --output raw` which uses the `airun.aks.io/result-{path,pvc}`
   annotations recorded by `rune submit --output PATH` and pulls the file via a
   one-shot helper Pod. Binary-safe (returns `bytes`).
2. **`fetch_result`** (legacy / overrides): shells out to `kubectl cp` from the
   benchmark pod or any pod with the PVC mounted. Used for jobs submitted
   without `--output`, or when the caller wants a specific pod (debugging).

The `LigerPerkernelRun.fetch_result` helper prefers (1) and only falls back to
(2) when annotations are missing on the Job (legacy case) or the caller passes
explicit `pod=` / `pod_label_selector=`. All other errors propagate so callers
notice them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ResultFetchError(RuntimeError):
    pass


class RuneSubmitGetMissingAnnotationsError(ResultFetchError):
    """Raised when `rune submit get` reports the Job has no result-path annotation.

    Callers can catch this specifically to fall back to legacy kubectl-cp.
    All other failures (auth, missing artifact, helper-pod failure) raise the
    base `ResultFetchError` and should propagate.
    """


@dataclass(frozen=True)
class FetchedResult:
    name: str
    pod: str
    remote_path: str
    local_path: Path

    @property
    def parsed(self) -> dict:
        return json.loads(self.local_path.read_text())


def fetch_via_rune_submit_get(
    *,
    name: str,
    namespace: str = "ray",
    context: str | None = None,
    path: str | None = None,
    pvc: str | None = None,
    artifact: str | None = None,
    rune_bin: str = "rune",
) -> bytes:
    """Pull a result file from the cluster via `rune submit get NAME -o raw`.

    Returns the raw bytes verbatim. Caller decides whether to decode (for JSON)
    or write to disk (for `.ncu-rep` / `.nsys-rep` binary traces).

    Override semantics (see ai2/applications/rune/internal/cli/submit_get.go):
    - If neither `path` nor `pvc` is set, rune reads the recorded
      `airun.aks.io/result-path` and `airun.aks.io/result-pvc` annotations.
    - If `path` is set, it overrides the recorded path. PVC must come from
      either `pvc=` or the recorded annotation.
    - `artifact` is appended to the path (must be a directory). For example,
      a profile-mode trace lives at `/data/<name>/profile/profile.<ext>`,
      so pass `path=/data/<name>/profile`, `artifact=profile.<ext>`.

    Raises `RuneSubmitGetMissingAnnotationsError` if rune reports the Job has
    no recorded path (subclass of `ResultFetchError`); raises `ResultFetchError`
    for everything else.
    """
    if shutil.which(rune_bin) is None:
        raise ResultFetchError(f"{rune_bin} not on PATH; install rune or pass rune_bin=")

    args: list[str] = [rune_bin, "submit", "get", name, "-n", namespace, "--output", "raw"]
    if context:
        args += ["--context", context]
    if path:
        args += ["--path", path]
    if pvc:
        args += ["--pvc", pvc]
    if artifact:
        args += ["--artifact", artifact]

    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        # Decode stderr for inspection (rune prints English errors there).
        stderr = proc.stderr.decode("utf-8", errors="replace")
        # Rune writes "has no airun.aks.io/result-path" when the Job's
        # annotations are missing — see internal/cli/submit_get.go:resolveTarget.
        if "has no airun.aks.io/result-path" in stderr:
            raise RuneSubmitGetMissingAnnotationsError(
                f"job {name!r} has no airun.aks.io/result-path annotation; "
                "resubmit with `--output PATH` or pass --path/--pvc explicitly. "
                f"rune stderr: {stderr.strip()}"
            )
        raise ResultFetchError(f"rune submit get failed ({proc.returncode}): {stderr.strip()}")
    return proc.stdout


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

    Legacy / explicit-pod path. Prefer `fetch_via_rune_submit_get` for jobs
    submitted with `rune submit --output PATH`.

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
