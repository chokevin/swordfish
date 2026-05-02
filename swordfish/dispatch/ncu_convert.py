"""Cluster-side `.ncu-rep → .ncu.csv` converter.

When a benchmark runs with `rune --profile-mode=ncu`, the cluster writes a
binary `profile.ncu-rep` to the PVC. Reading it locally requires NVIDIA's
Nsight Compute (`ncu_report` Python module) installed on the dev box. Mac
users get this via `brew install --cask nvidia-nsight-compute`. Linux
agents / CI runners without nsight installed need a different path.

This helper closes the gap: it spawns a tiny CPU-only Pod (no GPU, no
Kueue admission since kernel-mode-* queues aren't required for non-GPU
work) that mounts the same PVC and runs `ncu --import .ncu-rep --csv`
to produce a `.ncu-summary.csv` companion file alongside the binary.

The Pod:
- runs the same swordfish-bench image (which has ncu CLI baked in);
- mounts `training-nfs` at `/data` (read-write — needs to write the CSV);
- has `restartPolicy: Never` and is deleted on success;
- runs as the same UID as the benchmarks (matters because the PVC is
  mode-755 and was originally written by `nobody`).

After conversion:
- `inspect-run --profile-mode ncu` finds the new `profile.ncu-summary.csv`
  via the existing `*.ncu.csv` glob and prefers it over the `.ncu-rep`.
- The `.ncu-rep` stays on the PVC for power-users who want full Nsight UI.

Why a bare Pod and not `rune submit`:
- Conversion is non-GPU and short-lived (<30s typically); Kueue admission
  delay (often minutes) would dwarf the actual work.
- `rune submit` would also tag this with airun.aks.io annotations meant
  for benchmark runs, which is misleading.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import PurePosixPath


class NcuConvertError(RuntimeError):
    """Raised when the converter Pod fails to produce the expected CSV."""


# Pod-spec template. We render with str.format rather than YAML to avoid
# pulling in pyyaml just for one templated apply. The image and PVC default
# to the same values the benchmark profiles use; override via kwargs.
_POD_SPEC = """\
apiVersion: v1
kind: Pod
metadata:
  name: {pod_name}
  namespace: {namespace}
  labels:
    app.kubernetes.io/name: swordfish-ncu-convert
    app.kubernetes.io/managed-by: swordfish
    swordfish.ai/source-job: {job_name}
spec:
  restartPolicy: Never
  # Scheduled on any node that can mount training-nfs — no GPU constraint.
  # Tolerate spegel-broken H100 nodes since this is a cheap CPU pod that
  # only needs the PVC, not a working image-puller for GPU images.
  tolerations:
    - operator: Exists
  containers:
    - name: convert
      image: {image}
      imagePullPolicy: IfNotPresent
      command: ["bash", "-lc"]
      args:
        - |
          set -euo pipefail
          : "$REP_PATH" "$CSV_PATH"
          if [ ! -f "$REP_PATH" ]; then
            echo "ERROR: $REP_PATH does not exist on PVC" >&2
            exit 2
          fi
          echo "converting $REP_PATH -> $CSV_PATH"
          # `--page details` produces the long-form CSV (one row per
          # kernel×section×metric) that swordfish.runner.ncu_summary.parse_ncu_csv_full
          # expects. `--page raw` produces a wide-form CSV that our parser
          # can't read; do not use it here.
          ncu --import "$REP_PATH" --csv --log-file "$CSV_PATH" --page details
          echo "done; $(wc -c < "$CSV_PATH") bytes written"
      env:
        - name: REP_PATH
          value: {rep_path}
        - name: CSV_PATH
          value: {csv_path}
      volumeMounts:
        - name: data
          mountPath: /data
      resources:
        requests:
          cpu: "100m"
          memory: "256Mi"
        limits:
          cpu: "1"
          memory: "1Gi"
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: {pvc}
"""


@dataclass(frozen=True)
class NcuConvertResult:
    """Where the converter Pod wrote the CSV (path inside the PVC)."""

    pod_name: str
    rep_path: str
    csv_path: str
    elapsed_seconds: float


def submit_ncu_convert(
    *,
    job_name: str,
    namespace: str = "ray",
    pvc: str = "training-nfs",
    image: str = "voiceagentcr.azurecr.io/swordfish-bench:latest",
    rep_path: str | None = None,
    csv_path: str | None = None,
    timeout_seconds: int = 180,
    poll_interval_seconds: float = 2.0,
    cleanup: bool = True,
    kubectl_bin: str = "kubectl",
    context: str | None = None,
) -> NcuConvertResult:
    """Spin up a CPU-only Pod that runs `ncu --import` to write `.ncu-summary.csv`.

    Args:
        job_name: name of the original swordfish job whose .ncu-rep we want
            converted. The default rep/csv paths follow the rune
            `--profile-mode=ncu` convention `/data/<job_name>/profile/`.
        namespace: Kubernetes namespace to launch the converter Pod in.
        pvc: PVC name to mount at /data. Default `training-nfs` matches the
            value baked into the swordfish profile pack.
        image: container image with `ncu` on PATH. Defaults to the
            swordfish-bench image which has Nsight Compute installed.
        rep_path: explicit absolute path to the `.ncu-rep` inside /data.
            Defaults to `/data/<job_name>/profile/profile.ncu-rep`.
        csv_path: explicit absolute path to write the CSV. Defaults to the
            same dir as the rep with extension `.ncu-summary.csv` (matches
            what `inspect-run` looks for via `*.ncu-summary.csv`).
        timeout_seconds: how long to wait for the Pod to reach Succeeded.
        cleanup: delete the Pod on success. Set False to leave it around
            for debugging.
        kubectl_bin: kubectl binary to call (override for testing / paths).
        context: optional kubectl --context value.

    Returns:
        NcuConvertResult describing the Pod and the file paths it touched.

    Raises:
        NcuConvertError: kubectl fails, Pod times out, or Pod reports failure.
    """
    if shutil.which(kubectl_bin) is None:
        raise NcuConvertError(
            f"{kubectl_bin} not found on PATH; install kubectl or pass kubectl_bin="
        )

    rep = rep_path or f"/data/{job_name}/profile/profile.ncu-rep"
    if csv_path is None:
        # Path stem stays the same — only the extension changes. Using
        # `.ncu-summary.csv` (not `.ncu.csv`) keeps it visually distinct
        # from the legacy SWORDFISH_PROFILE=ncu output and matches the
        # naming `inspect-run` already globs for.
        csv = str(PurePosixPath(rep).with_suffix("").with_suffix(".ncu-summary.csv"))
    else:
        csv = csv_path

    pod_name = _make_pod_name(job_name)
    pod_spec = _POD_SPEC.format(
        pod_name=pod_name,
        namespace=namespace,
        job_name=job_name,
        image=image,
        rep_path=rep,
        csv_path=csv,
        pvc=pvc,
    )

    base = [kubectl_bin]
    if context:
        base += ["--context", context]

    started = time.monotonic()
    apply = subprocess.run(
        base + ["apply", "-f", "-"],
        input=pod_spec,
        text=True,
        capture_output=True,
        check=False,
    )
    if apply.returncode != 0:
        raise NcuConvertError(f"kubectl apply failed ({apply.returncode}): {apply.stderr.strip()}")

    try:
        _wait_for_pod_terminal(
            pod_name=pod_name,
            namespace=namespace,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            kubectl_bin=kubectl_bin,
            context=context,
        )
    except NcuConvertError:
        # Capture logs to surface in the error message before cleanup.
        logs = _try_pod_logs(
            pod_name=pod_name,
            namespace=namespace,
            kubectl_bin=kubectl_bin,
            context=context,
        )
        if cleanup:
            _delete_pod_quiet(
                pod_name=pod_name,
                namespace=namespace,
                kubectl_bin=kubectl_bin,
                context=context,
            )
        raise NcuConvertError(f"converter Pod {pod_name} did not succeed. Logs:\n{logs}") from None

    if cleanup:
        _delete_pod_quiet(
            pod_name=pod_name,
            namespace=namespace,
            kubectl_bin=kubectl_bin,
            context=context,
        )

    return NcuConvertResult(
        pod_name=pod_name,
        rep_path=rep,
        csv_path=csv,
        elapsed_seconds=time.monotonic() - started,
    )


def _make_pod_name(job_name: str) -> str:
    """Generate a 1H-unique Pod name. Truncated for the 63-char DNS limit."""
    suffix = format(int(time.time()) % 100_000, "05d")
    base = job_name.lower().replace("_", "-")[:40]
    return f"sf-ncu-convert-{base}-{suffix}"


def _wait_for_pod_terminal(
    *,
    pod_name: str,
    namespace: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
    kubectl_bin: str,
    context: str | None,
) -> None:
    """Poll until the Pod reaches Succeeded or fails. Raises on timeout/Failed."""
    base = [kubectl_bin]
    if context:
        base += ["--context", context]

    deadline = time.monotonic() + timeout_seconds
    last_phase = ""
    while time.monotonic() < deadline:
        proc = subprocess.run(
            base + ["get", "pod", pod_name, "-n", namespace, "-o", "json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            time.sleep(poll_interval_seconds)
            continue
        try:
            phase = json.loads(proc.stdout).get("status", {}).get("phase", "")
        except json.JSONDecodeError:
            phase = ""
        if phase != last_phase:
            last_phase = phase
        if phase == "Succeeded":
            return
        if phase == "Failed":
            raise NcuConvertError(f"pod {pod_name} reached phase=Failed")
        time.sleep(poll_interval_seconds)
    raise NcuConvertError(
        f"pod {pod_name} did not reach Succeeded within {timeout_seconds}s "
        f"(last observed phase: {last_phase or 'unknown'})"
    )


def _try_pod_logs(
    *,
    pod_name: str,
    namespace: str,
    kubectl_bin: str,
    context: str | None,
) -> str:
    base = [kubectl_bin]
    if context:
        base += ["--context", context]
    proc = subprocess.run(
        base + ["logs", pod_name, "-n", namespace, "--tail=200"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout.strip() or "<empty>"
    return f"<could not fetch logs: {proc.stderr.strip()}>"


def _delete_pod_quiet(
    *,
    pod_name: str,
    namespace: str,
    kubectl_bin: str,
    context: str | None,
) -> None:
    base = [kubectl_bin]
    if context:
        base += ["--context", context]
    subprocess.run(
        base + ["delete", "pod", pod_name, "-n", namespace, "--wait=false"],
        capture_output=True,
        text=True,
        check=False,
    )
