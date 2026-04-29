"""Render airun/Kueue jobs for the Week 1 GEMM benchmark matrix."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from swordfish.runner.backends import available_gemm_backends
from swordfish.runner.matrix import DEFAULT_ARCH_LABELS
from swordfish.runner.schema import GPU_PEAKS, NCU_METRICS


class LiteralString(str):
    """Marker for strings that should be emitted as YAML block scalars."""


@dataclass(frozen=True)
class AirunArchConfig:
    arch: str
    queue: str
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: list[dict[str, Any]] = field(default_factory=list)
    gpu_resource: str = "nvidia.com/gpu"
    gpu_count: int = 1
    resource_claim_template: str | None = None
    container_security_context: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, arch: str, raw: dict[str, Any]) -> "AirunArchConfig":
        if arch not in GPU_PEAKS:
            raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(GPU_PEAKS)}")
        queue = raw.get("queue")
        if not isinstance(queue, str) or not queue:
            raise ValueError(f"arch {arch!r} must set a non-empty queue")
        gpu_count = int(raw.get("gpu_count", 1))
        if gpu_count < 1:
            raise ValueError(f"arch {arch!r} must request at least one GPU")
        container_security_context = raw.get("container_security_context")
        if container_security_context is not None and not isinstance(
            container_security_context, dict
        ):
            raise ValueError(f"arch {arch!r} container_security_context must be an object")
        return cls(
            arch=arch,
            queue=queue,
            node_selector=dict(raw.get("node_selector", {})),
            tolerations=list(raw.get("tolerations", [])),
            gpu_resource=str(raw.get("gpu_resource", "nvidia.com/gpu")),
            gpu_count=gpu_count,
            resource_claim_template=raw.get("resource_claim_template"),
            container_security_context=container_security_context,
        )


@dataclass(frozen=True)
class AirunGemmConfig:
    namespace: str
    image: str
    output_dir: str
    archs: list[AirunArchConfig]
    kubectl_context: str | None = None
    python_command: list[str] = field(default_factory=lambda: ["uv", "run", "python"])
    working_dir: str = "/workspace/swordfish"
    image_pull_policy: str = "IfNotPresent"
    service_account: str | None = None
    priority_class: str | None = None
    pvc_claim_name: str | None = None
    pvc_mount_path: str | None = None
    cpu_request: str | None = None
    memory_request: str | None = None
    cpu_limit: str | None = None
    memory_limit: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    name_prefix: str = "swordfish-gemm"
    result_prefix: str = "torch-gemm"
    backend: str = "torch"
    profile_ncu: bool = True
    m: int = 4096
    n: int = 4096
    k: int = 4096
    dtype: str = "fp16"
    repeats: int = 5
    warmup: int = 10
    iters: int = 50
    seed: int = 0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "AirunGemmConfig":
        archs_raw = raw.get("archs")
        if not isinstance(archs_raw, dict) or not archs_raw:
            raise ValueError("config must include a non-empty archs object")

        archs = [AirunArchConfig.from_mapping(arch, cfg) for arch, cfg in archs_raw.items()]
        for default_arch in DEFAULT_ARCH_LABELS:
            if default_arch not in archs_raw:
                raise ValueError(f"config must include archs.{default_arch}")

        namespace = raw.get("namespace")
        image = raw.get("image")
        output_dir = raw.get("output_dir")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("config must set namespace")
        if not isinstance(image, str) or not image:
            raise ValueError("config must set image")
        if not isinstance(output_dir, str) or not output_dir.startswith("/"):
            raise ValueError("config must set an absolute output_dir")
        backend = str(raw.get("backend", "torch"))
        if backend not in available_gemm_backends():
            raise ValueError(
                f"unknown GEMM backend {backend!r}; expected one of {available_gemm_backends()}"
            )

        pvc = raw.get("pvc", {})
        resources = raw.get("resources", {})
        python_command = raw.get("python_command", ["uv", "run", "python"])
        if isinstance(python_command, str):
            python_command = [python_command]
        if not isinstance(python_command, list) or not all(
            isinstance(part, str) and part for part in python_command
        ):
            raise ValueError("python_command must be a non-empty string list")
        return cls(
            namespace=namespace,
            image=image,
            output_dir=output_dir,
            archs=archs,
            kubectl_context=raw.get("kubectl_context"),
            python_command=python_command,
            working_dir=str(raw.get("working_dir", "/workspace/swordfish")),
            image_pull_policy=str(raw.get("image_pull_policy", "IfNotPresent")),
            service_account=raw.get("service_account"),
            priority_class=raw.get("priority_class"),
            pvc_claim_name=pvc.get("claim_name") if isinstance(pvc, dict) else None,
            pvc_mount_path=pvc.get("mount_path") if isinstance(pvc, dict) else None,
            cpu_request=resources.get("cpu_request") if isinstance(resources, dict) else None,
            memory_request=resources.get("memory_request") if isinstance(resources, dict) else None,
            cpu_limit=resources.get("cpu_limit") if isinstance(resources, dict) else None,
            memory_limit=resources.get("memory_limit") if isinstance(resources, dict) else None,
            env={str(k): str(v) for k, v in raw.get("env", {}).items()},
            name_prefix=str(raw.get("name_prefix", "swordfish-gemm")),
            result_prefix=str(raw.get("result_prefix", f"{backend}-gemm")),
            backend=backend,
            profile_ncu=bool(raw.get("profile_ncu", True)),
            m=int(raw.get("m", 4096)),
            n=int(raw.get("n", 4096)),
            k=int(raw.get("k", 4096)),
            dtype=str(raw.get("dtype", "fp16")),
            repeats=int(raw.get("repeats", 5)),
            warmup=int(raw.get("warmup", 10)),
            iters=int(raw.get("iters", 50)),
            seed=int(raw.get("seed", 0)),
        )


def load_airun_gemm_config(path: Path) -> AirunGemmConfig:
    return AirunGemmConfig.from_mapping(json.loads(path.read_text()))


def _gemm_args(config: AirunGemmConfig, arch: AirunArchConfig, out_path: str) -> list[str]:
    return [
        *config.python_command,
        "-m",
        "swordfish.runner",
        "run-gemm",
        "--backend",
        config.backend,
        "--m",
        str(config.m),
        "--n",
        str(config.n),
        "--k",
        str(config.k),
        "--dtype",
        config.dtype,
        "--repeats",
        str(config.repeats),
        "--warmup",
        str(config.warmup),
        "--iters",
        str(config.iters),
        "--device",
        "auto",
        "--arch-label",
        arch.arch,
        "--seed",
        str(config.seed),
        "--out",
        out_path,
    ]


def _import_probe_script(config: AirunGemmConfig) -> list[str]:
    probe = shlex.join([*config.python_command, "-c", "import swordfish.runner"])
    return [
        "for attempt in $(seq 1 30); do",
        f"  if {probe} >/dev/null 2>&1; then",
        "    break",
        "  fi",
        '  if [ "$attempt" -eq 30 ]; then',
        f"    {probe}",
        "  fi",
        "  sleep 2",
        "done",
    ]


def _job_script(config: AirunGemmConfig, arch: AirunArchConfig) -> str:
    output_dir = str(PurePosixPath(config.output_dir))
    final_path = str(PurePosixPath(output_dir) / f"{config.result_prefix}-{arch.arch}.json")
    raw_path = str(PurePosixPath(output_dir) / f"{config.result_prefix}-{arch.arch}.raw.json")
    ncu_raw_path = str(
        PurePosixPath(output_dir) / f"{config.result_prefix}-{arch.arch}.profile.raw.json"
    )
    ncu_path = str(PurePosixPath(output_dir) / f"{config.result_prefix}-{arch.arch}.ncu.csv")

    lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(config.working_dir)}",
        *_import_probe_script(config),
        f"mkdir -p {shlex.quote(output_dir)}",
    ]
    if config.profile_ncu:
        timing_cmd = _gemm_args(config, arch, raw_path)
        ncu_cmd = [
            "ncu",
            "--csv",
            "--metrics",
            ",".join(NCU_METRICS),
            "--target-processes",
            "all",
            "--replay-mode",
            "kernel",
            *_gemm_args(config, arch, ncu_raw_path),
        ]
        attach_cmd = [
            *config.python_command,
            "-m",
            "swordfish.runner",
            "attach-ncu",
            "--result",
            raw_path,
            "--ncu-csv",
            ncu_path,
            "--out",
            final_path,
        ]
        lines.extend(
            [
                shlex.join(timing_cmd),
                f"if ! {shlex.join(ncu_cmd)} > {shlex.quote(ncu_path)}; then",
                '  echo "WARNING: ncu failed; attaching partial profiler output" >&2',
                "fi",
                f"test -s {shlex.quote(raw_path)}",
                shlex.join(attach_cmd),
            ]
        )
    else:
        lines.append(shlex.join(_gemm_args(config, arch, final_path)))

    lines.append(
        f"{shlex.join(config.python_command)} -m json.tool {shlex.quote(final_path)} >/dev/null"
    )
    return "\n".join(lines)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(str(value))


def _is_scalar(value: Any) -> bool:
    return value is None or (
        not isinstance(value, LiteralString) and isinstance(value, str | int | float | bool)
    )


def _dump_yaml(value: Any, indent: int = 0) -> list[str]:
    pad = " " * indent
    if isinstance(value, LiteralString):
        lines = [f"{pad}|-"]
        lines.extend(f"{pad}  {line}" if line else f"{pad}  " for line in value.splitlines())
        return lines
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if item is None:
                continue
            if _is_scalar(item):
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
            else:
                lines.append(f"{pad}{key}:")
                lines.extend(_dump_yaml(item, indent + 2))
        return lines or [f"{pad}{{}}"]
    if isinstance(value, list):
        lines = []
        for item in value:
            if _is_scalar(item):
                lines.append(f"{pad}- {_yaml_scalar(item)}")
            else:
                lines.append(f"{pad}-")
                lines.extend(_dump_yaml(item, indent + 2))
        return lines or [f"{pad}[]"]
    return [f"{pad}{_yaml_scalar(value)}"]


def _container_resources(config: AirunGemmConfig, arch: AirunArchConfig) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    requests: dict[str, str | int] = {}
    limits: dict[str, str | int] = {}

    if arch.resource_claim_template:
        resources["claims"] = [{"name": "gpu"}]
    else:
        requests[arch.gpu_resource] = arch.gpu_count
        limits[arch.gpu_resource] = arch.gpu_count

    if config.cpu_request:
        requests["cpu"] = config.cpu_request
    if config.memory_request:
        requests["memory"] = config.memory_request
    if config.cpu_limit:
        limits["cpu"] = config.cpu_limit
    if config.memory_limit:
        limits["memory"] = config.memory_limit

    if requests:
        resources["requests"] = requests
    if limits:
        resources["limits"] = limits
    return resources


def render_airun_gemm_job(config: AirunGemmConfig, arch: AirunArchConfig) -> str:
    labels = {
        "app.kubernetes.io/name": "swordfish-runner",
        "swordfish.dev/benchmark": f"{config.backend}-gemm",
        "swordfish.dev/backend": config.backend,
        "swordfish.dev/gpu-class": arch.arch,
        "kueue.x-k8s.io/queue-name": arch.queue,
    }
    env = {"CONTAINER_IMAGE": config.image, **config.env}
    container: dict[str, Any] = {
        "name": "runner",
        "image": config.image,
        "imagePullPolicy": config.image_pull_policy,
        "command": ["/bin/bash", "-lc"],
        "args": [LiteralString(_job_script(config, arch))],
        "env": [{"name": name, "value": value} for name, value in env.items()],
    }

    if config.pvc_claim_name and config.pvc_mount_path:
        container["volumeMounts"] = [{"name": "results", "mountPath": config.pvc_mount_path}]
    if arch.container_security_context:
        container["securityContext"] = arch.container_security_context

    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container],
    }
    if config.service_account:
        pod_spec["serviceAccountName"] = config.service_account
    if config.priority_class:
        pod_spec["priorityClassName"] = config.priority_class
    if arch.node_selector:
        pod_spec["nodeSelector"] = arch.node_selector
    if arch.tolerations:
        pod_spec["tolerations"] = arch.tolerations
    if config.pvc_claim_name:
        pod_spec["volumes"] = [
            {"name": "results", "persistentVolumeClaim": {"claimName": config.pvc_claim_name}}
        ]

    if arch.resource_claim_template:
        pod_spec["resourceClaims"] = [
            {"name": "gpu", "resourceClaimTemplateName": arch.resource_claim_template}
        ]
    container["resources"] = _container_resources(config, arch)

    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"{config.name_prefix}-{arch.arch}",
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "backoffLimit": 0,
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }
    return "\n".join(_dump_yaml(job)) + "\n"


def _shell_array(name: str, values: list[str]) -> str:
    return f"{name}=({shlex.join(values)})"


def _selector_arg(selector: dict[str, str]) -> str | None:
    if not selector:
        return None
    return ",".join(f"{key}={value}" for key, value in sorted(selector.items()))


def _security_context_adds_capability(
    security_context: dict[str, Any] | None, capability: str
) -> bool:
    if not isinstance(security_context, dict):
        return False
    capabilities = security_context.get("capabilities")
    if not isinstance(capabilities, dict):
        return False
    added = capabilities.get("add", [])
    return isinstance(added, list) and capability in added


def render_airun_preflight_script(
    config: AirunGemmConfig,
    *,
    arch_label: str,
    blocker_pod: str | None = None,
) -> str:
    """Render a bash preflight that fails before submitting into a known-bad route."""
    try:
        arch = next(candidate for candidate in config.archs if candidate.arch == arch_label)
    except StopIteration as exc:
        raise ValueError(f"config does not contain arch {arch_label!r}") from exc

    kubectl = ["kubectl"]
    if config.kubectl_context:
        kubectl.extend(["--context", config.kubectl_context])
    selector = _selector_arg(arch.node_selector)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        _shell_array("KUBECTL", kubectl),
        f"NS={shlex.quote(config.namespace)}",
        "failed=0",
        f"echo 'preflighting airun arch {arch.arch}'",
        'echo "== Kueue queues =="',
        '"${KUBECTL[@]}" -n "$NS" get localqueue || failed=1',
        f'"${{KUBECTL[@]}}" -n "$NS" get localqueue {shlex.quote(arch.queue)} '
        ">/dev/null || failed=1",
        '"${KUBECTL[@]}" get clusterqueue || failed=1',
    ]
    if arch.resource_claim_template:
        lines.extend(
            [
                'echo "== DRA resource claim template =="',
                f'"${{KUBECTL[@]}}" -n "$NS" get resourceclaimtemplate '
                f"{shlex.quote(arch.resource_claim_template)} >/dev/null || failed=1",
            ]
        )
    lines.append('echo "== matching nodes =="')
    if selector:
        quoted_selector = shlex.quote(selector)
        lines.extend(
            [
                'node_rows="$("${KUBECTL[@]}" get nodes '
                f"-l {quoted_selector} "
                '-o jsonpath=\'{range .items[*]}{.metadata.name}{"\\t"}'
                '{.spec.unschedulable}{"\\t"}'
                '{range .status.conditions[?(@.type=="Ready")]}{.status}{end}'
                '{"\\n"}{end}\' 2>/dev/null || true)"',
                'if [ -z "$node_rows" ]; then',
                f"  echo 'ERROR: no live nodes match selector {selector}' >&2",
                "  failed=1",
                "elif ! printf '%s\\n' \"$node_rows\" | "
                'awk -F \'\\t\' \'($2 != "true" && $3 == "True") { found=1 } '
                "END { exit found ? 0 : 1 }'; then",
                f"  echo 'ERROR: selector {selector} matched nodes, "
                "but none are Ready and schedulable' >&2",
                "  printf '%s\\n' \"$node_rows\" >&2",
                "  failed=1",
                "else",
                f'  "${{KUBECTL[@]}}" get nodes -l {quoted_selector} '
                "-L gpu,nvidia.com/gpu.product,kubernetes.azure.com/agentpool",
                "fi",
            ]
        )
    else:
        lines.append('"${KUBECTL[@]}" get nodes -L gpu,nvidia.com/gpu.product || failed=1')

    lines.extend(
        [
            'echo "== arch nodepools =="',
            f'"${{KUBECTL[@]}}" get nodepools -A 2>/dev/null | grep -Ei '
            f"{shlex.quote(arch.arch + '|NAME')} || true",
        ]
    )

    if blocker_pod:
        quoted_blocker = shlex.quote(blocker_pod)
        lines.extend(
            [
                'echo "== known blocker pod =="',
                f'if "${{KUBECTL[@]}}" -n "$NS" get pod {quoted_blocker} >/dev/null 2>&1; then',
                f'  blocker_phase="$("${{KUBECTL[@]}}" -n "$NS" get pod {quoted_blocker} '
                "-o jsonpath='{.status.phase}')\"",
                f'  blocker_deleted="$("${{KUBECTL[@]}}" -n "$NS" get pod {quoted_blocker} '
                "-o jsonpath='{.metadata.deletionTimestamp}')\"",
                f'  "${{KUBECTL[@]}}" -n "$NS" get pod {quoted_blocker} -o wide || true',
                '  if [ "$blocker_phase" = "Failed" ] && [ -n "$blocker_deleted" ]; then',
                f"    echo 'WARNING: known blocker pod {blocker_pod} is Failed and "
                "deletion-marked; continuing because live nodes passed' >&2",
                "  else",
                f"    echo 'ERROR: known blocker pod {blocker_pod} still exists' >&2",
                "    failed=1",
                "  fi",
                "fi",
            ]
        )

    if config.profile_ncu and arch.arch == "a100":
        if not _security_context_adds_capability(arch.container_security_context, "SYS_ADMIN"):
            lines.extend(
                [
                    'echo "== A100 NCU permissions =="',
                    "echo 'ERROR: A100 NCU profiling requires container_security_context "
                    'capabilities.add: ["SYS_ADMIN"] in the arch config\' >&2',
                    "failed=1",
                ]
            )
        lines.extend(
            [
                'echo "== DCGM exporter / NCU profiler conflict =="',
                '"${KUBECTL[@]}" -n gpu-operator get pods '
                "-l app=nvidia-dcgm-exporter -o wide || true",
                'dcgm_rows="$("${KUBECTL[@]}" -n gpu-operator get pods '
                "-l app=nvidia-dcgm-exporter "
                '-o jsonpath=\'{range .items[*]}{.metadata.name}{"\\t"}'
                '{.spec.nodeName}{"\\t"}{.status.phase}{"\\n"}{end}\' '
                '2>/dev/null || true)"',
                'conflicting_dcgm=""',
                'if [ -z "${node_rows:-}" ]; then',
                "  echo 'WARNING: cannot map target A100 nodes for DCGM check; "
                "set a node_selector for precise NCU preflight' >&2",
                'elif [ -n "$dcgm_rows" ]; then',
                "  conflicting_dcgm=\"$(awk -F '\\t' '",
                "    NR == FNR {",
                '      if ($2 != "true" && $3 == "True") ready[$1] = 1',
                "      next",
                "    }",
                '    $3 == "Running" && ready[$2] { print $2 }',
                "  ' <(printf '%s\\n' \"$node_rows\") "
                '<(printf \'%s\\n\' "$dcgm_rows") | sort -u)"',
                "fi",
                'if [ -n "$conflicting_dcgm" ]; then',
                "  echo 'ERROR: nvidia-dcgm-exporter is running on target A100 "
                "node(s), and DCGM profiling metrics contend with Nsight Compute' >&2",
                "  printf '%s\\n' \"$conflicting_dcgm\" >&2",
                "  echo 'ERROR: temporarily exclude DCGM exporter from the target A100 "
                "node(s), rerun this preflight, run NCU, then restore DCGM immediately' >&2",
                "  failed=1",
                "else",
                "  echo 'No running DCGM exporter pods found on Ready target A100 nodes'",
                "fi",
            ]
        )

    lines.extend(
        [
            'if [ "$failed" -ne 0 ]; then',
            f"  echo 'preflight failed for {arch.arch}; do not submit benchmark jobs' >&2",
            "  exit 2",
            "fi",
            f"echo 'preflight passed for {arch.arch}'",
        ]
    )
    return "\n".join(lines) + "\n"


def _select_archs(config: AirunGemmConfig, arch_labels: list[str] | None) -> list[AirunArchConfig]:
    if arch_labels is None:
        return config.archs
    requested = set(arch_labels)
    known = {arch.arch for arch in config.archs}
    unknown = requested - known
    if unknown:
        raise ValueError(f"config does not contain archs: {sorted(unknown)}")
    return [arch for arch in config.archs if arch.arch in requested]


def write_airun_gemm_manifests(
    config: AirunGemmConfig,
    manifest_dir: Path,
    *,
    arch_labels: list[str] | None = None,
) -> list[Path]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for arch in _select_archs(config, arch_labels):
        out_path = manifest_dir / f"{config.name_prefix}-{arch.arch}.yaml"
        out_path.write_text(render_airun_gemm_job(config, arch))
        written.append(out_path)
    return written


def write_airun_preflight_script(
    config: AirunGemmConfig,
    *,
    arch_label: str,
    out_path: Path,
    blocker_pod: str | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_airun_preflight_script(config, arch_label=arch_label, blocker_pod=blocker_pod)
    )
    out_path.chmod(0o755)
    return out_path
