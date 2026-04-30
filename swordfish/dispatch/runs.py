"""Typed swordfish benchmark runs that compile to `rune submit` invocations.

The defaults are tuned for the kernel-mode-training queue on voice-agent-flex
and the `ghcr.io/chokevin/swordfish-bench:latest` image. Override per run as
needed.
"""

from __future__ import annotations

import os
import re
import shlex
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from swordfish.dispatch.results import FetchedResult
from swordfish.dispatch.rune import RuneSubmit, RuneSubmitResult


# Defaults — change these in one place, not at every call site.
DEFAULT_NAMESPACE = "ray"
DEFAULT_IMAGE = "ghcr.io/chokevin/swordfish-bench:latest"
DEFAULT_PVC = "training-nfs"
DEFAULT_PRESET = "azure.kernel-mode.training.l"
DEFAULT_BENCH_SCRIPT = Path("infra/rune/scripts/swordfish-bench.sh")
DEFAULT_RESULT_ROOT = "/data/swordfish/week1"
IN_POD_BENCH_SCRIPT = "/work/swordfish/infra/rune/scripts/swordfish-bench.sh"

ARCH_TO_PRESET = {
    "a100": "azure.kernel-mode.training.l",
    "h100": "azure.kernel-mode.training.l",  # same lane today; H100 in kernel-mode pending
    "h200": "azure.kernel-mode.large-memory.xl",
}

LIGER_KERNELS = ("rmsnorm", "swiglu", "rope", "fused_linear_ce")
LIGER_KERNELS_IMPLEMENTED = ("rmsnorm", "swiglu")
PROFILE_MODES = ("ncu", "nsys")
PROFILE_EXTENSIONS = {"ncu": "ncu.csv", "nsys": "nsys-rep"}

_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def _normalize_name(raw: str) -> str:
    """Coerce a benchmark id into a kueue/kubernetes-safe job name."""
    name = raw.lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    if not name:
        raise ValueError(f"name {raw!r} normalizes to empty string")
    if len(name) > 53:
        raise ValueError(f"name {name!r} is too long ({len(name)} chars; max 53)")
    if not _NAME_RE.match(name):
        raise ValueError(f"name {name!r} is not a valid kubernetes object name")
    return name


@dataclass
class LigerPerkernelRun:
    """One paired baseline-vs-Liger per-kernel benchmark on a single arch."""

    kernel: str
    arch: str = "a100"
    dtype: str = "bf16"
    batch: int = 4
    seq: int = 2048
    hidden: int = 4096
    intermediate: int = 14336
    repeats: int = 5
    warmup: int = 10
    iters: int = 50
    name: str | None = None
    namespace: str = DEFAULT_NAMESPACE
    context: str | None = None
    image: str = DEFAULT_IMAGE
    script: str | Path = DEFAULT_BENCH_SCRIPT
    pvc: str = DEFAULT_PVC
    result_root: str = DEFAULT_RESULT_ROOT
    preset: str | None = None  # falls back to ARCH_TO_PRESET[arch]
    profile: str | None = None
    extra_args: list[str] = field(default_factory=list)
    rune_bin: str = "rune"
    profile_mode: str | None = None  # ncu | nsys | None

    def __post_init__(self) -> None:
        if self.kernel not in LIGER_KERNELS:
            raise ValueError(
                f"kernel {self.kernel!r} not in {LIGER_KERNELS}; "
                "rope and fused_linear_ce raise NotImplementedError until follow-up work"
            )
        if self.arch not in ARCH_TO_PRESET:
            raise ValueError(
                f"unknown arch {self.arch!r}; expected one of {sorted(ARCH_TO_PRESET)}"
            )
        if self.preset and self.profile:
            raise ValueError("preset and profile are mutually exclusive")
        if self.profile_mode and self.profile_mode not in PROFILE_MODES:
            raise ValueError(f"profile_mode {self.profile_mode!r} not in {PROFILE_MODES}")

    @property
    def resolved_name(self) -> str:
        return _normalize_name(self.name or f"sf-liger-{self.kernel}-{self.arch}")

    @property
    def resolved_preset(self) -> str:
        if self.profile:
            return ""  # profile path does not use preset
        return self.preset or ARCH_TO_PRESET[self.arch]

    @property
    def out_path(self) -> str:
        return f"{self.result_root}/liger-perkernel/{self.kernel}-{self.arch}.json"

    @property
    def profile_out_path(self) -> str | None:
        if not self.profile_mode:
            return None
        ext = PROFILE_EXTENSIONS[self.profile_mode]
        return f"{self.result_root}/liger-perkernel/{self.kernel}-{self.arch}.{ext}"

    @property
    def forwarded_args(self) -> list[str]:
        return [
            "liger-perkernel",
            "--kernel",
            self.kernel,
            "--batch",
            str(self.batch),
            "--seq",
            str(self.seq),
            "--hidden",
            str(self.hidden),
            "--intermediate",
            str(self.intermediate),
            "--dtype",
            self.dtype,
            "--repeats",
            str(self.repeats),
            "--warmup",
            str(self.warmup),
            "--iters",
            str(self.iters),
            "--device",
            "auto",
            "--arch-label",
            self.arch,
            "--out",
            self.out_path,
        ]

    def _render_profile_wrapper(self) -> Path:
        """Generate a tempfile bash wrapper that exports SWORDFISH_PROFILE then
        execs the in-pod bench script. Used when profile_mode is set so the
        bench script wraps the python invocation in ncu/nsys."""
        assert self.profile_mode is not None
        out_path = self.profile_out_path or ""
        contents = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"export SWORDFISH_PROFILE={shlex.quote(self.profile_mode)}\n"
            f"export SWORDFISH_PROFILE_OUT={shlex.quote(out_path)}\n"
            f'exec bash {shlex.quote(IN_POD_BENCH_SCRIPT)} "$@"\n'
        )
        fd, tmp_path = tempfile.mkstemp(prefix=f"sf-rune-{self.profile_mode}-", suffix=".sh")
        with os.fdopen(fd, "w") as f:
            f.write(contents)
        os.chmod(tmp_path, 0o755)
        return Path(tmp_path)

    def to_rune_submit(self) -> RuneSubmit:
        # When profile_mode is set, the only supported flow is the bash
        # entrypoint (because the wrapper execs into the in-pod bench
        # script). Custom Python scripts under profile mode are out of scope
        # for now — the user can do their own ncu/nsys wrapping inside.
        if self.profile_mode is not None:
            if Path(self.script).resolve() != Path(DEFAULT_BENCH_SCRIPT).resolve():
                raise ValueError(
                    "profile_mode is only supported with the default bench "
                    "script; for custom scripts, wrap the python call yourself"
                )
            script_path: str | Path = self._render_profile_wrapper()
        else:
            script_path = self.script

        kwargs: dict = dict(
            name=self.resolved_name,
            image=self.image,
            script=script_path,
            namespace=self.namespace,
            context=self.context,
            volumes=[f"data=pvc:{self.pvc}"],
            extra_args=list(self.extra_args),
            forwarded_args=self.forwarded_args,
            rune_bin=self.rune_bin,
        )
        if self.profile:
            kwargs["profile"] = self.profile
        else:
            kwargs["preset"] = self.resolved_preset
        return RuneSubmit(**kwargs)

    def to_command(self, *, dry_run: str | None = None) -> str:
        return self.to_rune_submit().to_command(dry_run=dry_run)

    def submit(
        self,
        *,
        dry_run: str | None = None,
        check: bool = True,
        local_image: bool = False,
        push_local: bool = True,
        container_cmd: str = "podman",
        platform: str | None = None,
    ) -> RuneSubmitResult:
        """Dispatch the run.

        With local_image=True, builds the swordfish-bench image from the
        local working tree and (by default) pushes it to GHCR with a
        dev-<sha> tag, then submits with that tag instead of self.image.
        Use this when iterating on swordfish/ internals; for experiments/
        edits, the stable :latest image plus --script is the fast path.
        """
        run = self
        if local_image:
            from swordfish.dispatch.image import build_and_push_dev_image

            new_image = build_and_push_dev_image(
                push=push_local,
                container_cmd=container_cmd,
                platform=platform,
            )
            run = replace(self, image=new_image)
        return run.to_rune_submit().submit(dry_run=dry_run, check=check)

    def fetch_result(
        self,
        local_path: str | Path | None = None,
        *,
        pod: str | None = None,
        pod_label_selector: str | None = None,
        kubectl_bin: str = "kubectl",
        include_traces: bool = False,
    ) -> FetchedResult:
        """Copy this run's result JSON back from the cluster PVC.

        Uses kubectl cp via the benchmark pod (if still around) or any pod
        with the training-nfs PVC mounted. Best-effort convenience until
        swordfish promotes to a level-2 managed eval.

        With include_traces=True, also fetches the profiler artifact
        (`.ncu.csv` for profile_mode='ncu', `.nsys-rep` for 'nsys') next to
        the result JSON.
        """
        from swordfish.dispatch.results import fetch_result as _fetch

        target = (
            Path(local_path)
            if local_path
            else Path("runs/airun/week1") / Path(self.out_path).relative_to("/data/swordfish/week1")
        )
        result = _fetch(
            job_name=self.resolved_name,
            remote_path=self.out_path,
            local_path=target,
            namespace=self.namespace,
            context=self.context,
            pod=pod,
            pod_label_selector=pod_label_selector,
            kubectl_bin=kubectl_bin,
        )

        if include_traces and self.profile_mode and self.profile_out_path:
            trace_local = target.with_suffix("." + PROFILE_EXTENSIONS[self.profile_mode])
            try:
                _fetch(
                    job_name=self.resolved_name,
                    remote_path=self.profile_out_path,
                    local_path=trace_local,
                    namespace=self.namespace,
                    context=self.context,
                    pod=pod or result.pod,
                    pod_label_selector=pod_label_selector,
                    kubectl_bin=kubectl_bin,
                )
            except Exception as e:  # noqa: BLE001
                import sys

                print(
                    f"warning: trace fetch failed ({self.profile_mode} -> {trace_local}): {e}",
                    file=sys.stderr,
                )

        return result


@dataclass
class LigerPerkernelMatrix:
    """Cross-arch x cross-kernel sweep of per-kernel benchmarks."""

    kernels: Iterable[str] = LIGER_KERNELS_IMPLEMENTED
    archs: Iterable[str] = ("a100",)
    dtype: str = "bf16"
    namespace: str = DEFAULT_NAMESPACE
    context: str | None = None
    image: str = DEFAULT_IMAGE
    script: str | Path = DEFAULT_BENCH_SCRIPT
    pvc: str = DEFAULT_PVC
    result_root: str = DEFAULT_RESULT_ROOT
    rune_bin: str = "rune"

    def runs(self) -> list[LigerPerkernelRun]:
        out: list[LigerPerkernelRun] = []
        for arch in self.archs:
            for kernel in self.kernels:
                out.append(
                    LigerPerkernelRun(
                        kernel=kernel,
                        arch=arch,
                        dtype=self.dtype,
                        namespace=self.namespace,
                        context=self.context,
                        image=self.image,
                        script=self.script,
                        pvc=self.pvc,
                        result_root=self.result_root,
                        rune_bin=self.rune_bin,
                    )
                )
        return out

    def submit(self, *, dry_run: str | None = None, check: bool = True) -> list[RuneSubmitResult]:
        return [run.submit(dry_run=dry_run, check=check) for run in self.runs()]

    def to_commands(self, *, dry_run: str | None = None) -> list[str]:
        return [run.to_command(dry_run=dry_run) for run in self.runs()]
