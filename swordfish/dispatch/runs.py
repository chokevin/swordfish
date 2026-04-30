"""Typed swordfish benchmark runs that compile to `rune submit` invocations.

The defaults are tuned for the kernel-mode-training queue on voice-agent-flex
and the `ghcr.io/chokevin/swordfish-bench:latest` image. Override per run as
needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from swordfish.dispatch.rune import RuneSubmit, RuneSubmitResult


# Defaults — change these in one place, not at every call site.
DEFAULT_NAMESPACE = "ray"
DEFAULT_IMAGE = "ghcr.io/chokevin/swordfish-bench:latest"
DEFAULT_PVC = "training-nfs"
DEFAULT_PRESET = "azure.kernel-mode.training.l"
DEFAULT_BENCH_SCRIPT = Path("infra/rune/scripts/swordfish-bench.sh")
DEFAULT_RESULT_ROOT = "/data/swordfish/week1"

ARCH_TO_PRESET = {
    "a100": "azure.kernel-mode.training.l",
    "h100": "azure.kernel-mode.training.l",  # same lane today; H100 in kernel-mode pending
    "h200": "azure.kernel-mode.large-memory.xl",
}

LIGER_KERNELS = ("rmsnorm", "swiglu", "rope", "fused_linear_ce")
LIGER_KERNELS_IMPLEMENTED = ("rmsnorm", "swiglu")

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

    def to_rune_submit(self) -> RuneSubmit:
        kwargs: dict = dict(
            name=self.resolved_name,
            image=self.image,
            script=self.script,
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

    def submit(self, *, dry_run: str | None = None, check: bool = True) -> RuneSubmitResult:
        return self.to_rune_submit().submit(dry_run=dry_run, check=check)


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
