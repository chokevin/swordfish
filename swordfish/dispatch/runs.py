"""Typed swordfish benchmark runs that compile to `rune submit` invocations.

The defaults are tuned for the kernel-mode-training queue on voice-agent-flex
and the `voiceagentcr.azurecr.io/airun/swordfish-bench:dev` image (sourced
from `swordfish.dispatch.profiles.IMAGE` to keep the dispatch SDK and the
profile pack in sync). Override per run as needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from swordfish.dispatch.profiles import IMAGE as PROFILE_PACK_IMAGE
from swordfish.dispatch.results import FetchedResult
from swordfish.dispatch.rune import RuneSubmit, RuneSubmitResult


# Defaults — change these in one place, not at every call site.
DEFAULT_NAMESPACE = "ray"
DEFAULT_IMAGE = PROFILE_PACK_IMAGE
DEFAULT_PVC = "training-nfs"
DEFAULT_PRESET = "azure.kernel-mode.training.l"
DEFAULT_BENCH_SCRIPT = Path("infra/rune/scripts/swordfish-bench.sh")
DEFAULT_RESULT_ROOT = "/data/swordfish/week1"

# Profile pack defined in swordfish/dispatch/profiles.py and rendered to
# infra/rune/profiles/swordfish-pack.yaml. Dispatching by profile (rather than
# by raw preset) is what makes the YAML pack the source of truth: edit
# profiles.py, regenerate, and `make rune-submit-*` automatically picks up the
# new image / queue / persistence shape.
SWORDFISH_PROFILE_PREFIX = "swordfish-bench-"


def default_profile_for(arch: str) -> str:
    return f"{SWORDFISH_PROFILE_PREFIX}{arch}"


ARCH_TO_PRESET = {
    "a100": "azure.kernel-mode.training.l",
    "h100": "azure.kernel-mode.training.l",  # same lane today; H100 in kernel-mode pending
    "h200": "azure.kernel-mode.large-memory.xl",
}

# Per-arch rune `--gpu-class` injected into every submit. Without this, rune's
# topology planner picks h200-nvlink-141gb by default for any lane=training/
# large-memory profile, and Kueue then routes the pod to whichever GPU node
# has a free DRA claim — usually wrong (e.g. an A100-targeted job lands on
# H200). The values match the `rune.ai/gpu-class` node labels on
# voice-agent-flex.
ARCH_TO_GPU_CLASS = {
    "a100": "a100-nvlink-80gb",
    "h100": "h100-standalone-95gb",
    "h200": "h200-nvlink-141gb",
}


def _inject_gpu_class(arch: str, extra_args: Iterable[str]) -> list[str]:
    """Prepend `--gpu-class <arch class>` unless the caller already set one."""
    out = list(extra_args)
    if "--gpu-class" in out:
        return out
    return ["--gpu-class", ARCH_TO_GPU_CLASS[arch], *out]


LIGER_KERNELS = ("rmsnorm", "swiglu", "rope", "fused_linear_ce")
LIGER_KERNELS_IMPLEMENTED = ("rmsnorm", "swiglu")
# ncu/nsys are external binary wrappers handled by `rune --profile-mode`.
# 'torch' is in-process (torch.profiler) and bypasses rune's wrapper —
# the dispatch SDK injects SWORDFISH_PROFILE/SWORDFISH_PROFILE_OUT env
# vars and the bench main wraps itself via swordfish.runner.profile_torch.
PROFILE_MODES = ("ncu", "nsys", "torch")
RUNE_NATIVE_PROFILE_MODES = ("ncu", "nsys")  # subset rune knows about
# Rune's --profile-mode=ncu produces an .ncu-rep binary report (NOT the .ncu.csv
# format the legacy SWORDFISH_PROFILE script-side path produces). Downstream
# tooling that expects CSV needs to call `ncu --import` to convert.
PROFILE_EXTENSIONS = {"ncu": "ncu-rep", "nsys": "nsys-rep", "torch": "json"}

_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def _profile_out_dir_for(name: str) -> str:
    """Where rune's renderer writes profile artifacts (matches LigerPerkernelRun)."""
    return f"/data/{name}/profile"


def _profile_out_path_for(name: str, profile_mode: str) -> str:
    return f"{_profile_out_dir_for(name)}/profile.{PROFILE_EXTENSIONS[profile_mode]}"


def _resolve_torch_profile(
    profile_mode: str | None,
    resolved_name: str,
    container_env: dict[str, str],
) -> tuple[str | None, dict[str, str]]:
    """Inject SWORDFISH_PROFILE env vars when profile_mode == 'torch'.

    Returns (rune_native_profile_mode, container_env). For torch mode,
    rune_native_profile_mode is None (rune doesn't know about 'torch';
    the bench main wraps itself in-process). For ncu/nsys, the env is
    untouched and rune handles the wrapping.

    container_env is copied; this function does not mutate the caller's
    dict.
    """
    env = dict(container_env)
    if profile_mode != "torch":
        return profile_mode, env
    env.setdefault("SWORDFISH_PROFILE", "torch")
    env.setdefault(
        "SWORDFISH_PROFILE_OUT", _profile_out_path_for(resolved_name, "torch")
    )
    return None, env


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
    container_env: dict[str, str] = field(default_factory=dict)
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
    def resolved_profile(self) -> str | None:
        """Which rune profile to submit with.

        Defaults to the swordfish-bench pack profile for this arch so the
        committed `infra/rune/profiles/swordfish-pack.yaml` (regenerated from
        `swordfish.dispatch.profiles`) is what actually drives image, queue,
        and persistence on every submit. Callers can opt back to the raw
        preset shortcut by setting `preset=...` explicitly.
        """
        if self.profile:
            return self.profile
        if self.preset:
            return None
        return default_profile_for(self.arch)

    @property
    def out_path(self) -> str:
        return f"{self.result_root}/liger-perkernel/{self.kernel}-{self.arch}.json"

    @property
    def profile_out_dir(self) -> str | None:
        """The PVC directory rune writes the profile artifact into.

        rune's renderer hardcodes `/data/<job-name>/profile/profile.<ext>` (see
        applications/rune/internal/submit/render.go:applyProfileMode). Returns
        the directory for use as `rune submit get --path ...`.
        """
        if not self.profile_mode:
            return None
        # If result_root is ever overridden away from /data/..., the profile
        # artifact still lands under /data/<name>/ because rune uses
        # storage.DurableRoot, not the --output path. The profile is fetched
        # via explicit --path so this divergence is fine.
        return f"/data/{self.resolved_name}/profile"

    @property
    def profile_out_path(self) -> str | None:
        """The full PVC path of the profile artifact (`<dir>/profile.<ext>`)."""
        if not self.profile_mode:
            return None
        ext = PROFILE_EXTENSIONS[self.profile_mode]
        return f"{self.profile_out_dir}/profile.{ext}"

    @property
    def profile_out_artifact(self) -> str | None:
        """The artifact filename inside `profile_out_dir`, for `--artifact NAME`."""
        if not self.profile_mode:
            return None
        ext = PROFILE_EXTENSIONS[self.profile_mode]
        return f"profile.{ext}"

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
        rune_native_mode, container_env = _resolve_torch_profile(
            self.profile_mode, self.resolved_name, self.container_env
        )
        kwargs: dict = dict(
            name=self.resolved_name,
            image=self.image,
            script=self.script,
            namespace=self.namespace,
            context=self.context,
            volumes=[f"data=pvc:{self.pvc}"],
            extra_args=_inject_gpu_class(self.arch, self.extra_args),
            forwarded_args=self.forwarded_args,
            container_env=container_env,
            rune_bin=self.rune_bin,
            profile_mode=rune_native_mode,
            output=self.out_path,
        )
        resolved_profile = self.resolved_profile
        if resolved_profile:
            kwargs["profile"] = resolved_profile
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
        rune_bin: str | None = None,
        include_traces: bool = False,
    ) -> FetchedResult:
        """Copy this run's result JSON back from the cluster PVC.

        Default path: shells out to `rune submit get NAME -o raw`, which uses
        the `airun.aks.io/result-{path,pvc}` annotations recorded by
        `rune submit --output ...`. Falls back to `kubectl cp` only when:

        1. rune reports the Job has no result-path annotation (legacy jobs
           submitted before swordfish was migrated to `--output`), OR
        2. the caller explicitly passes `pod=` or `pod_label_selector=`
           (debugging / non-rune-managed pods).

        All other failures (auth, missing artifact, helper-pod failure)
        propagate as-is — silent kubectl-cp fallback would mask real bugs.

        With include_traces=True, also fetches the profiler artifact
        (`.ncu-rep` for profile_mode='ncu', `.nsys-rep` for 'nsys'). The
        trace lives at rune's hardcoded `/data/<name>/profile/profile.<ext>`
        (NOT at the recorded result-path), so it's fetched via explicit
        `--path/--pvc/--artifact` overrides. Local target keeps the
        kernel/arch-named filename so downstream tooling stays consistent.
        """
        from swordfish.dispatch.results import (
            RuneSubmitGetMissingAnnotationsError,
            fetch_result as _kubectl_cp_fetch,
            fetch_via_rune_submit_get,
        )

        target = (
            Path(local_path)
            if local_path
            else Path("runs/rune/week1") / Path(self.out_path).relative_to("/data/swordfish/week1")
        )
        target.parent.mkdir(parents=True, exist_ok=True)

        rune = rune_bin or self.rune_bin
        explicit_pod_override = pod is not None or pod_label_selector is not None

        if explicit_pod_override:
            # Caller wants a specific pod (debugging, scratch helpers etc.).
            # Skip rune-submit-get entirely.
            result = _kubectl_cp_fetch(
                job_name=self.resolved_name,
                remote_path=self.out_path,
                local_path=target,
                namespace=self.namespace,
                context=self.context,
                pod=pod,
                pod_label_selector=pod_label_selector,
                kubectl_bin=kubectl_bin,
            )
        else:
            try:
                payload = fetch_via_rune_submit_get(
                    name=self.resolved_name,
                    namespace=self.namespace,
                    context=self.context,
                    rune_bin=rune,
                )
            except RuneSubmitGetMissingAnnotationsError:
                # Legacy job (pre-`--output` migration). Fall back to kubectl-cp.
                result = _kubectl_cp_fetch(
                    job_name=self.resolved_name,
                    remote_path=self.out_path,
                    local_path=target,
                    namespace=self.namespace,
                    context=self.context,
                    kubectl_bin=kubectl_bin,
                )
            else:
                target.write_bytes(payload)
                result = FetchedResult(
                    name=self.resolved_name,
                    pod="<rune submit get>",
                    remote_path=self.out_path,
                    local_path=target,
                )

        if include_traces and self.profile_mode and self.profile_out_dir:
            ext = PROFILE_EXTENSIONS[self.profile_mode]
            trace_local = target.with_suffix(f".{ext}")
            trace_payload = fetch_via_rune_submit_get(
                name=self.resolved_name,
                namespace=self.namespace,
                context=self.context,
                path=self.profile_out_dir,
                pvc=self.pvc,
                artifact=self.profile_out_artifact,
                rune_bin=rune,
            )
            trace_local.write_bytes(trace_payload)

        return result


@dataclass
class TorchGemmRun:
    """One torch/cuBLAS GEMM benchmark on a single arch.

    Programmatic equivalent of `make rune-submit-gemm-<arch>`. The shell-out
    target shape is identical (script-mode, kernel-mode-training queue,
    `--output` annotations); this class collapses the argv-construction into
    typed fields.
    """

    arch: str = "a100"
    backend: str = "torch"
    m: int = 4096
    n: int = 4096
    k: int = 4096
    dtype: str = "fp16"
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
    preset: str | None = None
    profile: str | None = None
    extra_args: list[str] = field(default_factory=list)
    container_env: dict[str, str] = field(default_factory=dict)
    rune_bin: str = "rune"
    profile_mode: str | None = None  # ncu | nsys | None

    def __post_init__(self) -> None:
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
        return _normalize_name(self.name or f"sf-gemm-{self.backend}-{self.arch}")

    @property
    def resolved_preset(self) -> str:
        if self.profile:
            return ""
        return self.preset or ARCH_TO_PRESET[self.arch]

    @property
    def resolved_profile(self) -> str | None:
        if self.profile:
            return self.profile
        if self.preset:
            return None
        return default_profile_for(self.arch)

    @property
    def out_path(self) -> str:
        return f"{self.result_root}/{self.backend}-gemm-{self.arch}.json"

    @property
    def forwarded_args(self) -> list[str]:
        return [
            "run-gemm",
            "--backend",
            self.backend,
            "--m",
            str(self.m),
            "--n",
            str(self.n),
            "--k",
            str(self.k),
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
        rune_native_mode, container_env = _resolve_torch_profile(
            self.profile_mode, self.resolved_name, self.container_env
        )
        kwargs: dict = dict(
            name=self.resolved_name,
            image=self.image,
            script=self.script,
            namespace=self.namespace,
            context=self.context,
            volumes=[f"data=pvc:{self.pvc}"],
            extra_args=_inject_gpu_class(self.arch, self.extra_args),
            forwarded_args=self.forwarded_args,
            container_env=container_env,
            rune_bin=self.rune_bin,
            profile_mode=rune_native_mode,
            output=self.out_path,
        )
        resolved_profile = self.resolved_profile
        if resolved_profile:
            kwargs["profile"] = resolved_profile
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
