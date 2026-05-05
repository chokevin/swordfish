"""Researcher-facing experiment registry grounded in generated Rune profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from swordfish.dispatch.profiles import (
    ARCHES,
    ARCH_TO_GPU_PRODUCT,
    QUEUE_CLUSTER,
    ProfileSpec,
    all_profiles,
)
from swordfish.dispatch.runs import (
    ARCH_TO_GPU_CLASS,
    LigerFsdpRun,
    LigerPerkernelRun,
    TorchGemmRun,
    VectorSumRun,
    default_fsdp_profile_for,
    default_profile_for,
)

ExperimentWorkload = Literal[
    "gemm",
    "vectorsum-v2",
    "liger-rmsnorm",
    "liger-swiglu",
    "liger-fsdp",
]
ProfileFamily = Literal["bench", "fsdp"]
ExperimentRun = TorchGemmRun | VectorSumRun | LigerPerkernelRun | LigerFsdpRun

COMMON_RUN_OVERRIDES = {"name", "profile_mode", "result_root", "script", "context", "image"}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    workload: ExperimentWorkload
    profile_family: ProfileFamily
    allowed_arches: tuple[str, ...]
    defaults: Mapping[str, object]
    description: str


@dataclass(frozen=True)
class QueueSummary:
    profile: str
    profile_family: ProfileFamily
    arch: str
    lane: str
    cluster_queue: str
    local_queue: str
    gpu_class: str
    gpu_product: str
    gpu_size: str
    gpus_per_node: int
    cpu_request: str
    memory_request: str
    claim_template: str


@dataclass(frozen=True)
class ResolvedExperiment:
    spec: ExperimentSpec
    arch: str
    profile: str
    queue_summary: QueueSummary


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "gemm": ExperimentSpec(
        name="gemm",
        workload="gemm",
        profile_family="bench",
        allowed_arches=ARCHES,
        defaults={
            "backend": "torch",
            "dtype": "fp16",
            "m": 4096,
            "n": 4096,
            "k": 4096,
            "repeats": 5,
            "warmup": 10,
            "iters": 50,
        },
        description="One-GPU torch/cuBLAS GEMM baseline.",
    ),
    "vectorsum-v2": ExperimentSpec(
        name="vectorsum-v2",
        workload="vectorsum-v2",
        profile_family="bench",
        allowed_arches=ARCHES,
        defaults={
            "backend": "triton",
            "size": 1_638_400,
            "dtype": "fp32",
            "repeats": 5,
            "warmup": 10,
            "iters": 50,
            "block_size": 8192,
        },
        description="One-GPU vector sum reduction target.",
    ),
    "liger-rmsnorm": ExperimentSpec(
        name="liger-rmsnorm",
        workload="liger-rmsnorm",
        profile_family="bench",
        allowed_arches=ARCHES,
        defaults={
            "dtype": "bf16",
            "repeats": 5,
            "warmup": 10,
            "iters": 50,
        },
        description="One-GPU paired baseline-vs-Liger RMSNorm microbenchmark.",
    ),
    "liger-swiglu": ExperimentSpec(
        name="liger-swiglu",
        workload="liger-swiglu",
        profile_family="bench",
        allowed_arches=ARCHES,
        defaults={
            "dtype": "bf16",
            "repeats": 5,
            "warmup": 10,
            "iters": 50,
        },
        description="One-GPU paired baseline-vs-Liger SwiGLU microbenchmark.",
    ),
    "liger-fsdp": ExperimentSpec(
        name="liger-fsdp",
        workload="liger-fsdp",
        profile_family="fsdp",
        allowed_arches=ARCHES,
        defaults={
            "mode": "baseline",
            "model_source": "transformers",
            "model_preset": "llama3-8b",
            "micro_batch_size": 1,
            "seq_len": 2048,
            "dtype": "bf16",
            "repeats": 3,
            "warmup": 1,
            "iters": 5,
            "nproc_per_node": 8,
            "gradient_checkpointing": True,
            "profile_steady_state": False,
            "fsdp_wrap_policy": "root",
            "fsdp_backward_prefetch": "default",
            "fsdp_forward_prefetch": False,
            "fsdp_limit_all_gathers": True,
        },
        description="8-GPU Llama train-step reproduction row for baseline or Liger FSDP.",
    ),
}


def list_experiments() -> list[ExperimentSpec]:
    return [EXPERIMENTS[name] for name in sorted(EXPERIMENTS)]


def get_experiment(name: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"unknown experiment {name!r}; expected one of: {available}") from exc


def profile_for_family(profile_family: ProfileFamily, arch: str) -> str:
    if profile_family == "bench":
        return default_profile_for(arch)
    if profile_family == "fsdp":
        return default_fsdp_profile_for(arch)
    raise ValueError(f"unknown profile family {profile_family!r}")


def _profile_spec_for(profile: str) -> ProfileSpec:
    profiles = {p.name: p for p in all_profiles()}
    try:
        return profiles[profile]
    except KeyError as exc:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"profile {profile!r} is not in the generated pack: {available}") from exc


def resolve_experiment(name: str, arch: str) -> ResolvedExperiment:
    spec = get_experiment(name)
    if arch not in spec.allowed_arches:
        allowed = ", ".join(spec.allowed_arches)
        raise ValueError(f"arch {arch!r} is not allowed for {name!r}; expected one of: {allowed}")

    profile = profile_for_family(spec.profile_family, arch)
    profile_spec = _profile_spec_for(profile)
    queue = QueueSummary(
        profile=profile,
        profile_family=spec.profile_family,
        arch=arch,
        lane=profile_spec.lane,
        cluster_queue=QUEUE_CLUSTER,
        local_queue=profile_spec.local_queue,
        gpu_class=ARCH_TO_GPU_CLASS[arch],
        gpu_product=ARCH_TO_GPU_PRODUCT[arch],
        gpu_size=profile_spec.gpu_size,
        gpus_per_node=profile_spec.gpus_per_node,
        cpu_request=profile_spec.cpu_request,
        memory_request=profile_spec.memory_request,
        claim_template=profile_spec.claim_template,
    )
    return ResolvedExperiment(spec=spec, arch=arch, profile=profile, queue_summary=queue)


def _merge_overrides(
    spec: ExperimentSpec, overrides: Mapping[str, object] | None
) -> dict[str, Any]:
    values: dict[str, Any] = dict(spec.defaults)
    if not overrides:
        return values
    allowed = set(spec.defaults) | COMMON_RUN_OVERRIDES
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"overrides not valid for experiment {spec.name!r}: {', '.join(unknown)}")
    values.update(overrides)
    return values


def build_run_for_experiment(
    name: str,
    arch: str,
    overrides: Mapping[str, object] | None = None,
) -> ExperimentRun:
    resolved = resolve_experiment(name, arch)
    spec = resolved.spec
    values = _merge_overrides(spec, overrides)
    common = {
        "name": values.get("name"),
        "profile_mode": values.get("profile_mode"),
        "result_root": values.get("result_root"),
        "script": values.get("script"),
        "context": values.get("context"),
        "image": values.get("image"),
        "profile": resolved.profile,
    }
    common = {k: v for k, v in common.items() if v is not None}

    if spec.workload == "gemm":
        return TorchGemmRun(
            arch=arch,
            backend=str(values["backend"]),
            m=int(values["m"]),
            n=int(values["n"]),
            k=int(values["k"]),
            dtype=str(values["dtype"]),
            repeats=int(values["repeats"]),
            warmup=int(values["warmup"]),
            iters=int(values["iters"]),
            **common,
        )

    if spec.workload == "vectorsum-v2":
        return VectorSumRun(
            arch=arch,
            backend=str(values["backend"]),
            size=int(values["size"]),
            dtype=str(values["dtype"]),
            repeats=int(values["repeats"]),
            warmup=int(values["warmup"]),
            iters=int(values["iters"]),
            block_size=int(values["block_size"]),
            **common,
        )

    if spec.workload in {"liger-rmsnorm", "liger-swiglu"}:
        kernel = spec.workload.removeprefix("liger-")
        return LigerPerkernelRun(
            kernel=kernel,
            arch=arch,
            dtype=str(values["dtype"]),
            repeats=int(values["repeats"]),
            warmup=int(values["warmup"]),
            iters=int(values["iters"]),
            **common,
        )

    if spec.workload == "liger-fsdp":
        return LigerFsdpRun(
            arch=arch,
            mode=str(values["mode"]),
            model_source=str(values["model_source"]),
            model_preset=str(values["model_preset"]),
            micro_batch_size=int(values["micro_batch_size"]),
            seq_len=int(values["seq_len"]),
            dtype=str(values["dtype"]),
            repeats=int(values["repeats"]),
            warmup=int(values["warmup"]),
            iters=int(values["iters"]),
            nproc_per_node=int(values["nproc_per_node"]),
            gradient_checkpointing=bool(values["gradient_checkpointing"]),
            profile_steady_state=bool(values["profile_steady_state"]),
            fsdp_wrap_policy=str(values["fsdp_wrap_policy"]),
            fsdp_backward_prefetch=str(values["fsdp_backward_prefetch"]),
            fsdp_forward_prefetch=bool(values["fsdp_forward_prefetch"]),
            fsdp_limit_all_gathers=bool(values["fsdp_limit_all_gathers"]),
            **common,
        )

    raise ValueError(f"unsupported experiment workload {spec.workload!r}")


def format_experiment_table() -> str:
    lines = ["experiment        workload         profile-family  arches       description"]
    for spec in list_experiments():
        lines.append(
            f"{spec.name:<17} {spec.workload:<16} {spec.profile_family:<15} "
            f"{','.join(spec.allowed_arches):<12} {spec.description}"
        )
    return "\n".join(lines)


def format_experiment_explain(name: str, arch: str) -> str:
    resolved = resolve_experiment(name, arch)
    spec = resolved.spec
    queue = resolved.queue_summary
    default_lines = [f"  {key}: {spec.defaults[key]}" for key in sorted(spec.defaults)]
    return "\n".join(
        [
            f"experiment:     {spec.name}",
            f"description:    {spec.description}",
            f"workload:       {spec.workload}",
            f"arch:           {arch}",
            f"profile:        {queue.profile}",
            f"profile_family: {queue.profile_family}",
            f"queue:          {queue.cluster_queue}/{queue.local_queue}",
            f"lane:           {queue.lane}",
            f"gpu:            {queue.gpu_class} ({queue.gpu_product})",
            f"shape:          gpu.size={queue.gpu_size}, gpusPerNode={queue.gpus_per_node}, claimTemplate={queue.claim_template}, cpu={queue.cpu_request}, memory={queue.memory_request}",
            "defaults:",
            *default_lines,
        ]
    )
