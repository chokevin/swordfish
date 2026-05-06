"""Thin wrapper around the `rune submit` CLI.

The shell-out shape mirrors `aurora-research/rune-sdk/rune/runner.py` so we can
collapse later if the rune-sdk grows a generic submit class. For now this stays
project-local because the rune-sdk currently targets `rune finetune submit`
(level 3) only.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class RuneCommandError(RuntimeError):
    def __init__(self, args: list[str], returncode: int, stdout: str, stderr: str):
        self.args_list = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"rune command failed ({returncode}): {' '.join(shlex.quote(a) for a in args)}\n"
            f"{stderr.strip() or stdout.strip()}"
        )


class RuneProfileSecurityError(RuntimeError):
    """Raised when the local rune binary drops profile-required pod security."""


@dataclass(frozen=True)
class RuneSubmitResult:
    name: str
    args: list[str]
    rendered_yaml: str | None
    stdout: str
    stderr: str

    @property
    def submitted(self) -> bool:
        """True when the rune submit was real (not a dry-run)."""
        return self.rendered_yaml is None


@dataclass
class RuneSubmit:
    """One rune-submit invocation, programmatic equivalent of the CLI flags."""

    name: str
    preset: str | None = None
    profile: str | None = None
    image: str | None = None
    script: str | Path | None = None
    namespace: str = "ray"
    context: str | None = None
    volumes: list[str] = field(default_factory=list)
    mounts: list[str] = field(default_factory=list)
    after_success: str | None = None
    extra_args: list[str] = field(default_factory=list)
    forwarded_args: list[str] = field(default_factory=list)
    # `env` is the *subprocess* environment for the local `rune` invocation.
    # `container_env` is rendered as repeated `--env KEY=VAL` and ends up on
    # the workload pod. The two are deliberately separate so passing
    # KUBECONFIG locally doesn't accidentally leak into the cluster job.
    env: dict[str, str] = field(default_factory=dict)
    container_env: dict[str, str] = field(default_factory=dict)
    profile_mode: str | None = None  # ncu | nsys | None
    output: str | None = None
    rune_bin: str = "rune"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name is required")
        if not (self.preset or self.profile):
            raise ValueError("either preset or profile must be set")
        if self.preset and self.profile:
            raise ValueError("preset and profile are mutually exclusive")
        if self.script is None:
            raise ValueError("script is required for level-1 submits")
        if self.profile_mode is not None and self.profile_mode not in ("ncu", "nsys"):
            raise ValueError(
                f"profile_mode must be one of ncu|nsys|None; got {self.profile_mode!r}"
            )
        for k in self.container_env:
            if k.startswith("RUNE_") or k.startswith("AIRUN_"):
                raise ValueError(
                    f"container_env key {k!r} uses reserved RUNE_/AIRUN_ namespace; "
                    "rune rejects these on the wire — pick a different name"
                )

    def to_args(self, *, dry_run: str | None = None) -> list[str]:
        """Render the equivalent rune submit argv (without env)."""
        args: list[str] = [self.rune_bin, "submit", self.name]
        if self.preset:
            args += ["--preset", self.preset]
        if self.profile:
            args += ["--profile", self.profile]
        if self.image:
            args += ["--image", self.image]
        if self.script:
            args += ["--script", str(self.script)]
        for v in self.volumes:
            args += ["--volume", v]
        for m in self.mounts:
            args += ["--mount", m]
        # Sort container_env for deterministic argv (test stability and
        # easier diffing of recorded commands).
        for k in sorted(self.container_env):
            args += ["--env", f"{k}={self.container_env[k]}"]
        if self.profile_mode:
            args += ["--profile-mode", self.profile_mode]
        if self.output:
            args += ["--output", self.output]
        if self.after_success:
            args += ["--after-success", self.after_success]
        if self.context:
            args += ["--context", self.context]
        args += ["-n", self.namespace]
        for extra in self.extra_args:
            args.append(extra)
        if dry_run:
            if dry_run not in ("client", "server"):
                raise ValueError("dry_run must be one of: None, 'client', 'server'")
            args += ["--dry-run", dry_run]
        if self.forwarded_args:
            args.append("--")
            args.extend(self.forwarded_args)
        return args

    def to_command(self, *, dry_run: str | None = None) -> str:
        """Human-readable shell form of the submit command."""
        return " ".join(shlex.quote(a) for a in self.to_args(dry_run=dry_run))

    def submit(
        self,
        *,
        dry_run: str | None = None,
        check: bool = True,
        auto_topology_policy: bool = True,
    ) -> RuneSubmitResult:
        """Run rune submit. Returns the rendered YAML on dry_run='client', else None.

        When auto_topology_policy is True (default), the SDK looks for the
        airun-zero azure-topology-policy.yaml in common locations and injects
        $RUNE_TOPOLOGY_POLICY automatically — preset-based dispatch fails
        otherwise. Set False to opt out (e.g. when the env var is set
        intentionally to a non-default policy).
        """
        from swordfish.dispatch.topology import topology_policy_env

        env = {**os.environ, **self.env}
        if auto_topology_policy:
            env.update(topology_policy_env())
        if dry_run is None and _requires_sys_admin_ncu_guard(self):
            self._preflight_sys_admin_ncu(env=env)

        args = self.to_args(dry_run=dry_run)
        proc = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=env,
        )
        if check and proc.returncode != 0:
            raise RuneCommandError(args, proc.returncode, proc.stdout, proc.stderr)
        rendered = proc.stdout if dry_run == "client" else None
        return RuneSubmitResult(
            name=self.name,
            args=args,
            rendered_yaml=rendered,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    def _preflight_sys_admin_ncu(self, *, env: dict[str, str]) -> None:
        args = self.to_args(dry_run="client")
        proc = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            raise RuneCommandError(args, proc.returncode, proc.stdout, proc.stderr)
        rendered = proc.stdout
        if "securityContext:" not in rendered or "SYS_ADMIN" not in rendered:
            raise RuneProfileSecurityError(
                "A100 NCU requires container securityContext.capabilities.add=[SYS_ADMIN], "
                f"but {self.rune_bin} dry-run did not render it for profile {self.profile!r}. "
                "Rebuild/install a Rune binary that supports profile spec.runtime.securityContext "
                "before submitting, otherwise Nsight Compute fails with ERR_NVGPUCTRPERM."
            )


def _requires_sys_admin_ncu_guard(run: RuneSubmit) -> bool:
    return (
        run.profile_mode == "ncu" and run.profile is not None and run.profile.endswith("-a100-ncu")
    )
