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
    env: dict[str, str] = field(default_factory=dict)
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

        args = self.to_args(dry_run=dry_run)
        env = {**os.environ, **self.env}
        if auto_topology_policy:
            env.update(topology_policy_env())
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
