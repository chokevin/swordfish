"""Resolve RUNE_TOPOLOGY_POLICY for the preset-based dispatch path.

V0/V1 rune resolves preset names against a topology policy YAML that lives in
the airun-zero repo. The path must be on $RUNE_TOPOLOGY_POLICY when `rune
submit --preset ...` is invoked, otherwise rune errors with:

    topology policy not found; set RUNE_TOPOLOGY_POLICY or run from repo root

Researchers shouldn't have to hand-set this every session. This module finds
the file by walking common locations:

1. The env var, if already set and the file exists.
2. A few well-known paths under $HOME (where users typically check out
   aks-ai-runtime / ai2 / airun).
3. The current working directory's ancestors (handles "run from repo root").

Returns None if nothing is found; the caller should error with a helpful
message rather than silently submitting and getting rejected by rune.
"""

from __future__ import annotations

import os
from pathlib import Path

POLICY_BASENAMES = ("azure-topology-policy.yaml",)
RELATIVE_HINTS = (
    "applications/airun-zero/platform/azure-topology-policy.yaml",
    "applications/airun-zero/topology/azure-topology-policy.yaml",
    "platform/azure-topology-policy.yaml",
    "azure-topology-policy.yaml",
)
HOME_HINTS = (
    "~/dev/ai2",
    "~/dev/aks-ai-runtime",
    "~/dev/airun",
    "~/dev/airun-zero",
    "~/src/aks-ai-runtime",
    "~/code/aks-ai-runtime",
)


def find_topology_policy() -> Path | None:
    """Return a path to azure-topology-policy.yaml or None if not found."""
    # 1. env var wins
    env = os.environ.get("RUNE_TOPOLOGY_POLICY")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p

    # 2. well-known checkout locations
    for hint in HOME_HINTS:
        base = Path(hint).expanduser()
        if not base.is_dir():
            continue
        for rel in RELATIVE_HINTS:
            p = base / rel
            if p.is_file():
                return p

    # 3. ancestor walk from cwd
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        for rel in RELATIVE_HINTS:
            p = parent / rel
            if p.is_file():
                return p

    return None


def topology_policy_env() -> dict[str, str]:
    """Build an env dict that injects RUNE_TOPOLOGY_POLICY when discoverable.

    Returns {} if the env var is already set or the file isn't found. The
    caller should layer this on top of os.environ before invoking rune.
    """
    if os.environ.get("RUNE_TOPOLOGY_POLICY"):
        return {}
    found = find_topology_policy()
    if found is None:
        return {}
    return {"RUNE_TOPOLOGY_POLICY": str(found)}
