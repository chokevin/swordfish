"""Build the swordfish-bench image locally and push to GHCR.

Wraps `infra/rune/image/build.sh PUSH=1`. Returns the resulting full image
ref (registry/name:tag) so the caller can pass it to `LigerPerkernelRun`.

Image rebuild is fast (~10s after first build) because the swordfish/ COPY
layer is the only one that invalidates on Python edits — base image, apt
deps, liger-kernel, and uv all sit above it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class ImageBuildError(RuntimeError):
    pass


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "swordfish").is_dir():
            return parent
    raise ImageBuildError(f"could not locate swordfish repo root from {here}")


def build_and_push_dev_image(
    *,
    push: bool = True,
    container_cmd: str = "podman",
    platform: str | None = None,
    liger_version: str | None = None,
    tag: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Build and (optionally) push the swordfish-bench image. Returns the full ref.

    The script auto-derives a `dev-<sha>[-dirty]` tag from `git describe`. With
    push=False, the image is only built locally (useful when the cluster can
    pull from a local registry); with push=True, it goes to GHCR.
    """
    root = _repo_root()
    script = root / "infra" / "rune" / "image" / "build.sh"
    if not script.is_file():
        raise ImageBuildError(f"build script not found: {script}")

    env = {**os.environ}
    env["CONTAINER_CMD"] = container_cmd
    env["PUSH"] = "1" if push else "0"
    if platform:
        env["PLATFORM"] = platform
    if liger_version:
        env["LIGER_VERSION"] = liger_version
    if tag:
        env["TAG"] = tag
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise ImageBuildError(
            f"build.sh failed ({proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    # build.sh prints the full pushed tag as the last stdout line when PUSH=1.
    # When PUSH=0 we synthesize the tag from the git SHA the same way the
    # script does so the caller can still reference the local image.
    if push:
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("ghcr.io/") or "/" in line and ":" in line:
                return line
        raise ImageBuildError(f"build.sh succeeded but did not print a pushed tag:\n{proc.stdout}")

    sha_proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    sha = sha_proc.stdout.strip()
    dirty_proc = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet"],
        check=False,
    )
    dirty = "-dirty" if dirty_proc.returncode != 0 else ""
    resolved_tag = tag or f"dev-{sha}{dirty}"
    return f"ghcr.io/chokevin/swordfish-bench:{resolved_tag}"
