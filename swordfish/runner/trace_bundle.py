"""Fetch profiled Rune jobs into a stable trace handoff bundle."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from swordfish.dispatch import FetchedRunArtifacts, fetch_run_artifacts

PROFILE_EXTENSIONS = {"ncu": "ncu-rep", "nsys": "nsys-rep", "torch": "json"}


@dataclass(frozen=True)
class TraceJobSpec:
    name: str
    profile_mode: str | None


@dataclass(frozen=True)
class TraceBundleResult:
    bundle_name: str
    bundle_dir: Path
    manifest_path: Path
    archive_path: Path | None
    jobs: tuple[TraceJobSpec, ...]


def parse_trace_job_spec(raw: str, *, default_profile_mode: str | None = None) -> TraceJobSpec:
    """Parse ``NAME[:PROFILE_MODE]`` for trace handoff commands."""
    if ":" in raw:
        name, mode = raw.rsplit(":", 1)
        profile_mode = mode or None
    else:
        name = raw
        profile_mode = default_profile_mode
    name = name.strip()
    if not name:
        raise ValueError("trace job name cannot be empty")
    if profile_mode is not None and profile_mode not in PROFILE_EXTENSIONS:
        allowed = ", ".join(sorted(PROFILE_EXTENSIONS))
        raise ValueError(f"profile mode {profile_mode!r} not in: {allowed}")
    return TraceJobSpec(name=name, profile_mode=profile_mode)


def bundle_traces(
    jobs: list[TraceJobSpec] | tuple[TraceJobSpec, ...],
    *,
    bundle_name: str | None = None,
    local_root: Path | str = Path("runs/traces"),
    namespace: str = "ray",
    context: str | None = None,
    pvc: str = "training-nfs",
    rune_bin: str = "rune",
    overwrite: bool = False,
    create_archive: bool = True,
) -> TraceBundleResult:
    """Fetch result JSON + traces for jobs and write a portable tarball.

    Local layout is intentionally stable so either a human or Hermes can pick it
    up without knowing Rune's PVC internals:

    ``runs/traces/<bundle>/<job>/<job>.json``
    ``runs/traces/<bundle>/<job>/<job>.{ncu-rep|nsys-rep|json}``
    ``runs/traces/<bundle>/manifest.json``
    ``runs/traces/<bundle>.tar.gz``
    """
    if not jobs:
        raise ValueError("at least one job is required")

    root = Path(local_root)
    resolved_name = bundle_name or f"trace-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    bundle_dir = root / resolved_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    fetched_jobs: list[dict[str, object]] = []
    specs = tuple(jobs)
    for spec in specs:
        fetched = fetch_run_artifacts(
            name=spec.name,
            profile_mode=spec.profile_mode,
            local_dir=bundle_dir / spec.name,
            namespace=namespace,
            context=context,
            pvc=pvc,
            rune_bin=rune_bin,
            overwrite=overwrite,
        )
        fetched_jobs.append(_manifest_entry(fetched, bundle_dir=bundle_dir, pvc=pvc))

    manifest = {
        "schema_version": "swordfish.trace-bundle.v1",
        "bundle_name": resolved_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "namespace": namespace,
        "context": context,
        "pvc": pvc,
        "jobs": fetched_jobs,
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    archive_path = None
    if create_archive:
        archive_path = root / f"{resolved_name}.tar.gz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, mode="w:gz") as archive:
            archive.add(bundle_dir, arcname=resolved_name)

    return TraceBundleResult(
        bundle_name=resolved_name,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        archive_path=archive_path,
        jobs=specs,
    )


def _manifest_entry(
    fetched: FetchedRunArtifacts,
    *,
    bundle_dir: Path,
    pvc: str,
) -> dict[str, object]:
    files = [_file_entry(fetched.result_json, bundle_dir)]
    remote_profile_path = None
    if fetched.profile_artifact is not None and fetched.profile_mode is not None:
        files.append(_file_entry(fetched.profile_artifact, bundle_dir))
        ext = PROFILE_EXTENSIONS[fetched.profile_mode]
        remote_profile_path = f"/data/{fetched.name}/profile/profile.{ext}"
    return {
        "name": fetched.name,
        "profile_mode": fetched.profile_mode,
        "result_json": str(fetched.result_json.relative_to(bundle_dir)),
        "profile_artifact": (
            str(fetched.profile_artifact.relative_to(bundle_dir))
            if fetched.profile_artifact is not None
            else None
        ),
        "remote_profile_path": remote_profile_path,
        "remote_profile_pvc": pvc if remote_profile_path else None,
        "files": files,
    }


def _file_entry(path: Path, bundle_dir: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(bundle_dir)),
        "bytes": path.stat().st_size,
    }
