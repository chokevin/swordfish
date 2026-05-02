"""Project-local dispatch SDK for swordfish benchmarks.

Wraps the verbose `rune submit ...` CLI invocation in typed dataclasses so a
researcher writes ~3 lines of Python instead of an 11-line shell command.

Pattern mirrors `aurora-research/rune-sdk`:
- typed dataclasses describe the run
- `.submit()` shells out to the `rune` CLI
- dry-run / wait / follow are flags on `.submit()`

This is a level-1 (submit-only) wrapper. When swordfish promotes to a level-2
managed eval (`rune eval --harness swordfish-bench`), this module's API stays
the same and the implementation switches to `rune eval` under the hood.
"""

from __future__ import annotations

from swordfish.dispatch.image import (
    ImageBuildError,
    build_and_push_dev_image,
)
from swordfish.dispatch.results import (
    FetchedResult,
    FetchedRunArtifacts,
    ResultFetchError,
    RuneSubmitGetMissingAnnotationsError,
    fetch_result,
    fetch_run_artifacts,
    fetch_via_rune_submit_get,
)
from swordfish.dispatch.rune import (
    RuneCommandError,
    RuneSubmit,
    RuneSubmitResult,
)
from swordfish.dispatch.runs import (
    DEFAULT_IMAGE,
    DEFAULT_NAMESPACE,
    DEFAULT_PRESET,
    DEFAULT_PVC,
    LigerPerkernelMatrix,
    LigerPerkernelRun,
    TorchGemmRun,
)
from swordfish.dispatch.topology import (
    find_topology_policy,
    topology_policy_env,
)

__all__ = [
    "DEFAULT_IMAGE",
    "DEFAULT_NAMESPACE",
    "DEFAULT_PRESET",
    "DEFAULT_PVC",
    "FetchedResult",
    "FetchedRunArtifacts",
    "ImageBuildError",
    "LigerPerkernelMatrix",
    "LigerPerkernelRun",
    "ResultFetchError",
    "RuneCommandError",
    "RuneSubmit",
    "RuneSubmitGetMissingAnnotationsError",
    "RuneSubmitResult",
    "TorchGemmRun",
    "build_and_push_dev_image",
    "fetch_result",
    "fetch_run_artifacts",
    "fetch_via_rune_submit_get",
    "find_topology_policy",
    "topology_policy_env",
]
