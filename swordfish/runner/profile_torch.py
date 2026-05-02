"""In-process `torch.profiler` wrapping for swordfish bench mains.

Why a separate module: nsys / ncu are external CLI wrappers (rune handles
them via `--profile-mode ncu|nsys`, no Python code change). `torch.profiler`
is a Python context manager that has to live inside the bench process, so
we expose a single helper that the bench mains in `swordfish.runner.cli`
wrap their call in. The result is a Chrome trace JSON (Perfetto-loadable)
at the path the dispatch SDK pre-computed.

Env-var contract (set by `swordfish.dispatch.runs` for `profile_mode='torch'`):

    SWORDFISH_PROFILE       = "torch"   # opts in
    SWORDFISH_PROFILE_OUT   = "/data/<job-name>/profile/profile.json"

The dispatch layer also tells `infra/rune/scripts/swordfish-bench.sh` to
mkdir the parent dir before exec'ing python, so by the time we get here
the path is writable.

Falls back to a no-op context manager when:
  - SWORDFISH_PROFILE is not "torch" (most invocations)
  - torch.profiler is unavailable (e.g. CPU-only smoke envs without CUDA)

Activities: CPU + (CUDA if available). record_shapes is on for op-shape
breakdown in Perfetto. with_stack is off — Python stack capture multiplies
profile size 5-10x and we don't need source attribution for kernel-mode
benchmarks (NCU is the right tool for that).
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator


SWORDFISH_PROFILE_ENV = "SWORDFISH_PROFILE"
SWORDFISH_PROFILE_OUT_ENV = "SWORDFISH_PROFILE_OUT"
TORCH_PROFILE_MODE = "torch"


def resolve_torch_profile_out() -> Path | None:
    """Return the chrome-trace output path when SWORDFISH_PROFILE=torch.

    Returns None when torch profiling is not requested. Raises ValueError
    when SWORDFISH_PROFILE=torch is set but SWORDFISH_PROFILE_OUT is missing
    — this is a dispatch contract violation, not a user error, so failing
    loudly is the right call.
    """
    mode = os.environ.get(SWORDFISH_PROFILE_ENV, "").strip()
    if mode != TORCH_PROFILE_MODE:
        return None
    out = os.environ.get(SWORDFISH_PROFILE_OUT_ENV, "").strip()
    if not out:
        raise ValueError(
            f"{SWORDFISH_PROFILE_ENV}={TORCH_PROFILE_MODE} but "
            f"{SWORDFISH_PROFILE_OUT_ENV} is unset; dispatch SDK should have "
            "set both"
        )
    return Path(out)


@contextlib.contextmanager
def torch_profiler_context(out_path: Path | None) -> Iterator[None]:
    """Wrap a code block in torch.profiler when out_path is set.

    No-op when out_path is None. When set, captures CPU + CUDA activities
    and exports a Chrome trace JSON to out_path on exit. The trace can be
    loaded into Perfetto (https://ui.perfetto.dev/) or chrome://tracing.

    The parent directory must already exist (the bash entrypoint creates
    it). Failures inside torch.profiler are non-fatal — the benchmark
    body runs to completion either way and the failure is logged.
    """
    if out_path is None:
        yield
        return

    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError as exc:  # pragma: no cover — torch always present in image
        print(
            f"[swordfish-profile] torch.profiler unavailable ({exc}); running without profile",
            file=sys.stderr,
        )
        yield
        return

    activities = [ProfilerActivity.CPU]
    try:
        import torch

        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
    except Exception:  # pragma: no cover
        pass

    print(
        f"[swordfish-profile] torch.profiler enabled (activities={activities}, out={out_path})",
        file=sys.stderr,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        yield
    try:
        prof.export_chrome_trace(str(out_path))
        print(f"[swordfish-profile] wrote {out_path}", file=sys.stderr)
    except Exception as exc:  # pragma: no cover
        print(
            f"[swordfish-profile] export_chrome_trace failed: {exc!r}",
            file=sys.stderr,
        )
