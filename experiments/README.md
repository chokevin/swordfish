# swordfish experiments

Day-to-day iteration unit. **Each file in this directory is one runnable
benchmark or one-off experiment** with a `#!/usr/bin/env python3` shebang.

The `swordfish-bench` image bakes the `swordfish` package + `liger-kernel` +
torch + Triton + Nsight tools. `rune submit --script` base64-encodes this
file into the pod, the pod respects the shebang, and your edits run in the
next dispatch with no image rebuild.

## Iteration loop

1. Edit a file in this directory (or add a new one).
2. Dispatch via the SDK:

   ```python
   from swordfish.dispatch import LigerPerkernelRun
   LigerPerkernelRun(
       kernel="rmsnorm", arch="a100",
       script="experiments/liger_rmsnorm.py",
   ).submit()
   ```

3. Watch logs: `rune logs <job-name> -n ray -f`.
4. Fetch result: `LigerPerkernelRun(...).fetch_result()`
   (uses `kubectl exec` against a helper pod with the PVC mounted).

## When to rebuild the image

Image rebuild is only needed when the dependency surface itself changes:

- new `pip install` package the experiment imports
- new system tool (apt, gh, nsys version)
- bumped `liger-kernel` pin
- swordfish package internals you want stable across many experiments

For those: edit `infra/rune/image/Dockerfile`, then either let CI rebuild on
push to main (`build-swordfish-image.yml`) or run locally:

```bash
PUSH=1 ./infra/rune/image/build.sh
```

The local build auto-tags `dev-<sha>[-dirty]` and pushes to GHCR. The SDK
picks it up via the `image=` arg:

```python
LigerPerkernelRun(
    kernel="rmsnorm", arch="a100",
    image="ghcr.io/chokevin/swordfish-bench:dev-abc1234",
    script="experiments/liger_rmsnorm.py",
).submit()
```

## What goes in this directory

- One file per benchmark or experiment.
- Self-contained — imports from `swordfish.*`, `torch`, and stdlib only;
  anything else needs to be in the image first.
- Reads `RUNE_DATA_DIR` env var for output path so the same script works
  whether `/data` or another mount is used.
- Auto-detects GPU arch via `torch.cuda.get_device_name()` — same script
  produces correctly-named result JSONs across A100 / H100 NVL / H200.

## What does NOT go here

- Library code — that belongs in `swordfish/`.
- One-off shell pipelines — those go in `infra/rune/scripts/`.
- Dashboard/result indexing — that lives in `swordfish/runner/index.py`.
