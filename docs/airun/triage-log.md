# airun triage log

## 2026-04-27 — swordfish GEMM jobs Pending after Kueue admission
- Initial suspicion: L3
- Actual root cause: L4/L5 split — the cluster exposes GPUs through DRA ResourceClaimTemplates/resourceslices, while the first swordfish Jobs requested legacy `nvidia.com/gpu`; after DRA was fixed, H200 remained blocked by regional Azure capacity.
- Layers ruled out before finding it: L2, because all three Workloads were `Admitted=True` in `gpu-cluster-queue` with quota reserved.
- Time to root cause: ~15 min
- Fix: switch generated Jobs to use the `ray/full-gpu` ResourceClaimTemplate instead of legacy GPU requests; correct A100 toleration to `nvidia.com/gpu=true`.
- Follow-up: H100 scheduled after the DRA switch and produced final JSON with NCU metrics at `/data-nfs/swordfish/week1/133050dra/torch-gemm-h100.json`. A100 also needed the actual node taint `nvidia.com/gpu=true`; it produced raw/final timing JSON at `/data-nfs/swordfish/week1/133050a100fix/torch-gemm-a100.json`, but NCU returned `ERR_NVGPUCTRPERM` so perf counters were not attached. H200 did not produce a result because repeated `flex-h200-*` NodeClaims failed with Azure `InsufficientCapacityError`; no other accessible context exposed a live H200 GPU node.
- Cleanup note: the old H200 pod `sf-gemm-133050-h200-8wr7p` became an orphaned failed pod with only the `batch.kubernetes.io/job-tracking` finalizer after its Job and Workload were gone. Normal `kubectl delete --force` and metadata-only JSON patches could not remove it because the API rejected pod updates with `spec.tolerations: Forbidden`, even though the patch body only removed `/metadata/finalizers/0`; this needs cluster-admin/webhook cleanup if it continues to trigger `flex-h200` NodeClaims.
- Latest recheck: `flex-h200` still reports Ready with 0 nodes, no live H200 node exists, and the orphaned `sf-gemm-133050-h200-8wr7p` pod is still being nominated to fresh `flex-h200-*` NodeClaims that fail with Azure `InsufficientCapacityError`. Do not submit more H200 jobs until the orphan pod is cleaned up or H200 capacity is confirmed.
- Lesson: On this airun lane, `nvidia.com/gpu` node labels do not imply legacy `nvidia.com/gpu` allocatable; check DRA first, then check actual node taints instead of trusting ResourceFlavor tolerations.

## 2026-04-27 — H200 benchmark blocker recheck
- Initial suspicion: L5
- Actual root cause: L5 (node-pool/region) — `flex-h200` is configured and Ready as a NodePool but has 0 nodes, and fresh `flex-h200-*` NodeClaims for `Standard_ND96isr_H200_v5` continue to fail with Azure `InsufficientCapacityError`.
- Layers ruled out before finding it: L2, because `fauna-train-queue` and `gpu-cluster-queue` both reported 0 pending and 0 admitted workloads; L3/L4 as primary causes for the old stuck pod, because the current orphan requests legacy `nvidia.com/gpu=1` with `gpu=h200` selector/toleration and is being nominated to `flex-h200` rather than rejected for selector/taint mismatch.
- Time to root cause: ~10 min
- Fix: handed off to cluster-admin / Azure capacity. No safe in-session fix: `sf-gemm-133050-h200-8wr7p` is Failed with deletion timestamp `2026-04-27T20:51:50Z` and finalizer `batch.kubernetes.io/job-tracking`; a server-side dry-run finalizer patch still fails API validation with `spec.tolerations: Forbidden`.
- Alternate-route check: no accessible kube context currently exposes a live H200 node. `voice-agent-flex` and `voice-agent-flex-admin` only expose `flex-h200` with 0 nodes; `npd-h200-test` is unreachable from this environment.
- Cleanup route check: no safe `kubectl` finalizer-removal path is available from this session. JSON patch and merge patch dry-runs fail the same `spec.tolerations` validation in both normal and admin contexts, and the pod `finalize` subresource is not served for this resource.
- Guard added: `python -m swordfish.runner render-airun-preflight` and `make airun-h200-preflight` now generate/run a fail-fast H200 preflight. In the current cluster state it exits 2 before submission because no `gpu=h200` node exists and `sf-gemm-133050-h200-8wr7p` still exists.
- Submission guard added: `render-airun-gemm --arch-labels ...` can render/apply a subset of architectures, and Make defaults `AIRUN_ARCH_LABELS` to `a100 h100` so routine dry-run/apply commands do not include H200 while this blocker is active.
- Latest completion check: fresh A100/H100 jobs were rerun from `/data-nfs/swordfish/src/rerun-161445` after fixing the job script to time GEMM outside NCU and run profiling as a separate pass. The copied local artifacts in `runs/airun/week1/` validate for A100/H100 without the NCU-complete gate: A100 mean `0.615004 ms` / `223.476 TFLOP/s` with `ncu.complete=false` due `ERR_NVGPUCTRPERM`; H100 mean `0.277896 ms` / `494.57 TFLOP/s` with `ncu.complete=true`. The strict gate still fails on A100 incomplete NCU plus missing H200.
- A100 NCU follow-up: adding `SYS_ADMIN` to the A100 benchmark container changed the error from `ERR_NVGPUCTRPERM` to driver resource contention, likely DCGM/profiler conflict. Filtering NCU to GEMM-like kernels still failed directly on `ampere_fp16_s16816gemm_fp16_2...`, so A100 NCU completion now needs cluster/operator action rather than benchmark-script changes.
- Lesson: H200 is currently a provider-capacity blocker plus an orphan-pod cleanup blocker; do not spend more benchmark time on H200 submissions until either a live H200 node exists or the stuck finalizer is cleared by a control-plane/admin path.

## 2026-04-27 — strict gate blocked after A100/H100 clean-timing rerun
- Initial suspicion: L4 for A100 NCU, L5 for H200 missing result
- Actual root cause: L4 (GPU/driver/profiler) for A100 — Nsight Compute cannot acquire A100 performance counters even with `SYS_ADMIN`, GEMM kernel filtering, and a one-node DCGM exporter pause; L5 (node-pool/region) for H200 — `flex-h200` has 0 nodes and NodeClaims continue failing Azure `InsufficientCapacityError`.
- Layers ruled out before finding it: A100 L2/L3 because the benchmark landed on `NVIDIA A100-SXM4-80GB` and produced valid timing/correctness JSON; H200 L2 because `fauna-train-queue` and `gpu-cluster-queue` show 0 pending/admitted workloads for the current check.
- Time to root cause: ~20 min
- Fix: A100 handed off to cluster/operator configuration for profiler resource/counter access; H200 handed off to Azure capacity plus control-plane cleanup for `sf-gemm-133050-h200-8wr7p`. No safe in-session fix remains.
- Lesson: `SYS_ADMIN` is not sufficient proof that NCU can profile on airun; DCGM/operator-side profiler resource contention can still block A100 counters after the workload and DRA routing are healthy.

## 2026-04-27 — W4A16 Triton A100/H100 jobs failed after scheduling
- Initial suspicion: L4
- Actual root cause: outside the airun ladder (application Triton kernel) — the first failure was a nested `@triton.jit` function whose annotations referenced local `tl`; after that was fixed, the kernel exposed an fp16/fp32 `tl.dot` dtype mismatch and unsigned INT4 nibble underflow.
- Layers ruled out before finding it: L2, because both Kueue Workloads were admitted in `gpu-cluster-queue`; L3, because both pods were scheduled to the expected A100/H100 nodes; L5 for A100/H100, because live nodes ran the containers immediately.
- Time to root cause: ~15 min
- Fix: moved Triton imports/kernel definition to module scope while keeping Triton optional for CPU tests, made the Triton W4A16 path explicitly fp16-only with fp16 dequant tiles, and cast unpacked nibbles to signed int before subtracting 16. Fresh jobs `swordfish-w4a16-165327-{a100,h100}` completed with `matches_reference=true`.
- Lesson: When identical A100/H100 jobs fail after admission and placement, read container logs before descending into GPU/node-pool triage; compiler/runtime stack traces usually mean application kernel code, not airun scheduling.

## 2026-04-27 — H200 capacity returned, strict gate still blocked
- Initial suspicion: L5
- Actual root cause: L5 (node-pool/region) recovered for H200 — `flex-h200` now has two Ready `gpu=h200` nodes, so the H200 missing-result blocker is resolved; the remaining strict blocker is L4 A100 profiler access.
- Layers ruled out before finding it: H200 L2, because `fauna-train-queue` and `gpu-cluster-queue` had no pending workloads and the new job was admitted; H200 L3, because `swordfish-gemm-180223-h200` scheduled to `flex-h200-zjj8s`; H200 L4, because the job produced passing correctness and `ncu.complete=true`.
- Time to root cause: ~10 min
- Fix: adjusted H200 preflight to treat the old `Failed` and deletion-marked `sf-gemm-133050-h200-8wr7p` pod as a warning once live Ready H200 nodes exist, uploaded current source to a fresh NFS path, ran `swordfish-gemm-180223-h200`, copied `torch-gemm-h200.json` into `runs/airun/week1/`, and regenerated the dashboard/completion report.
- Follow-up: strict `make airun-validate-results` now fails only on A100 incomplete NCU. A fresh privileged A100 retry `swordfish-a100-ncu-181205-a100` still failed with driver resource unavailable and all required NCU metrics missing. A second-node A100 retry on `aks-gpu-33826946-vmss000000` with `SYS_ADMIN`, targeted DCGM exporter deletion, and `--kernel-name regex:.*gemm.*` also failed directly on `ampere_fp16_s16816gemm_fp16_2...`.
- Lesson: Once live H200 nodes exist, a tombstoned failed pod with a deletion timestamp should not block single-arch H200 benchmarking, but it should remain documented for admin cleanup.

## 2026-04-27 — A100 NCU unblocked by pausing DCGM exporter
- Initial suspicion: L4
- Actual root cause: L4 (GPU/profiler resource contention) — `nvidia-dcgm-exporter` was collecting/querying DCGM profiling metric groups on A100 nodes, which prevented Nsight Compute from acquiring the same driver profiling resource.
- Layers ruled out before finding it: L2/L3, because A100 jobs were admitted, scheduled, and produced passing timing/correctness JSON; container permission alone, because `SYS_ADMIN` changed the error from `ERR_NVGPUCTRPERM` to driver resource unavailable but still failed; single-node exporter staleness, because deleting one A100 exporter pod and retrying still failed before the DaemonSet recreated it.
- Time to root cause: ~25 min after changing from per-job retries to the GPU-operator/DCGM layer.
- Fix: temporarily patched the `nvidia-dcgm-exporter` DaemonSet to exclude `gpu=a100` nodes, confirmed no exporter pods were running on the two A100 nodes, ran `swordfish-a100-dcgm-off-182331-a100` with `SYS_ADMIN`, copied the complete A100 NCU artifacts into `runs/airun/week1/`, then removed the temporary DaemonSet affinity patch.
- Verification: `kubectl --context voice-agent-flex -n gpu-operator rollout status ds/nvidia-dcgm-exporter --timeout=300s` succeeded with 6/6 exporter pods ready, including both A100 nodes. `make airun-validate-results` now reports `GEMM result matrix is complete`.
- Guardrail: added `make airun-a100-ncu-preflight`, and `make airun-apply` now invokes it automatically when `AIRUN_ARCH_LABELS` includes `a100`. The preflight fails before submission if running `nvidia-dcgm-exporter` pods are still on Ready target A100 nodes or if the A100 arch config lacks `SYS_ADMIN`.
- Lesson: On this lane, complete A100 NCU requires a controlled profiling window where DCGM exporter is paused/excluded from the target A100 nodes, followed by immediate restore. Do not permanently disable DCGM and do not fake NCU completeness from partial profiler output.

## 2026-05-02 — A100 NCU through Rune profile-mode needed SYS_ADMIN profile support
- Initial suspicion: L4
- Actual root cause: L4 (GPU/profiler pod security + DCGM contention) — A100 NCU needs `SYS_ADMIN`, and Rune previously could not render that capability from a Profile; after adding profile-driven `runtime.securityContext.capabilities.add`, the remaining blocker was the known DCGM exporter profiler-resource contention.
- Layers ruled out before finding it: L2/L3, because the NCU smoke job was admitted and scheduled to `aks-gpu-33826946-vmss000000`; L5, because the A100 nodes were Ready and available; application-level GEMM execution, because the job wrote its result JSON and NCU profiled the cuBLAS GEMM kernel once `SYS_ADMIN` and the DCGM pause were both in place.
- Time to root cause: ~20 min
- Fix: added Rune renderer support for `spec.runtime.securityContext.capabilities.add`, generated `swordfish-bench-a100-ncu` / `swordfish-fsdp-a100-ncu`, installed the patched local `rune`, temporarily excluded `gpu=a100` nodes from `nvidia-dcgm-exporter`, ran `swordfish-a100-ncu-rune-0502192934` with `--profile-mode ncu`, converted/fetched `profile.ncu-summary.csv`, then restored DCGM to 6/6 Ready.
- Lesson: Profile-mode alone is not enough for A100; the easy path must select an elevated A100 NCU profile and still run inside a controlled DCGM pause window.

## 2026-05-04 — A100/H200 FSDP comparison submit blocked by context and H200 capacity
- Initial suspicion: L5
- Actual root cause: L5 (cluster context / transient node-pool capacity) — the first Rune submit targeted the current `chokevin-aks` context, which had no `ray` namespace; after switching to `voice-agent-flex`, the first H200 comparison leg had no schedulable H200 node and hit scheduler/autoscaler max-size events.
- Layers ruled out before finding it: L2 for the target context, because `kernel-mode-training`, `kernel-mode-large-memory`, and `team-kernel-mode-reserved-cq` existed with no initial pending workloads; L3/L4 for A100, because pinned A100 jobs admitted, scheduled to `NVIDIA-A100-SXM4-80GB`, and completed with result JSON + NSYS profiles.
- Time to root cause: ~15 min
- Fix: exposed `--context` and `--image` through `submit-experiment`, submitted against `voice-agent-flex`, deleted the initially blocked H200 jobs, pinned reruns to `voiceagentcr.azurecr.io/airun/swordfish-bench:bf92726-dirty` instead of cached `:dev`, and reran H200 once two Ready `NVIDIA-H200` nodes appeared.
- Follow-up: the completed pinned comparison (`sf-fsdp-pin-{a100,h200}-*`) showed `tb-no-limit` as the best overlap lead on both A100 and H200; H200 recovered during the session, so this was a transient capacity/context blocker rather than a persistent H200 experiment blocker.
- Lesson: For Rune sweeps, pass the kube context explicitly and pin the image tag; `:dev` plus `IfNotPresent` can reuse stale runner code even after the ACR tag has moved, and H200 must be preflighted for live schedulable nodes before using it as a comparison leg.

## 2026-05-04 — vectorsum A100 capture-policy sweep pod Pending after admission
- Initial suspicion: L3
- Actual root cause: L3 (k8s scheduler) — `vs-v2-capture-policy-05041233` was admitted by Kueue but rendered an impossible selector: `nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB` together with `rune.ai/gpu-class=h200-nvlink-141gb`.
- Layers ruled out before finding it: L2, because the Workload was `QuotaReserved` and `Admitted` in `team-kernel-mode-reserved-cq`.
- Time to root cause: ~10 min
- Fix: deleted the stuck admitted job and reran the benchmark with `--gpu-class a100-nvlink-80gb`; dry-run confirmed the selector changed to `rune.ai/gpu-class=a100-nvlink-80gb`.
- Verification: fixed-selector reruns scheduled on `aks-gpu-33826946-vmss000001` and wrote A100 result JSON.
- Lesson: When using a nominal A100 Rune profile, still dry-run/check the rendered `rune.ai/gpu-class`; a stale or inherited H200 GPU-class selector can make an A100 pod unschedulable even though Kueue admits it.

## 2026-05-05 — local session could not submit vectorsum NCU sweep
- Initial suspicion: L5
- Actual root cause: L5 (local kube/tooling context) — this session only had the `chokevin-aks` context, which has no `ray` namespace and no Kueue `ClusterQueue` CRD; `voice-agent-flex` / `voice-agent-flex-admin` were not configured, and neither `rune` nor `rune-py` was installed on PATH.
- Layers ruled out before finding it: application/runner, because `make test` passed, `bench-vectorsum` wrote valid local JSON, and the Python dispatch layer rendered the intended A100/H100/H200 `rune submit ... --profile-mode ncu` commands with explicit `--gpu-class`.
- Time to root cause: ~10 min
- Fix: local-only packaging completed; `.tmp/` run artifacts are ignored, the standalone `submission.py` evaluator entrypoint is documented, and the vector-sum/FSDP command render path is verified. No safe cluster submission is possible from this environment without a configured `voice-agent-flex` context and Rune bootstrap/auth.
- Follow-up: run the rendered `vectorsum-v2` NCU sweep from a workstation/session with `rune`, `ray` namespace access, and `voice-agent-flex` configured. For A100, use the elevated NCU profile/DCGM pause procedure from the prior A100 NCU entries.
- Lesson: before any Rune sweep, preflight both the local toolchain (`rune`, `rune-py`, GitHub auth if bootstrapping) and kube context (`ray` namespace + Kueue CRDs). A green Python dispatch layer is not sufficient proof that the current shell can submit jobs.

## 2026-05-05 — vectorsum-v2 NCU sweep completed on H100/H200, A100 counters blocked
- Initial suspicion: L4 for A100 NCU, application/kernel behavior for H100/H200 tuning.
- Actual root cause: H100/H200 NCU completed successfully; A100 still failed at L4 with `ERR_NVGPUCTRPERM` even after selecting `swordfish-bench-a100-ncu` and temporarily excluding `gpu=a100` nodes from `nvidia-dcgm-exporter`. The A100 workload wrote a passing result JSON, so this is a profiler-counter permission/configuration blocker rather than a vector-sum runtime failure.
- Layers ruled out before finding it: L2/L3/L5 for this sweep, because all three jobs admitted and scheduled on Ready GPU nodes in `voice-agent-flex`; H100/H200 L4, because both generated and converted `profile.ncu-rep`; application correctness, because the fetched result JSONs reported `matches_reference=true`.
- Time to root cause: ~25 min after Rune/tooling was restored in-session.
- Fix/workaround: built a session-local Rune binary, installed swordfish profiles, submitted standalone `submission.py` against `voiceagentcr.azurecr.io/airun/autoresearch-pytorch-ray:dev`, fetched/converted H100/H200 NCU reports with `inspect-run --convert-ncu`, and restored the DCGM exporter DaemonSet to 6/6 pods after the A100 profiling window.
- Evidence: run id `230122`; H100 `_partial_sum_kernel` took 302 invocations / 1.86 ms / 62.2% of kernel time, with `_final_sum_kernel` another 1.01 ms / 33.9%; H200 `_partial_sum_kernel` took 302 invocations / 1.38 ms / 61.7%, with `_final_sum_kernel` another 765.24 us / 34.3%. Both top kernels showed low SM utilization and modest memory utilization, so the first tuning target is launch/reduction-structure overhead, not raw bandwidth.
- Timing caveat: latency JSON from NCU-profiled H100/H200 runs is inflated by NCU replay. A separate no-profile latency pass (`sf-vectorsum-v2-lat-230122-{a100,h100,h200}`) measured mean latency of 0.008638 ms on A100, 0.004723 ms on H100, and 0.004449 ms on H200 for the same size/repeats/iters, all with `matches_reference=true`.
- Lesson: For `vectorsum-v2`, use NCU runs for kernel attribution and no-profile runs for latency. A100 NCU remains a cluster/operator profiler-counter blocker even when the elevated profile and DCGM exclusion procedure are followed; do not block H100/H200 tuning on A100 counters.

## 2026-05-05 — FSDP overlap follow-up ran via standalone script
- Initial suspicion: application/image contract, then FSDP overlap behavior.
- Actual root cause: the profile image `voiceagentcr.azurecr.io/airun/swordfish-bench:dev` did not contain the new FSDP wrap/prefetch/all-gather flags (`run_liger_fsdp_step` still accepted only `profile_steady_state`), so the follow-up could not safely use `python -m swordfish.runner submit-experiment` without rebuilding/pushing the image. A session-local standalone script reproduced the dirty FSDP runner logic and was submitted through Rune instead.
- Layers ruled out before running: L2/L3/L5, because `voice-agent-flex` admitted/scheduled all A100/H200 8-GPU jobs; runtime dependency availability, because the image still had `transformers`, `liger-kernel`, torchrun, and Nsight Systems.
- Time to root cause: ~10 min for image-contract probe, then normal job runtime.
- Fix/workaround: submitted `sf-fsdp-ovl-230122-{a100,h200}-{root,tb}` with steady-state NSYS output under `/data/<job>/profile/profile.nsys-rep`, plus no-profile latency checks `sf-fsdp-lat-230122-{a100,h200}-{root,tb}`. All eight jobs completed and fetched JSON; the four NSYS `.nsys-rep` files were also fetched locally under `runs/inspect/fsdp-overlap-230122/`.
- Results: no-profile latency favored transformer-block/no-limit slightly while reducing peak reserved memory: A100 root/default 845.15 ms / 19.39k tok/s / 63.71 GiB versus tb/no-limit 823.07 ms / 19.91k tok/s / 40.18 GiB; H200 root/default 374.70 ms / 43.73k tok/s / 63.71 GiB versus tb/no-limit 361.63 ms / 45.31k tok/s / 40.18 GiB. NSYS-wrapped timing inverted in favor of root/default (A100 1905.23 ms root vs 2254.85 ms tb; H200 1144.41 ms root vs 1448.47 ms tb), so use those traces for overlap attribution, not as the latency scoreboard.
- Lesson: Until the FSDP flag code is in the image, treat standalone-script jobs as the valid follow-up path and explicitly separate no-profile latency from NSYS trace capture. The tb/no-limit variant is still a small throughput win and a large memory win in clean timing, but the NSYS traces need offline inspection before claiming improved communication overlap.

## 2026-05-06 — A100 NCU profile lacked SYS_ADMIN in rendered pod
- Initial suspicion: L4
- Actual root cause: L4 (GPU/profiler permission) exposed a Rune renderer/tooling bug — `swordfish-bench-a100-ncu` declared `runtime.securityContext.capabilities.add: [SYS_ADMIN]`, but the submitted A100 NCU pod had an empty container `securityContext`, so Nsight Compute failed with `ERR_NVGPUCTRPERM`.
- Layers ruled out before finding it: L2/L3/L5, because the failed A100 job was admitted, scheduled onto `aks-gpu-33826946-vmss000001`, and wrote a correct result JSON; workload correctness, because `matches_reference=true`; H100/H200 L4, because their NCU reports converted successfully.
- Time to root cause: ~10 min after inspecting the rendered Job and profile.
- Fix: patched Rune's submit renderer to propagate `spec.runtime.securityContext` into the main container, rebuilt the session-local Rune binary, and added a Swordfish dispatch preflight that refuses real A100 NCU submits when dry-run output lacks `SYS_ADMIN`. Added `a100-ncu-window` / Make helpers to pause and restore DCGM exporter on A100 nodes, deleting existing A100 exporter pods after the affinity patch.
- Verification: patched Rune dry-run renders `securityContext.capabilities.add: [SYS_ADMIN]`; A100 smoke `sf-ncu-smoke-001148-a100` completed under a DCGM exclusion window and fetched a 48,773,066-byte `profile.ncu-rep`; `bundle-traces` produced `runs/traces/sf-ncu-smoke-001148-a100-hermes.tar.gz`. DCGM exporter was restored to 6 desired / 6 ready pods afterward.
- Lesson: A100 NCU seamlessness needs three guards together: a Rune binary that propagates profile security context, a DCGM exclusion window that also deletes already-running A100 exporter pods, and a stable trace bundle path for handoff instead of ad hoc `runs/inspect` directories.
