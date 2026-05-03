.PHONY: test validate-results dashboard-index completion-report

RESULT_DIR ?= runs/rune/week1
RESULT_PREFIX ?= torch-gemm
VALIDATE_ARCH_LABELS ?= a100 h100 h200
REQUIRE_NCU ?= --require-ncu
VALIDATE_RECURSIVE ?= --recursive
DASHBOARD_RESULT_INDEX ?= docs/dashboard/results-index.json
COMPLETION_REPORT ?= docs/dashboard/completion-report.md

test:
	uv run ruff format --check swordfish tests
	uv run ruff check swordfish tests
	uv run pytest -q

validate-results:
	uv run python -m swordfish.runner validate-gemm-matrix \
		--result-dir $(RESULT_DIR) \
		--prefix $(RESULT_PREFIX) \
		--backend torch \
		--dtype fp16 \
		--m 4096 --n 4096 --k 4096 \
		--arch-labels $(VALIDATE_ARCH_LABELS) \
		$(VALIDATE_RECURSIVE) \
		$(REQUIRE_NCU)

dashboard-index:
	uv run python -m swordfish.runner index-results \
		--result-dir $(RESULT_DIR) \
		$(VALIDATE_RECURSIVE) \
		--out $(DASHBOARD_RESULT_INDEX)

completion-report:
	uv run python -m swordfish.runner render-completion-report \
		--result-dir $(RESULT_DIR) \
		--prefix $(RESULT_PREFIX) \
		--backend torch \
		--dtype fp16 \
		--m 4096 --n 4096 --k 4096 \
		--arch-labels $(VALIDATE_ARCH_LABELS) \
		$(VALIDATE_RECURSIVE) \
		$(REQUIRE_NCU) \
		--out $(COMPLETION_REPORT)


# ---- rune dispatch -------------------------------------------------------

RUNE_PROFILES_DIR ?= $(HOME)/.config/rune/profiles
RUNE_NAME_PREFIX ?= swordfish-gemm
RUNE_RUN_ID ?= $(shell date +%H%M%S)
RUNE_SHAPE_M ?= 4096
RUNE_SHAPE_N ?= 4096
RUNE_SHAPE_K ?= 4096
RUNE_DTYPE ?= fp16
RUNE_REPEATS ?= 5
RUNE_WARMUP ?= 10
RUNE_ITERS ?= 50
RUNE_RESULT_DIR ?= /data/swordfish/week1
RUNE_BENCH_SCRIPT ?= infra/rune/scripts/swordfish-bench.sh
RUNE_PROFILE_PACK ?= infra/rune/profiles/swordfish-pack.yaml
RUNE_NAMESPACE ?= ray
RUNE_PVC ?= training-nfs
RUNE_IMAGE_REF ?= voiceagentcr.azurecr.io/swordfish-bench:latest
RUNE_RELEASE_TAG ?= rune-cli-v0.2.0
RUNE_REPO ?= azure-management-and-platforms/aks-ai-runtime
GITHUB_HOST ?= github.com
RUNE_PY_SPEC ?= rune-py @ git+https://github.com/azure-management-and-platforms/aks-ai-runtime.git@$(RUNE_RELEASE_TAG)\#subdirectory=applications/rune-py
SUBMIT_BENCH = uv run python -m swordfish.runner submit-bench \
    --result-root $(RUNE_RESULT_DIR) \
    --script $(RUNE_BENCH_SCRIPT)

.PHONY: rune-bootstrap \
        rune-profiles rune-profiles-check \
        rune-install-profiles \
        rune-submit-gemm-a100 rune-submit-gemm-h100 rune-submit-gemm-h200 \
        rune-submit-gemm-matrix \
        rune-submit-liger-rmsnorm-a100 rune-submit-liger-swiglu-a100 \
        rune-submit-liger-fsdp-a100-baseline rune-submit-liger-fsdp-a100-liger \
        rune-convert-ncu

rune-profiles:
	uv run python -m swordfish.runner generate-rune-profiles --out $(RUNE_PROFILE_PACK)

rune-profiles-check:
	uv run python -m swordfish.runner generate-rune-profiles --out $(RUNE_PROFILE_PACK) --check

rune-install-profiles: rune-profiles-check
	mkdir -p $(RUNE_PROFILES_DIR)
	for f in infra/rune/profiles/*.yaml; do \
		ln -sf $(PWD)/$$f $(RUNE_PROFILES_DIR)/$$(basename $$f); \
	done
	@echo "rune profiles installed under $(RUNE_PROFILES_DIR)/"
	@echo "verify with: rune profile list"
	@echo "expected: 4 swordfish-bench-* and 4 swordfish-fsdp-* profiles"

rune-submit-gemm-a100:
	$(SUBMIT_BENCH) --workload gemm --arch a100 \
	    --name $(RUNE_NAME_PREFIX)-$(RUNE_RUN_ID)-a100 \
	    --m $(RUNE_SHAPE_M) --n $(RUNE_SHAPE_N) --k $(RUNE_SHAPE_K) \
	    --dtype $(RUNE_DTYPE) \
	    --repeats $(RUNE_REPEATS) --warmup $(RUNE_WARMUP) --iters $(RUNE_ITERS)

rune-submit-gemm-h100:
	$(SUBMIT_BENCH) --workload gemm --arch h100 \
	    --name $(RUNE_NAME_PREFIX)-$(RUNE_RUN_ID)-h100 \
	    --m $(RUNE_SHAPE_M) --n $(RUNE_SHAPE_N) --k $(RUNE_SHAPE_K) \
	    --dtype $(RUNE_DTYPE) \
	    --repeats $(RUNE_REPEATS) --warmup $(RUNE_WARMUP) --iters $(RUNE_ITERS)

rune-submit-gemm-h200:
	$(SUBMIT_BENCH) --workload gemm --arch h200 \
	    --name $(RUNE_NAME_PREFIX)-$(RUNE_RUN_ID)-h200 \
	    --m $(RUNE_SHAPE_M) --n $(RUNE_SHAPE_N) --k $(RUNE_SHAPE_K) \
	    --dtype $(RUNE_DTYPE) \
	    --repeats $(RUNE_REPEATS) --warmup $(RUNE_WARMUP) --iters $(RUNE_ITERS)

rune-submit-gemm-matrix: rune-submit-gemm-a100 rune-submit-gemm-h100 rune-submit-gemm-h200

rune-submit-liger-rmsnorm-a100:
	$(SUBMIT_BENCH) --workload liger-rmsnorm --arch a100 \
	    --name swordfish-liger-rmsnorm-$(RUNE_RUN_ID)-a100 \
	    --repeats $(RUNE_REPEATS) --warmup $(RUNE_WARMUP) --iters $(RUNE_ITERS)

rune-submit-liger-swiglu-a100:
	$(SUBMIT_BENCH) --workload liger-swiglu --arch a100 \
	    --name swordfish-liger-swiglu-$(RUNE_RUN_ID)-a100 \
	    --repeats $(RUNE_REPEATS) --warmup $(RUNE_WARMUP) --iters $(RUNE_ITERS)

rune-submit-liger-fsdp-a100-baseline:
	$(SUBMIT_BENCH) --workload liger-fsdp --arch a100 \
	    --name swordfish-liger-fsdp-baseline-$(RUNE_RUN_ID)-a100 \
	    --liger-mode baseline \
	    --repeats 3 --warmup 1 --iters 5

rune-submit-liger-fsdp-a100-liger:
	$(SUBMIT_BENCH) --workload liger-fsdp --arch a100 \
	    --name swordfish-liger-fsdp-liger-$(RUNE_RUN_ID)-a100 \
	    --liger-mode liger \
	    --repeats 3 --warmup 1 --iters 5

# Convert a cluster-side .ncu-rep into a .ncu-summary.csv companion via a
# CPU-only Pod that runs `ncu --import` against the PVC. Mac developers with
# Nsight Compute installed don't need this — the runner's ncu_summary path
# reads .ncu-rep directly. This target is for CI / Linux runners that lack
# the local install. Usage: `make rune-convert-ncu JOB=sf-gemm-h100-XYZ`.
rune-convert-ncu:
	@if [ -z "$(JOB)" ]; then \
		echo "usage: make rune-convert-ncu JOB=<rune-job-name>"; exit 2; \
	fi
	uv run python -m swordfish.runner convert-ncu $(JOB) \
	    --namespace $(RUNE_NAMESPACE) \
	    --pvc $(RUNE_PVC) \
	    --image $(RUNE_IMAGE_REF)

rune-bootstrap:
	@if ! command -v gh >/dev/null 2>&1; then echo "missing: gh (install: brew install gh)" >&2; exit 1; fi
	@if ! command -v uv >/dev/null 2>&1; then echo "missing: uv (install: brew install uv)" >&2; exit 1; fi
	@if ! gh auth status --hostname $(GITHUB_HOST) >/dev/null 2>&1; then gh auth login --hostname $(GITHUB_HOST); fi
	gh auth setup-git --hostname $(GITHUB_HOST) >/dev/null
	gh release view $(RUNE_RELEASE_TAG) --repo $(RUNE_REPO) >/dev/null
	uv sync
	uv pip install "$(RUNE_PY_SPEC)"
	uv run rune-py bootstrap \
	    --tag $(RUNE_RELEASE_TAG) \
	    --repo $(RUNE_REPO) \
	    --github-host $(GITHUB_HOST)
	uv run rune-py doctor -n $(RUNE_NAMESPACE)
