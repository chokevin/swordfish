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
SUBMIT_BENCH = uv run python -m swordfish.runner submit-bench \
    --result-root $(RUNE_RESULT_DIR) \
    --script $(RUNE_BENCH_SCRIPT)

.PHONY: rune-profiles rune-profiles-check \
        rune-install-profiles \
        rune-submit-gemm-a100 rune-submit-gemm-h100 rune-submit-gemm-h200 \
        rune-submit-gemm-matrix \
        rune-submit-liger-rmsnorm-a100 rune-submit-liger-swiglu-a100

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
	@echo "expected: 3 swordfish-bench-* profiles (core parents are embedded in the rune binary)"

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
