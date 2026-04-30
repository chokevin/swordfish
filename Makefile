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
RUNE_BENCH_SCRIPT ?= infra/rune/scripts/swordfish-bench.sh
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

.PHONY: rune-install-profiles rune-submit-gemm-a100 rune-submit-gemm-h100 rune-submit-gemm-h200 \
        rune-submit-gemm-matrix rune-submit-liger-rmsnorm-a100 rune-submit-liger-swiglu-a100

rune-install-profiles:
	mkdir -p $(RUNE_PROFILES_DIR)
	for f in infra/rune/profiles/*.yaml; do \
		ln -sf $(PWD)/$$f $(RUNE_PROFILES_DIR)/$$(basename $$f); \
	done
	@echo "rune profiles installed under $(RUNE_PROFILES_DIR)/"
	@echo "verify with: rune profile list"
	@echo "expected: 3 swordfish-bench-* profiles (core parents are embedded in the rune binary)"

rune-submit-gemm-a100:
	rune submit $(RUNE_NAME_PREFIX)-$(RUNE_RUN_ID)-a100 \
	    --profile swordfish-bench-a100 \
	    --script $(RUNE_BENCH_SCRIPT) \
	    --output $(RUNE_RESULT_DIR)/torch-gemm-a100.json \
	    -- run-gemm --backend torch \
	       --m $(RUNE_SHAPE_M) --n $(RUNE_SHAPE_N) --k $(RUNE_SHAPE_K) \
	       --dtype $(RUNE_DTYPE) \
	       --repeats $(RUNE_REPEATS) --warmup $(RUNE_WARMUP) --iters $(RUNE_ITERS) \
	       --device auto \
	       --out $(RUNE_RESULT_DIR)/torch-gemm-a100.json

rune-submit-gemm-h100:
	rune submit $(RUNE_NAME_PREFIX)-$(RUNE_RUN_ID)-h100 \
	    --profile swordfish-bench-h100 \
	    --script $(RUNE_BENCH_SCRIPT) \
	    --output $(RUNE_RESULT_DIR)/torch-gemm-h100.json \
	    -- run-gemm --backend torch \
	       --m $(RUNE_SHAPE_M) --n $(RUNE_SHAPE_N) --k $(RUNE_SHAPE_K) \
	       --dtype $(RUNE_DTYPE) \
	       --repeats $(RUNE_REPEATS) --warmup $(RUNE_WARMUP) --iters $(RUNE_ITERS) \
	       --device auto \
	       --out $(RUNE_RESULT_DIR)/torch-gemm-h100.json

rune-submit-gemm-h200:
	rune submit $(RUNE_NAME_PREFIX)-$(RUNE_RUN_ID)-h200 \
	    --profile swordfish-bench-h200 \
	    --script $(RUNE_BENCH_SCRIPT) \
	    --output $(RUNE_RESULT_DIR)/torch-gemm-h200.json \
	    -- run-gemm --backend torch \
	       --m $(RUNE_SHAPE_M) --n $(RUNE_SHAPE_N) --k $(RUNE_SHAPE_K) \
	       --dtype $(RUNE_DTYPE) \
	       --repeats $(RUNE_REPEATS) --warmup $(RUNE_WARMUP) --iters $(RUNE_ITERS) \
	       --device auto \
	       --out $(RUNE_RESULT_DIR)/torch-gemm-h200.json

rune-submit-gemm-matrix: rune-submit-gemm-a100 rune-submit-gemm-h100 rune-submit-gemm-h200

rune-submit-liger-rmsnorm-a100:
	rune submit swordfish-liger-rmsnorm-$(RUNE_RUN_ID)-a100 \
	    --profile swordfish-bench-a100 \
	    --script $(RUNE_BENCH_SCRIPT) \
	    --output $(RUNE_RESULT_DIR)/liger-perkernel/rmsnorm-a100.json \
	    -- liger-perkernel --kernel rmsnorm \
	       --batch 4 --seq 2048 --hidden 4096 --intermediate 14336 \
	       --dtype bf16 \
	       --repeats $(RUNE_REPEATS) --warmup $(RUNE_WARMUP) --iters $(RUNE_ITERS) \
	       --device auto \
	       --out $(RUNE_RESULT_DIR)/liger-perkernel/rmsnorm-a100.json

rune-submit-liger-swiglu-a100:
	rune submit swordfish-liger-swiglu-$(RUNE_RUN_ID)-a100 \
	    --profile swordfish-bench-a100 \
	    --script $(RUNE_BENCH_SCRIPT) \
	    --output $(RUNE_RESULT_DIR)/liger-perkernel/swiglu-a100.json \
	    -- liger-perkernel --kernel swiglu \
	       --batch 4 --seq 2048 --hidden 4096 --intermediate 14336 \
	       --dtype bf16 \
	       --repeats $(RUNE_REPEATS) --warmup $(RUNE_WARMUP) --iters $(RUNE_ITERS) \
	       --device auto \
	       --out $(RUNE_RESULT_DIR)/liger-perkernel/swiglu-a100.json
