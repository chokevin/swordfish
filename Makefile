.PHONY: test airun-render airun-dry-run airun-apply airun-a100-ncu-preflight airun-a100-apply airun-h200-preflight airun-h200-apply airun-validate-results dashboard-index completion-report

AIRUN_CONFIG ?= infra/airun/airun-gemm.voice-agent-flex.json
AIRUN_MANIFEST_DIR ?= infra/airun/generated/week1
AIRUN_PREFLIGHT ?= infra/airun/generated/h200-preflight.sh
AIRUN_A100_PREFLIGHT ?= infra/airun/generated/a100-ncu-preflight.sh
AIRUN_ARCH_LABELS ?= a100 h100
AIRUN_VALIDATE_ARCH_LABELS ?= a100 h100 h200
AIRUN_RESULT_DIR ?= runs/airun/week1
AIRUN_RESULT_PREFIX ?= torch-gemm
AIRUN_REQUIRE_NCU ?= --require-ncu
AIRUN_VALIDATE_RECURSIVE ?= --recursive
DASHBOARD_RESULT_INDEX ?= docs/dashboard/results-index.json
COMPLETION_REPORT ?= docs/dashboard/completion-report.md

test:
	uv run ruff format --check swordfish tests
	uv run ruff check swordfish tests
	uv run pytest -q

airun-render:
	uv run python -m swordfish.runner render-airun-gemm \
		--config $(AIRUN_CONFIG) \
		--manifest-dir $(AIRUN_MANIFEST_DIR) \
		--arch-labels $(AIRUN_ARCH_LABELS)

airun-dry-run:
	uv run python -m swordfish.runner render-airun-gemm \
		--config $(AIRUN_CONFIG) \
		--manifest-dir $(AIRUN_MANIFEST_DIR) \
		--arch-labels $(AIRUN_ARCH_LABELS) \
		--dry-run-client

airun-apply:
	@if printf ' %s ' "$(AIRUN_ARCH_LABELS)" | grep -q ' a100 '; then \
		$(MAKE) airun-a100-ncu-preflight; \
	fi
	uv run python -m swordfish.runner render-airun-gemm \
		--config $(AIRUN_CONFIG) \
		--manifest-dir $(AIRUN_MANIFEST_DIR) \
		--arch-labels $(AIRUN_ARCH_LABELS) \
		--apply

airun-a100-ncu-preflight:
	uv run python -m swordfish.runner render-airun-preflight \
		--config $(AIRUN_CONFIG) \
		--arch-label a100 \
		--out $(AIRUN_A100_PREFLIGHT) \
		--run

airun-a100-apply: airun-a100-ncu-preflight
	uv run python -m swordfish.runner render-airun-gemm \
		--config $(AIRUN_CONFIG) \
		--manifest-dir $(AIRUN_MANIFEST_DIR) \
		--arch-labels a100 \
		--apply

airun-h200-preflight:
	uv run python -m swordfish.runner render-airun-preflight \
		--config $(AIRUN_CONFIG) \
		--arch-label h200 \
		--blocker-pod sf-gemm-133050-h200-8wr7p \
		--out $(AIRUN_PREFLIGHT) \
		--run

airun-h200-apply: airun-h200-preflight
	$(MAKE) airun-apply AIRUN_ARCH_LABELS=h200

airun-validate-results:
	uv run python -m swordfish.runner validate-gemm-matrix \
		--result-dir $(AIRUN_RESULT_DIR) \
		--prefix $(AIRUN_RESULT_PREFIX) \
		--backend torch \
		--dtype fp16 \
		--m 4096 --n 4096 --k 4096 \
		--arch-labels $(AIRUN_VALIDATE_ARCH_LABELS) \
		$(AIRUN_VALIDATE_RECURSIVE) \
		$(AIRUN_REQUIRE_NCU)

dashboard-index:
	uv run python -m swordfish.runner index-results \
		--result-dir $(AIRUN_RESULT_DIR) \
		$(AIRUN_VALIDATE_RECURSIVE) \
		--out $(DASHBOARD_RESULT_INDEX)

completion-report:
	uv run python -m swordfish.runner render-completion-report \
		--result-dir $(AIRUN_RESULT_DIR) \
		--prefix $(AIRUN_RESULT_PREFIX) \
		--backend torch \
		--dtype fp16 \
		--m 4096 --n 4096 --k 4096 \
		--arch-labels $(AIRUN_VALIDATE_ARCH_LABELS) \
		$(AIRUN_VALIDATE_RECURSIVE) \
		$(AIRUN_REQUIRE_NCU) \
		--out $(COMPLETION_REPORT)
