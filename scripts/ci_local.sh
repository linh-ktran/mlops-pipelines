#!/usr/bin/env bash
# Run CI checks locally before pushing.
# Usage: bash scripts/ci_local.sh

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }
pass() { echo -e "${GREEN}✓ $1${NC}"; }

echo "════════════════════════════════════════"
echo " Running local CI checks"
echo "════════════════════════════════════════"

# Avoid VIRTUAL_ENV mismatch warnings — let uv manage per-project envs
unset VIRTUAL_ENV

PROJECTS=("shared" "aws-sagemaker-pipeline" "ibm-code-engine-pipeline" "ibm-watsonx-pipeline")

# Lint
for project in "${PROJECTS[@]}"; do
  echo ""
  echo "▸ Linting $project..."
  cd "$project"
  uv sync --extra dev
  uv run ruff check src/ tests/ || fail "ruff check failed in $project"
  uv run ruff format --check src/ tests/ || fail "ruff format failed in $project"
  pass "$project lint"
  cd ..
done

# Tests
echo ""
echo "▸ Testing shared..."
cd shared
uv run pytest tests/ -v --tb=short || fail "shared tests failed"
pass "shared tests"
cd ..

echo ""
echo "▸ Testing aws-sagemaker-pipeline..."
cd aws-sagemaker-pipeline
uv sync --extra dev
uv run pytest tests/ -v --tb=short -k "not sagemaker_deploy and not sagemaker_pipeline" || fail "aws tests failed"
pass "aws tests"
cd ..

echo ""
echo "▸ Testing ibm-code-engine-pipeline..."
cd ibm-code-engine-pipeline
uv sync --extra dev
uv run pytest tests/ -v --tb=short || fail "code-engine tests failed"
pass "code-engine tests"
cd ..

echo ""
echo "▸ Testing ibm-watsonx-pipeline..."
cd ibm-watsonx-pipeline
uv sync --extra dev
uv run pytest tests/ -v --tb=short || fail "watsonx tests failed"
pass "watsonx tests"
cd ..

echo ""
echo "════════════════════════════════════════"
echo -e "${GREEN} All CI checks passed!${NC}"
echo "════════════════════════════════════════"


