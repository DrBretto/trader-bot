#!/bin/bash
# Fast-check runner with best-effort strategy
# Detects and runs available checks: typecheck, lint, tests, build

set -e

SKIPPED=()
FAILED=()

echo "Running fast checks..."

# Typecheck
if command -v tsc &> /dev/null && [ -f tsconfig.json ]; then
  echo "✓ Running TypeScript typecheck..."
  tsc --noEmit || FAILED+=("typecheck")
elif command -v pyright &> /dev/null; then
  echo "✓ Running Python typecheck..."
  pyright || FAILED+=("typecheck")
else
  SKIPPED+=("typecheck")
fi

# Lint
if [ -f package.json ] && grep -q '"lint"' package.json; then
  echo "✓ Running lint..."
  npm run lint || FAILED+=("lint")
elif [ -f pyproject.toml ] && grep -q "ruff\|flake8\|pylint" pyproject.toml; then
  echo "✓ Running Python lint..."
  ruff check . 2>/dev/null || flake8 . 2>/dev/null || FAILED+=("lint")
else
  SKIPPED+=("lint")
fi

# Unit tests (fast only)
if [ -f package.json ] && grep -q '"test"' package.json; then
  echo "✓ Running tests..."
  npm test -- --passWithNoTests 2>/dev/null || FAILED+=("tests")
elif command -v pytest &> /dev/null; then
  echo "✓ Running Python tests..."
  pytest -x --tb=short 2>/dev/null || FAILED+=("tests")
else
  SKIPPED+=("tests")
fi

# Build/smoke
if [ -f package.json ] && grep -q '"build"' package.json; then
  echo "✓ Running build..."
  npm run build || FAILED+=("build")
elif [ -f Makefile ] && grep -q "^build:" Makefile; then
  echo "✓ Running build..."
  make build || FAILED+=("build")
else
  SKIPPED+=("build")
fi

# Summary
echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "✓ All checks passed"
  if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "Skipped (not available): ${SKIPPED[*]}"
  fi
  exit 0
else
  echo "✗ Failed checks: ${FAILED[*]}"
  if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "Skipped (not available): ${SKIPPED[*]}"
  fi
  exit 1
fi
