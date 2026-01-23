---
description: Analyze test coverage and identify gaps
---

# Test Coverage Command

## Usage

```
/test-coverage [directory]
```

Optional `directory` argument specifies which directory to analyze. If omitted, analyze entire project.

## Inputs / Assumptions

- Detect test framework:
  - JavaScript/TypeScript: Jest, Mocha, Vitest (check `@package.json`)
  - Python: pytest, unittest (check `@pyproject.toml` or test files)
  - Rust: built-in `cargo test` (check `@Cargo.toml`)
  - Go: built-in `go test` (check for `*_test.go` files)
- Detect coverage tool:
  - JavaScript: `c8`, `nyc`, `jest --coverage`
  - Python: `coverage.py`, `pytest-cov`
  - Rust: `cargo-tarpaulin`, `cargo-llvm-cov`
  - Go: `go test -cover`

## Steps

1. **Detect test framework and coverage tool**
   - Check `@package.json` for test scripts and coverage tools
   - Check for coverage config files: `.nycrc`, `.coveragerc`, `coverage.toml`
   - Check test file patterns: `**/*.test.js`, `**/*_test.py`, `**/*_test.rs`, `**/*_test.go`
   - Report detected framework and tool

2. **Run coverage analysis**
   - If coverage tool available:
     - JavaScript: `!npm run test:coverage` or `!npx jest --coverage`
     - Python: `!coverage run -m pytest` then `!coverage report`
     - Rust: `!cargo tarpaulin --out stdout` or `!cargo llvm-cov --html`
     - Go: `!go test -coverprofile=coverage.out ./...` then `!go tool cover -html=coverage.out`
   - If coverage report exists: Read existing report files (e.g., `coverage/lcov.info`, `coverage.xml`, `coverage.json`)
   - Parse coverage data: file-level and line-level coverage percentages

3. **Identify untested code paths**
   - List files with 0% coverage
   - List files with <50% coverage (configurable threshold)
   - Identify specific functions/methods with no coverage
   - Identify critical paths (error handlers, authentication, data validation) with low/no coverage

4. **Highlight critical paths without coverage**
   - Security-sensitive code (auth, encryption, input validation)
   - Error handling paths
   - Edge cases and boundary conditions
   - Integration points (API endpoints, database queries)
   - Business logic core functions

5. **Suggest high-value test additions**
   - Prioritize tests by:
     - Security impact
     - Business logic importance
     - User-facing functionality
     - Error handling
   - For each suggestion:
     - File and function to test
     - Test type (unit/integration/e2e)
     - Test cases to cover
     - Estimated effort

6. **Generate coverage summary**
   - Overall coverage percentage
   - Coverage by file/directory
   - Coverage by file type (source vs test)
   - Critical paths coverage status
   - Recommendations prioritized by impact

## Outputs

- Coverage report with:
  - Overall coverage statistics
  - File-by-file coverage breakdown
  - List of untested files and functions
  - Critical paths coverage analysis
  - Prioritized test addition recommendations
  - Coverage trends (if historical data available)

## Struggle Tracking (When to Document)

Document in `@docs/POSTMORTEMS.md` when encountering:

- **Coverage tool failures**: Coverage tools fail to run or produce incorrect results
- **Configuration issues**: Coverage tool misconfigured requiring investigation
- **Unclear test infrastructure**: Test framework not detected or ambiguous setup requiring clarification
- **Coverage report parsing failures**: Unable to parse coverage output requiring manual analysis

**Criteria**: 3+ retries OR >15min resolution time OR workaround required

**Category**: TEST

## Post-Mortem Write Path

Before documenting a struggle:

1. Check `@docs/POSTMORTEMS.md` for similar entries using:
   - `!rg -n -i "coverage\|test" docs/POSTMORTEMS.md`
   - Search by category: `!rg -n "TEST" docs/POSTMORTEMS.md`

2. If duplicate found:
   - Ask user: "Similar entry found: [entry]. Update existing or add new?"
   - If update: Modify existing entry with additional context
   - If new: Add new entry (newest first)

3. If new entry:
   - Append to `@docs/POSTMORTEMS.md` in the "New Entries" section
   - Use standard format: date, category (TEST), title, task, struggle, resolution, time lost, prevention
   - Commit with message: `Document TEST struggle in POSTMORTEMS.md`
