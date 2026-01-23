---
description: Dependency audit and vulnerability scanning
---

# Audit Command

## Usage

```
/audit [--update]
```

Optional `--update` flag: If provided, propose safe update paths for outdated packages (requires explicit user confirmation before applying).

## Inputs / Assumptions

- Detect package ecosystem via manifests:
  - Node.js: `@package.json`, `@package-lock.json`, `@yarn.lock`
  - Python: `@requirements.txt`, `@pyproject.toml`, `@Pipfile`
  - Rust: `@Cargo.toml`, `@Cargo.lock`
  - Go: `@go.mod`, `@go.sum`
- Check for audit tools availability before running

## Steps

1. **Detect package manifest**
   - Check for `@package.json`: `!test -f package.json && echo "found"`
   - Check for `@pyproject.toml`: `!test -f pyproject.toml && echo "found"`
   - Check for `@Cargo.toml`: `!test -f Cargo.toml && echo "found"`
   - Check for `@go.mod`: `!test -f go.mod && echo "found"`
   - Report detected ecosystem

2. **Check audit tool availability**
   - Node.js: Check `npm audit` availability: `!npm audit --version 2>&1`
   - Python: Check `pip-audit` availability: `!pip-audit --version 2>&1`
   - Rust: Check `cargo audit` availability: `!cargo audit --version 2>&1`
   - Go: Check `govulncheck` availability: `!govulncheck -version 2>&1`
   - If tool missing, document installation instructions without installing

3. **Run vulnerability scan**
   - Node.js: `!npm audit --json` (parse JSON output)
   - Python: `!pip-audit` (parse output)
   - Rust: `!cargo audit --json` (parse JSON output)
   - Go: `!govulncheck ./...` (parse output)
   - Parse results and categorize by severity

4. **Check for outdated packages**
   - Node.js: `!npm outdated --json` (parse JSON)
   - Python: `!pip list --outdated --format=json` (parse JSON)
   - Rust: Check `@Cargo.toml` against latest versions (manual or tool)
   - Go: `!go list -u -m all` (parse output)
   - Identify packages with available updates

5. **Prioritize findings**
   - **Critical**: Remote code execution, authentication bypass, data exposure
   - **High**: Privilege escalation, denial of service, data corruption
   - **Medium**: Information disclosure, limited scope vulnerabilities
   - **Low**: Best practice violations, minor security improvements
   - Group by package and severity

6. **Generate audit report**
   - Summary: Total vulnerabilities by severity
   - Detailed findings: Package, version, vulnerability ID, severity, description, affected versions, fixed versions
   - Outdated packages: Current version, latest version, update type (patch/minor/major)

7. **Propose safe update paths** (if `--update` flag provided)
   - For each vulnerability: Identify minimal safe update path
   - Respect dependency discipline from `@CLAUDE.md`:
     - Prefer patch/minor over major
     - State why update is needed (1-2 lines)
     - Specify exact packages/versions
     - Keep changes minimal
   - Present update plan to user
   - **Require explicit confirmation** before applying any updates
   - If user confirms: Apply updates incrementally, run tests after each major update

## Outputs

- Audit report with:
  - Vulnerability summary (counts by severity)
  - Detailed vulnerability list with CVE IDs, descriptions, affected packages
  - Outdated packages list
  - Safe update recommendations (if `--update` flag provided)
  - Update plan with rationale (if confirmed)

## Struggle Tracking (When to Document)

Document in `@docs/POSTMORTEMS.md` when encountering:

- **Missing tools**: Audit tools not installed or unavailable, requiring manual installation
- **Conflicting dependency constraints**: Dependency updates blocked by version conflicts requiring resolution
- **Unclear update paths**: Multiple valid update paths with unclear trade-offs requiring investigation
- **Tool failures**: Audit tools fail to run or produce incorrect results requiring workarounds

**Criteria**: 3+ retries OR >15min resolution time OR workaround required

**Category**: DEPENDENCY

## Post-Mortem Write Path

Before documenting a struggle:

1. Check `@docs/POSTMORTEMS.md` for similar entries using:
   - `!rg -n -i "audit\|vulnerability\|dependency" docs/POSTMORTEMS.md`
   - Search by category: `!rg -n "DEPENDENCY" docs/POSTMORTEMS.md`

2. If duplicate found:
   - Ask user: "Similar entry found: [entry]. Update existing or add new?"
   - If update: Modify existing entry with additional context
   - If new: Add new entry (newest first)

3. If new entry:
   - Append to `@docs/POSTMORTEMS.md` in the "New Entries" section
   - Use standard format: date, category (DEPENDENCY), title, task, struggle, resolution, time lost, prevention
   - Commit with message: `Document DEPENDENCY struggle in POSTMORTEMS.md`
