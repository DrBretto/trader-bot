---
description: Migration helper for dependency/framework upgrades
---

# Migrate Command

## Usage

```
/migrate [target] [--dry-run]
```

- `target`: Migration target (e.g., `react@18`, `python@3.11`, `nextjs@14`)
- `--dry-run`: If provided, generate migration plan without applying changes

## Inputs / Assumptions

- Parse migration target from `$ARGUMENTS` (package name and version)
- Detect current versions from package manifests
- Respect dependency discipline from `@CLAUDE.md`: minimal changes, state rationale, require confirmation
- Check for breaking changes in changelog or migration guides

## Steps

1. **Identify migration target**
   - Parse `$ARGUMENTS` to extract:
     - Package/framework name (e.g., `react`, `nextjs`, `python`)
     - Target version (e.g., `18`, `14`, `3.11`)
   - If `$ARGUMENTS` not provided: Prompt user for migration target
   - Detect current version:
     - Read `@package.json` for Node.js packages
     - Read `@pyproject.toml` or `@requirements.txt` for Python
     - Read `@Cargo.toml` for Rust
   - Report current vs target version

2. **Generate migration plan**
   - Research breaking changes:
     - Check for official migration guide (e.g., `MIGRATION.md`, `CHANGELOG.md` in package repo)
     - If available: Read migration guide and extract key steps
     - If not available: Check changelog for breaking changes between current and target version
   - Identify migration steps:
     - Dependency version updates
     - Code changes required (API changes, deprecated features)
     - Configuration changes
     - Test updates
   - Estimate effort and risk level for each step

3. **Create backup strategy**
   - Suggest creating backup branch: `!git checkout -b backup/pre-migration-$(date +%Y%m%d)`
   - Recommend committing current state: `!git add -A && git commit -m "Pre-migration checkpoint"`
   - If user confirms: Create backup branch

4. **Apply migration steps incrementally** (unless `--dry-run` flag)
   - **Step 1: Update dependencies**
     - Update package manifest with target version
     - State rationale: "Upgrading [package] to [version] for [reason]"
     - If `--dry-run`: Show what would change without applying
     - If not dry-run: Apply update, run `!npm install` or equivalent
   - **Step 2: Update code for breaking changes**
     - For each breaking change identified:
       - Locate affected code
       - Apply code changes
       - Show diff of changes
   - **Step 3: Update configuration**
     - Update config files if needed
     - Show config changes
   - **Step 4: Run tests after each major step**
     - After dependency update: `!npm test` or equivalent
     - After code changes: `!npm test` or equivalent
     - If tests fail:
       - Analyze failures
       - Fix test code
       - Re-run tests
       - Document in post-mortem if 3+ test failures or >15min to fix (category: TEST)

5. **Document architectural changes** (if significant)
   - If migration involves architectural changes:
     - Record in `@docs/DEVIATIONS.md` following existing pattern
     - Include in same commit as migration
     - Reference migration target and rationale

6. **Final verification**
   - Run full test suite: `!npm test` or equivalent
   - Run build: `!npm run build` or equivalent
   - Run fast checks: `!make check` or equivalent
   - If all pass: Report migration complete
   - If failures: Rollback option or continue fixing

## Outputs

- Migration plan with:
  - Current vs target versions
  - Breaking changes identified
  - Step-by-step migration instructions
  - Risk assessment
  - Backup strategy
- Migration summary (if applied):
  - Files modified
  - Dependencies updated
  - Code changes applied
  - Test results
  - Rollback instructions if needed

## Struggle Tracking (When to Document)

Document in `@docs/POSTMORTEMS.md` when encountering:

- **Unclear breaking changes**: Migration guide or changelog unclear requiring investigation
- **Test failures during migration**: Tests break during migration requiring >15min to fix or 3+ retries
- **Rollback needed**: Migration fails requiring rollback to previous version
- **Dependency conflicts**: Migration blocked by dependency version conflicts requiring resolution

**Criteria**: 3+ retries OR >15min resolution time OR workaround required

**Category**: DEPENDENCY (or TEST if test failures, or BUILD if build failures)

## Post-Mortem Write Path

Before documenting a struggle:

1. Check `@docs/POSTMORTEMS.md` for similar entries using:
   - `!rg -n -i "migrate\|upgrade\|breaking" docs/POSTMORTEMS.md`
   - Search by category: `!rg -n "DEPENDENCY\|TEST\|BUILD" docs/POSTMORTEMS.md`

2. If duplicate found:
   - Ask user: "Similar entry found: [entry]. Update existing or add new?"
   - If update: Modify existing entry with additional context
   - If new: Add new entry (newest first)

3. If new entry:
   - Append to `@docs/POSTMORTEMS.md` in the "New Entries" section
   - Use standard format: date, category (DEPENDENCY/TEST/BUILD), title, task, struggle, resolution, time lost, prevention
   - Commit with message: `Document [category] struggle in POSTMORTEMS.md`
