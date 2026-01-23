---
description: Comprehensive code review of changes
---

# Review Command

## Usage

```
/review [file1] [file2] ...
```

If no arguments provided, review `git diff` of uncommitted changes. If arguments provided, review specified files or directories.

## Inputs / Assumptions

- If no arguments: analyze `!git diff` output
- If arguments provided: read specified files using `@file1`, `@file2`, etc.
- Reference project rules: `@CLAUDE.md`, `@.cursorrules`
- Check for existing lint/typecheck results in project

## Steps

1. **Get changes to review**
   - If no `$ARGUMENTS`: Run `!git diff` and `!git diff --cached` to get all uncommitted changes
   - If `$ARGUMENTS` provided: Read each specified file using `@` syntax
   - If directory provided: List files in directory and review each

2. **Check against project rules**
   - Read `@CLAUDE.md` for workflow and coding principles
   - Read `@.cursorrules` for project-specific rules
   - Identify violations of:
     - No drift principle
     - Dependency discipline
     - Security defaults
     - Fast checks requirements
     - Git workflow

3. **Run existing checks (if available)**
   - Check for lint output: `!npm run lint 2>&1` or equivalent
   - Check for typecheck output: `!npm run typecheck 2>&1` or equivalent
   - Check for test results: `!npm test 2>&1` or equivalent (if fast)
   - Reference these results in review report

4. **Analyze code patterns**
   - **Security issues**:
     - Hardcoded secrets or credentials
     - SQL injection vulnerabilities
     - XSS vulnerabilities
     - CSRF vulnerabilities
     - Insecure authentication/authorization patterns
     - CORS misconfigurations
   - **Potential bugs**:
     - Null/undefined access without checks
     - Race conditions
     - Memory leaks
     - Logic errors
     - Edge cases not handled
   - **Code patterns**:
     - Code duplication
     - High cyclomatic complexity
     - Anti-patterns
     - Missing error handling
   - **Best practices**:
     - Naming conventions
     - Code organization
     - Documentation
     - Test coverage gaps

5. **Generate categorized report**
   - **Critical**: Security issues, potential crashes, data loss risks
   - **High**: Bugs likely to cause issues, performance problems
   - **Medium**: Code quality issues, maintainability concerns
   - **Low**: Style suggestions, minor improvements
   - For each finding:
     - Location (file:line)
     - Issue description
     - Suggested fix
     - Rationale

## Outputs

- Categorized review report with:
  - Summary statistics (critical/high/medium/low counts)
  - Detailed findings with file locations
  - Actionable recommendations
  - References to project rules violated
  - Links to existing lint/typecheck results if available

## Struggle Tracking (When to Document)

Document in `@docs/POSTMORTEMS.md` when encountering:

- **Unable to detect project type**: Cannot determine language/framework to apply appropriate review patterns
- **Missing required tools**: Lint/typecheck tools not configured or unavailable
- **Ambiguous code patterns**: Code patterns that are unclear whether they violate rules or are acceptable
- **Review tool failures**: Automated review tools fail to run or produce incorrect results

**Criteria**: 3+ retries OR >15min resolution time OR workaround required

**Category**: UNDERSTANDING

## Post-Mortem Write Path

Before documenting a struggle:

1. Check `@docs/POSTMORTEMS.md` for similar entries using:
   - `!rg -n -i "review\|lint\|typecheck" docs/POSTMORTEMS.md`
   - Search by category: `!rg -n "UNDERSTANDING" docs/POSTMORTEMS.md`

2. If duplicate found:
   - Ask user: "Similar entry found: [entry]. Update existing or add new?"
   - If update: Modify existing entry with additional context
   - If new: Add new entry (newest first)

3. If new entry:
   - Append to `@docs/POSTMORTEMS.md` in the "New Entries" section
   - Use standard format: date, category (UNDERSTANDING), title, task, struggle, resolution, time lost, prevention
   - Commit with message: `Document UNDERSTANDING struggle in POSTMORTEMS.md`
