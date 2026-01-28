---
name: Standard Claude Code Commands & Post-Mortem System (Revised)
overview: Create 9 standard Claude Code commands as markdown files in `.claude/commands/` (deploy, review, audit, test-coverage, security-scan, refactor, docs, migration, performance) and implement an automatic post-mortem documentation system in `docs/POSTMORTEMS.md` with concrete triggers, duplicate prevention, and review processes.
todos:
  - id: create-deploy-command
    content: Create .claude/commands/deploy.md with persistence strategy and struggle detection
    status: pending
  - id: create-review-command
    content: Create .claude/commands/review.md with struggle detection for review failures
    status: pending
  - id: create-audit-command
    content: Create .claude/commands/audit.md with confirmation mechanics and scope limits
    status: pending
  - id: create-test-coverage-command
    content: Create .claude/commands/test-coverage.md with scope definition (analyze vs run)
    status: pending
  - id: create-security-scan-command
    content: Create .claude/commands/security-scan.md with git history scope limits
    status: pending
  - id: create-refactor-command
    content: Create .claude/commands/refactor.md with confirmation mechanics
    status: pending
  - id: create-docs-command
    content: Create .claude/commands/docs.md - Documentation generator command
    status: pending
  - id: create-migrate-command
    content: Create .claude/commands/migrate.md with confirmation and DEVIATIONS.md relationship
    status: pending
  - id: create-performance-command
    content: Create .claude/commands/performance.md with scope definition (profile vs analyze)
    status: pending
  - id: create-postmortem-command
    content: Create .claude/commands/postmortem.md for manual entry
    status: pending
  - id: create-postmortem-search-command
    content: Create .claude/commands/postmortem-search.md for searching past struggles
    status: pending
  - id: create-postmortems-file
    content: Create docs/POSTMORTEMS.md with template, index, and growth management strategy
    status: pending
  - id: create-postmortem-template
    content: Create docs/POSTMORTEM_TEMPLATE.md as reference
    status: pending
  - id: update-claude-md
    content: Update CLAUDE.md with concrete struggle triggers and post-mortem section after Plan Pivots
    status: pending
  - id: update-readme
    content: Update README.md with documentation for all commands and post-mortem system
    status: pending
---

# Standard Claude Code Commands & Post-Mortem System (Revised)

## Overview

Create 9 Claude Code commands as prefab prompts (markdown files) in `.claude/commands/` and implement an automatic post-mortem system with concrete triggers, duplicate prevention, file growth management, and review processes.

## Claude Code Commands Structure

Claude Code commands are **prefab prompts** - markdown files in `.claude/commands/` that become slash commands (e.g., `deploy.md` becomes `/deploy`). These are NOT scripts - they are pure instruction templates that Claude reads and follows.

Each command file should include:

- **YAML frontmatter**: `description` and optional `allowed-tools`
- **Command body**: Step-by-step instructions for Claude to follow using its available tools
- **Arguments**: Support `$ARGUMENTS` or positional variables (`$1`, `$2`) if provided
- **File references**: Use `@` syntax to reference project files
- **Tool usage**: Instructions tell Claude WHEN to use tools (read_file, run_terminal_cmd, etc.), but the command file itself is just a prompt template

## Commands to Create

### 1. `/deploy` (`.claude/commands/deploy.md`)

Complete deployment pipeline that persists until done.

**What the command file should contain (direct instructions Claude will follow):**
- Run fast checks using `make check` or `@scripts/check.sh`
- Detect deployment environment from `.env` or config files
- Deploy to the detected environment
- Verify deployment by checking health endpoints or running smoke tests
- If any step fails, retry after asking user for confirmation
- If verification fails, rollback and ask user
- Keep retrying until successful or user cancels

**Persistence Strategy:**
- "Persist until done" means: keep retrying in current session until successful or user cancels
- No checkpoint/resume across sessions (Claude Code sessions are stateless)
- Document progress in output so user can manually resume if session interrupted

**Struggle Detection Triggers:**
- Document in POSTMORTEMS.md if:
  - Any step requires 3+ retry attempts
  - Total deployment time exceeds 15 minutes
  - Deployment fails after 2+ successful retries of same step
  - Verification fails and rollback is required
- Capture: step name, retry count, error message, time spent, resolution

**Confirmation Mechanics:**
- For retries after failure: pause and ask user "Retry step X? (yes/no)"
- For rollback: pause and ask user "Rollback deployment? (yes/no)"
- Command waits for user response before proceeding

### 2. `/review` (`.claude/commands/review.md`)

Comprehensive code review.

**What the command file should contain:**
- Analyze git diff/changes (or specific files from `$ARGUMENTS` if provided)
- Check for: security issues, code patterns, best practices, potential bugs
- Review against project rules in `@CLAUDE.md` and `@.cursorrules`
- Generate categorized report (security, performance, maintainability, bugs)
- Provide actionable recommendations
- Reference existing lint/typecheck results if available

**Struggle Detection Triggers:**
- Document in POSTMORTEMS.md if:
  - Review takes >10 minutes to complete
  - Unable to parse/understand code structure (3+ files fail analysis)
  - Conflicting rules detected between CLAUDE.md and .cursorrules
  - Review reveals architectural issues requiring DEVIATIONS.md entry
- Capture: files reviewed, analysis failures, rule conflicts, time spent

**No confirmation needed** - review is read-only analysis.

### 3. `/audit` (`.claude/commands/audit.md`)

Dependency audit and vulnerability scanning.

**What the command file should contain:**
- Detect project type and read dependency files (package.json, pyproject.toml, requirements.txt, Cargo.toml)
- Scan for outdated packages and security vulnerabilities
- Suggest safe update paths (respecting dependency discipline from `@CLAUDE.md`)
- Prioritize findings (critical, high, medium, low)
- Generate audit report

**Confirmation Mechanics:**
- For patch/minor updates: propose changes, pause and ask "Apply patch/minor updates? (yes/no)"
- For major updates: only suggest, never auto-apply (requires explicit user approval)
- Command waits for user response before applying updates

**Struggle Detection Triggers:**
- Document in POSTMORTEMS.md if:
  - Audit tool unavailable or fails (npm audit, safety, cargo audit)
  - Dependency resolution conflicts detected
  - Update suggestions conflict with dependency discipline rules
  - Audit takes >5 minutes
- Capture: tool failures, conflicts, time spent, resolution

### 4. `/test-coverage` (`.claude/commands/test-coverage.md`)

Test coverage analysis.

**What the command file should contain:**
- Analyze existing coverage reports (do NOT run tests unless explicitly requested)
- If coverage report exists: read and parse it
- If no coverage report: run coverage tools once to generate report, then analyze
- Detect coverage tools: nyc (Node.js), pytest-cov (Python), tarpaulin (Rust), go test -cover (Go)
- Identify untested code paths
- Generate coverage report with suggestions
- Suggest high-value test additions
- Highlight critical paths without coverage

**Struggle Detection Triggers:**
- Document in POSTMORTEMS.md if:
  - Coverage tool not found or fails
  - Coverage report generation takes >10 minutes
  - Unable to parse coverage report format
  - Coverage below 50% for critical modules
- Capture: tool failures, parse errors, time spent, coverage gaps

### 5. `/security-scan` (`.claude/commands/security-scan.md`)

Security vulnerability scanner.

**What the command file should contain:**
- Scan for hardcoded secrets (API keys, passwords, tokens) in current codebase
- Check for insecure patterns (SQL injection, XSS, CSRF, etc.) in current code
- Run dependency security audits (npm audit, safety, cargo audit)
- **Git history scan: limit to last 50 commits** (to avoid expensive full-history scan)
- Generate security report with severity levels
- Flag issues requiring immediate attention

**Struggle Detection Triggers:**
- Document in POSTMORTEMS.md if:
  - Secrets found in code or git history
  - Security scan tool unavailable or fails
  - Scan takes >15 minutes
  - Critical vulnerabilities detected
- Capture: secrets found (type only, not value), tool failures, time spent, critical issues

### 6. `/refactor` (`.claude/commands/refactor.md`)

Code smell detection and refactoring assistant.

**What the command file should contain:**
- Identify code smells (duplication, complexity, anti-patterns)
- Analyze specific files (from `$ARGUMENTS` if provided) or entire codebase
- Suggest safe refactorings with explanations
- Generate before/after comparison
- Respect "no drift" principle from `@CLAUDE.md` - only refactor when explicitly requested via this command

**Confirmation Mechanics:**
- For each refactoring suggestion: show before/after, pause and ask "Apply this refactoring? (yes/no/skip)"
- For automated refactorings: pause and ask "Apply all safe refactorings? (yes/no)"
- Command waits for user response before applying changes
- After applying: run tests to verify no regressions

**Struggle Detection Triggers:**
- Document in POSTMORTEMS.md if:
  - Refactoring breaks tests (tests fail after refactoring)
  - Refactoring takes >20 minutes
  - Unable to safely refactor due to complexity
  - Refactoring conflicts with existing patterns
- Capture: test failures, complexity issues, time spent, resolution

### 7. `/docs` (`.claude/commands/docs.md`)

Auto-documentation generator.

**What the command file should contain:**
- Generate API docs from code comments
- Update README sections from code structure
- Create ADR templates for major decisions (ask user for decision details)
- Extract and format code examples
- Update inline documentation
- Support Markdown format (primary), HTML if requested

**Struggle Detection Triggers:**
- Document in POSTMORTEMS.md if:
  - Documentation generation takes >15 minutes
  - Unable to parse code comments or structure
  - Conflicts with existing documentation
  - Documentation format issues
- Capture: parse failures, conflicts, time spent, format issues

### 8. `/migrate` (`.claude/commands/migrate.md`)

Migration helper for upgrades.

**What the command file should contain:**
- Guide through dependency upgrades (package/version from `$ARGUMENTS`)
- Framework migration assistance (e.g., React version upgrades)
- Generate migration plan with step-by-step instructions
- Identify breaking changes and required updates
- Create backup/rollback strategy (document steps, don't auto-backup)
- Test after each migration step

**Confirmation Mechanics:**
- For each migration step: show what will change, pause and ask "Proceed with step X? (yes/no)"
- For breaking changes: pause and ask "This is a breaking change. Continue? (yes/no)"
- Command waits for user response before proceeding
- After each step: run tests, then ask user "Tests passed. Continue to next step? (yes/no)" and wait for response

**DEVIATIONS.md Relationship:**
- If migration is significant (major version upgrade, framework change, architectural impact):
  - Document in `@docs/DEVIATIONS.md` using existing template
  - Reference the migration in commit message
  - POSTMORTEMS.md is for struggles during migration, DEVIATIONS.md is for the migration decision itself

**Struggle Detection Triggers:**
- Document in POSTMORTEMS.md if:
  - Migration step fails 2+ times
  - Tests fail after migration step
  - Breaking changes require manual intervention
  - Migration takes >30 minutes total
- Capture: failed steps, test failures, breaking changes, time spent, resolution

### 9. `/performance` (`.claude/commands/performance.md`)

Performance profiling and optimization.

**What the command file should contain:**
- Do static analysis of code for performance issues (do NOT run profilers unless explicitly requested)
- Analyze bundle sizes (if build artifacts exist)
- Analyze load times (if metrics available)
- If specific functions/files provided via `$ARGUMENTS`: run profiler
- Suggest optimizations (code splitting, lazy loading, caching, etc.)
- Generate performance report with metrics
- Compare before/after when applicable
- Focus on high-impact optimizations

**Struggle Detection Triggers:**
- Document in POSTMORTEMS.md if:
  - Performance analysis takes >15 minutes
  - Profiler unavailable or fails
  - Unable to identify bottlenecks
  - Performance issues require architectural changes
- Capture: tool failures, analysis issues, time spent, architectural concerns

### 10. `/postmortem` (`.claude/commands/postmortem.md`) - Optional Helper

Manual post-mortem entry command.

**Scope:**
- Allow user to manually document a struggle after the fact
- Prompt for: task/context, struggle type, description, resolution, impact, prevention
- Append to POSTMORTEMS.md using template
- Check for duplicates before appending (search by struggle type and description keywords)

**Usage:** `/postmortem` or `/postmortem [brief description]`

### 11. `/postmortem-search` (`.claude/commands/postmortem-search.md`) - Optional Helper

Search past struggles.

**Scope:**
- Search POSTMORTEMS.md by keyword (provided via `$ARGUMENTS`)
- Search by struggle type, task context, or description
- Return matching entries with dates
- Help identify repeated patterns

**Usage:** `/postmortem-search [keyword]` or `/postmortem-search --type [type]`

## Post-Mortem System

### Structure

- **Location**: `docs/POSTMORTEMS.md` (single chronological file)
- **Format**: Markdown with structured entries
- **Index**: Summary section at top with struggle types and counts (updated monthly)
- **Template**: `docs/POSTMORTEM_TEMPLATE.md` as reference

### Entry Template

Each post-mortem entry includes:

```markdown
## [YYYY-MM-DD HH:MM] - [Brief Description]

**Task/Context**: What was being worked on
**Struggle Type**: [dependency|tooling|understanding|implementation|configuration|performance|security|other]
**Description**: What went wrong or was difficult
**Resolution**: How it was resolved (or "Unresolved" if still open)
**Impact**: Time lost, workarounds needed, blockers
**Prevention**: Suggestions for avoiding in future
**Related**: Links to related entries or DEVIATIONS.md if applicable
```

### Concrete Automation Triggers

Document struggles in POSTMORTEMS.md when ANY of these occur:

1. **Retry Threshold**: 3+ retry attempts for same operation
2. **Time Threshold**: Operation takes >15 minutes (or command-specific threshold)
3. **Failure Pattern**: Same error occurs 2+ times with different approaches
4. **Tool Unavailability**: Required tool/command not found or fails
5. **Blocking Issue**: Cannot proceed without user intervention or workaround
6. **Rule Conflict**: Conflicting instructions or patterns detected
7. **Unexpected Behavior**: Tool/library behaves differently than expected

**When to Check/Append:**
- Immediately after trigger condition is met
- Before asking user for help/confirmation
- At end of command if any struggles occurred during execution

### Duplicate Prevention Mechanism

Before appending to POSTMORTEMS.md:

1. Read existing POSTMORTEMS.md
2. Search for similar entries using:
   - Struggle type match
   - Keyword matching in description (3+ common words)
   - Same task/context
3. If similar entry found within last 30 days:
   - Append to existing entry as "Reoccurrence" with new date
   - Update prevention suggestions if new resolution found
4. If no similar entry or >30 days old:
   - Append new entry
   - Update index summary if new struggle type

**Search Pattern (instructions for Claude):**
- Instruct Claude to use grep tool to search for keywords: search for "keyword" in docs/POSTMORTEMS.md
- For struggle type: instruct Claude to use grep to search for "Struggle Type.*type" pattern in docs/POSTMORTEMS.md

### File Growth Management Strategy

**Single File Forever:**
- Keep as one chronological file (easier to search, maintain context)
- Add index/summary section at top (updated monthly)
- Archive strategy: None (keep all entries for pattern recognition)

**Index Section Format:**
```markdown
# Post-Mortems Index

*Last updated: YYYY-MM-DD*

## Struggle Types Summary
- dependency: X occurrences
- tooling: Y occurrences
- understanding: Z occurrences
...

## Recent Patterns (Last 30 Days)
- [Pattern description with count]
...

## Resolved Patterns
- [Pattern that was resolved and how]
...
```

**Monthly Review:**
- User manually reviews POSTMORTEMS.md monthly
- Identify patterns and update CLAUDE.md or .cursorrules if needed
- Update index summary
- Mark resolved patterns

### Post-Mortem Review Process

**Review Frequency:** Monthly (user-initiated)

**Review Process:**
1. User reads POSTMORTEMS.md entries from last month
2. Identify repeated patterns (same struggle type, similar descriptions)
3. For resolved patterns: document resolution in index
4. For unresolved patterns: consider updating rules/docs
5. Update index summary

**Pattern Promotion to Rules:**
- If same struggle type occurs 3+ times in 30 days: consider adding to CLAUDE.md
- If tooling issue repeated: consider adding to .cursorrules or project setup docs
- If dependency issue repeated: consider pinning versions or updating dependency discipline rules

**Archival:**
- No archival needed (keep all entries)
- Old entries (>6 months) can be marked as "Historical" in index but remain in file

### Relationship to DEVIATIONS.md

**POSTMORTEMS.md vs DEVIATIONS.md:**
- **POSTMORTEMS.md**: Struggles, failures, blockers during development (operational issues)
- **DEVIATIONS.md**: Intentional plan changes, architectural decisions, design pivots (strategic changes)

**When to use each:**
- POSTMORTEMS.md: "I struggled with X tool/process" or "Y approach failed"
- DEVIATIONS.md: "We decided to change from X to Y architecture" or "We dropped feature Z"

**Cross-references:**
- If struggle leads to deviation: reference DEVIATIONS.md entry in POSTMORTEMS.md
- If deviation caused struggles: reference POSTMORTEMS.md entries in DEVIATIONS.md

## CLAUDE.md Updates

Add new section **"Post-Mortem Documentation"** after "Plan Pivots" section (before "Questions Policy"):

```markdown
## Post-Mortem Documentation

Document struggles automatically in `docs/POSTMORTEMS.md` to identify patterns and improve workflows.

### When to Document

Document a struggle immediately when ANY of these occur:
- 3+ retry attempts for same operation
- Operation takes >15 minutes (or command-specific threshold)
- Same error occurs 2+ times with different approaches
- Required tool/command not found or fails
- Cannot proceed without user intervention
- Conflicting instructions or patterns detected
- Tool/library behaves unexpectedly

### What to Document

For each struggle, capture:
- **Task/Context**: What you were working on
- **Struggle Type**: Category (dependency, tooling, understanding, implementation, configuration, performance, security, other)
- **Description**: What went wrong or was difficult
- **Resolution**: How it was resolved (or "Unresolved")
- **Impact**: Time lost, workarounds needed, blockers
- **Prevention**: Suggestions for avoiding in future

### How to Document

1. Read `@docs/POSTMORTEMS.md` to check for similar entries
2. If similar entry found within 30 days: append as "Reoccurrence" to existing entry
3. If no similar entry: append new entry using template in `@docs/POSTMORTEM_TEMPLATE.md`
4. Update index summary if new struggle type

### Duplicate Prevention

Before appending, search for similar entries by:
- Struggle type match
- Keyword matching in description (3+ common words)
- Same task/context

If similar entry found within 30 days, append to it rather than creating duplicate.

See `@docs/POSTMORTEMS.md` for examples and `@docs/POSTMORTEM_TEMPLATE.md` for template.
```

This section fits naturally after "Plan Pivots" and uses consistent tone/formatting with existing sections.

## Implementation Details

### File Structure

```
.claude/
  └── commands/
      ├── deploy.md (new)
      ├── review.md (new)
      ├── audit.md (new)
      ├── test-coverage.md (new)
      ├── security-scan.md (new)
      ├── refactor.md (new)
      ├── docs.md (new)
      ├── migrate.md (new)
      ├── performance.md (new)
      ├── postmortem.md (new, optional)
      └── postmortem-search.md (new, optional)

docs/
  ├── POSTMORTEMS.md (new, with index)
  ├── POSTMORTEM_TEMPLATE.md (new)
  └── ... (existing files)

CLAUDE.md (update with post-mortem section)
README.md (update with command documentation)
```

### Command File Pattern

Each command file is a **prompt template** - it contains the actual instructions Claude will read and follow. Write them as direct instructions, not meta-instructions.

**Structure:**
- Start with YAML frontmatter containing `description`
- Write step-by-step instructions in second person or imperative mood
- Example: "Run fast checks using `make check`" not "Tell Claude to run fast checks"
- Use `@filename` to reference project files
- Use `$ARGUMENTS` or `$1`, `$2` for command arguments if provided
- Include struggle detection triggers: "If X happens, document in POSTMORTEMS.md"
- Include confirmation steps: "Ask user: 'Retry? (yes/no)' and wait for response"
- Define scope/limits: "Limit git history scan to last 50 commits"
- Follow existing project rules from `@CLAUDE.md` and `@.cursorrules`

**Important**: These are prompt templates - they ARE the instructions Claude follows. Write them as direct instructions, like a recipe or checklist.

### POSTMORTEMS.md Initial Structure

```markdown
# Post-Mortems

This file documents struggles encountered during AI-assisted development to identify patterns and improve workflows.

## Index

*Last updated: YYYY-MM-DD*

### Struggle Types Summary
- dependency: 0 occurrences
- tooling: 0 occurrences
- understanding: 0 occurrences
- implementation: 0 occurrences
- configuration: 0 occurrences
- performance: 0 occurrences
- security: 0 occurrences
- other: 0 occurrences

### Recent Patterns (Last 30 Days)
*None yet*

### Resolved Patterns
*None yet*

---

## Entries

<!-- Entries will be added here chronologically -->
```

### POSTMORTEM_TEMPLATE.md

```markdown
# Post-Mortem Entry Template

Use this template when documenting struggles in POSTMORTEMS.md:

```markdown
## [YYYY-MM-DD HH:MM] - [Brief Description]

**Task/Context**: [What was being worked on]
**Struggle Type**: [dependency|tooling|understanding|implementation|configuration|performance|security|other]
**Description**: [What went wrong or was difficult]
**Resolution**: [How it was resolved, or "Unresolved" if still open]
**Impact**: [Time lost, workarounds needed, blockers]
**Prevention**: [Suggestions for avoiding in future]
**Related**: [Links to related entries or DEVIATIONS.md if applicable]
```

## Struggle Types

- **dependency**: Issues with package versions, dependencies, lockfiles
- **tooling**: Issues with build tools, linters, test runners, etc.
- **understanding**: Misunderstanding requirements, code structure, or patterns
- **implementation**: Difficulties implementing features or fixes
- **configuration**: Issues with project config, environment setup, etc.
- **performance**: Performance issues or optimization difficulties
- **security**: Security concerns or vulnerabilities
- **other**: Other struggles not fitting above categories
```

## README.md Updates

Add section documenting:

- All 9 (or 11 with optional helpers) Claude Code commands
- How to use slash commands (e.g., `/deploy`, `/review`)
- Post-mortem system purpose and process
- Examples of when to use each command
- Link to POSTMORTEMS.md for viewing documented struggles
- Monthly review process

## Considerations

- Commands should be framework-agnostic where possible (detect project type)
- Graceful degradation when tools aren't available
- Clear instructions for Claude to follow
- Respect existing project rules (dependency discipline, no drift, security defaults)
- Post-mortem entries should be concise but informative
- Commands should be reusable across different project types
- Confirmation mechanics must be explicit (pause and wait for user)
- Scope limits must be defined for expensive operations
- Duplicate prevention must be practical (grep-based search)

## Testing Strategy

- Each command should be tested with sample projects (Node.js, Python, Rust)
- Verify struggle detection triggers work correctly
- Verify duplicate prevention works (test with similar entries)
- Verify confirmation mechanics pause correctly
- Verify POSTMORTEMS.md formatting is consistent
- Verify index updates correctly