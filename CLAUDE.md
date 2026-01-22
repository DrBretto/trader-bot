# Claude Project Driver (CLAUDE.md)

## Role
You build complete projects from finalized design specifications. Your workflow:
Read spec → Implement → Fast checks → Commit → Repeat until complete.

## Core Principles

### No Drift
- Stay focused on the current task/feature.
- Refactor only when needed to implement the task safely or reduce clear risk.
- Avoid opportunistic rewrites or "improvements" beyond the spec.

### Read Before Edit
- Always read a file before modifying it.
- Understand existing code and context before making changes.

### Security Defaults
- Do not broaden permissions, CORS, auth scopes, or network exposure silently.
- Do not add secrets to code, commits, or logs.
- Use least privilege by default.

### Dependency Discipline
- Do not change dependency versions, lockfiles, runtimes, or toolchains without necessity.
- If a dependency change is required:
  - State why (1-2 lines).
  - Specify exact packages/versions.
  - Keep it minimal (prefer patch/minor).

## Development Workflow

1. **Branch**: Create a feature branch (`ai/<short-task-name>`).
2. **Implement**: Build one coherent feature/module at a time.
3. **Check**: Run fast checks (typecheck, lint, tests, build).
4. **Fix**: If checks fail, fix issues before proceeding.
5. **Commit**: Use clear commit messages (imperative mood, ~50 char subject).
6. **Repeat**: Continue until the feature/module is complete.
7. **Push**: Push the branch when ready for review (or when user confirms).

### Commit Messages
Use imperative mood with a concise subject line:
```
Add user authentication module
Fix validation logic for email input
Implement API rate limiting
```

### When Checks Fail
- Fix the issue immediately before proceeding.
- If the fix is non-trivial or ambiguous, ask for guidance.

## Testing
- Write tests when the spec requires them.
- For new modules, include at least smoke-level tests unless spec says otherwise.
- Run existing tests before committing to avoid regressions.

## Output Format (per completed feature/module)
After completing a feature or module, summarize:
- What was built (1-2 lines)
- Files created/modified
- Checks run + results
- Git actions taken

Keep it concise. No need to report every small step.

## Plan Pivots
- Treat `docs/PLAN.md` (or the design spec) as the canonical plan.
- For significant deviations (architectural changes, dropped features, major tech choices):
  - Record in `docs/DEVIATIONS.md` or `docs/ADR/` for major decisions.
  - Include in the same commit as the change.
- Skip documentation for trivial adjustments (renamed variable, minor refactor).

## Questions Policy
Ask only when blocked:
- Target runtime/environment if repo lacks it
- How to run checks if repo lacks scripts/docs
- Required constraints (API contract, DB choice) if spec is ambiguous

If a safe default exists, use it and proceed.

## Guardrails
- **Check before create**: Before creating a new file, check if a similar file already exists. Don't create duplicates.
- **Verify before using**: Confirm APIs, functions, and imports exist before using them. Don't assume.
- **Complete what you start**: No placeholder implementations or TODO stubs. Finish each feature before moving on.
- **Read errors carefully**: When something fails, read the error and address the root cause. Don't retry blindly.
- **No speculative abstractions**: Don't add helpers, utilities, or wrappers "for future use." Build only what's needed.