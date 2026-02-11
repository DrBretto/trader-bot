# Claude Project Driver (CLAUDE.md)

## Role

You build and maintain solo projects. Your workflow:
Orient → Plan → Implement → Check → Commit → Deploy → Repeat until complete.

## Session Start

At the beginning of every session, before doing any work:

1. Read `docs/PLAN.md` to understand the current project roadmap and status.
2. Check `docs/plans/` for any in-progress plan documents. If one exists for the current or most recent task, read it and restate where things left off.
3. Skim `docs/POSTMORTEMS.md` for known pitfalls relevant to the current task area.
4. If the task involves deployment, read `docs/DEPLOY.md`.
5. Only after orienting, proceed with the task.

If any of these files don't exist yet, note it and proceed — don't block on missing docs.

## Core Principles

### No Drift

- Stay focused on the current task/feature.
- Refactor only when needed to implement the task safely or reduce clear risk.
- Avoid opportunistic rewrites or "improvements" beyond the spec.

### Read Before Edit

- Always read a file before modifying it.
- Understand existing code and context before making changes.

### Planning, Canonical Docs & Session Resilience

- The project repo is the **single source of truth** for plans, decisions, and change history. Do not rely on chat history as the only record.
- Before making any non-trivial change (new feature, bugfix spanning multiple files, or any refactor), you MUST:
  - Create or update a **plan document** under `docs/plans/` named `YYYY-MM-DD-<short-task-name>.md`.
  - At minimum, include these sections in that file:
    - `## Context` – what problem or goal is being addressed.
    - `## Plan` – bullet list of concrete steps and target files/modules.
    - `## Execution Log` – short notes as steps are completed or the approach changes.
    - `## Follow-ups` – known TODOs or deferred work.
  - One plan doc per task/feature. If the scope evolves within the same session, update the existing doc. Create a new doc only for a genuinely different task.
- Keep the plan document in sync with reality:
  - When you finish a step, mark it done in `## Plan` and add a brief note to `## Execution Log`.
  - If you change direction, append the new approach and why.
- When a session is interrupted or lost:
  - On resume, first read the relevant `docs/plans/` file.
  - Restate the current `## Plan`, indicate which items are already complete, and continue from there.
- For significant deviations from the original design (architectural changes, dropped features, major tech choices):
  - Record in `docs/DEVIATIONS.md` or `docs/ADR/` for major decisions.
  - Include in the same commit as the change.
- No significant code change or refactor should be started or committed unless there is an up-to-date corresponding plan in `docs/plans/` and, where appropriate, the high-level roadmap in `docs/PLAN.md` has been adjusted.
- **Exception**: For single-file, single-line fixes (typos, obvious one-liner bugs), a plan document is not required — just commit with a clear message.

### End-to-End Execution

- When given a task, execute the full cycle: plan → implement → check → fix → commit → push → deploy (if applicable).
- Do not stop to ask for permission at intermediate steps unless genuinely blocked or facing a destructive/irreversible action.
- **Natural checkpoints** (ask before proceeding):
  - **Before deploying**: "Feature is implemented and tests pass. Ready to deploy?"
  - **Before merging to main**: "Everything is deployed and verified. Ready to merge?"
- **Destructive actions** (always ask first):
  - Deleting production data or dropping database tables.
  - Force-pushing or rewriting git history.
  - Changing IAM policies, security groups, or auth scopes that affect other services.
  - Spinning up new persistent infrastructure that incurs ongoing cost (RDS, NAT gateways, etc.).
- Normal deploys following `docs/DEPLOY.md` are **not** destructive — proceed after the deploy checkpoint.

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

### AWS Cost Control

- **NEVER enable S3 versioning** — always keep it off/suspended.
- Avoid features that silently accumulate costs (versioning, cross-region replication, etc.).
- Use lifecycle policies to auto-delete old data when appropriate.
- Prefer spot instances and minimal resource allocation.

### AWS Lambda Constraints

- Treat Lambda functions as "thin":
  - No heavy data-science stacks in request handlers (`pandas`, `numpy`, `scipy`, `torch`, `tensorflow`, etc.).
  - If such deps are required, propose:
    - A separate containerized worker (ECS/Fargate/Batch), or
    - A dedicated Lambda with its own layer, invoked asynchronously.
- Prefer small, fast handlers that orchestrate work rather than performing heavy computation.

## Development Workflow

1. **Plan**: Write or update the plan document in `docs/plans/` before coding.
2. **Branch**: Create a feature branch (`ai/<short-task-name>`).
3. **Implement**: Build one coherent feature/module at a time.
4. **Check**: Run fast checks (typecheck, lint, tests, build).
5. **Fix**: If checks fail, fix issues before proceeding.
6. **Commit**: Commit with clear messages. Commit regularly — after each coherent unit of work, not just at the end.
7. **Push**: Push the feature branch when checks pass. Don't wait for permission to push feature branches.
8. **Deploy**: When implementation and tests are complete, ask the user before deploying. Follow `docs/DEPLOY.md` exactly. Never guess deploy targets.
9. **Merge**: After successful deployment and verification, ask the user before merging to main.
10. **Update docs**: Mark the plan document as complete. Update `docs/PLAN.md` if the roadmap has changed.

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

## Struggle Documentation

- Treat `docs/POSTMORTEMS.md` as a learning log for repeated or significant struggles.
- Document when you encounter:
  - **3+ retry attempts** on the same operation
  - Known problematic patterns (check existing `docs/POSTMORTEMS.md` first)
  - Workarounds required for tooling/dependency/environment issues
- Skip documentation for:
  - First-time errors that resolve immediately
  - User clarification on ambiguous requirements (normal workflow)
  - Expected failures like failing tests you're actively fixing

### Post-Mortem Entry Format

```markdown
## [Date] - [Category] - [Title]

**Task**: [What were you trying to accomplish?]
**Struggle**: [What went wrong? What errors did you encounter?]
**Resolution**: [How did you resolve it? What was the fix/workaround?]
**Retry Count**: [How many attempts before resolution?]
**Prevention**: [How can this be prevented in the future?]
```

**Categories**: `DEPENDENCY`, `TOOLING`, `CONFIG`, `AWS`, `BUILD`, `TEST`, `DEPLOY`, `UNDERSTANDING`

### Integration with Workflow

- Document struggles **as they occur**, not at the end of a session
- Check existing entries before adding (avoid duplicates; update existing if same pattern)
- Include post-mortem commit message: `Document [category] struggle in POSTMORTEMS.md`
- Review `docs/POSTMORTEMS.md` periodically to identify patterns and codify into `CLAUDE.md` or `.cursorrules`

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
- **Read before deploy**: Before running any deploy, sync, or infrastructure command, read `docs/DEPLOY.md` to verify the correct targets. Never guess deploy targets from AWS resource listings.
