# Post-Mortems

This file documents significant struggles encountered during development to identify patterns and improve workflows.

## Purpose

Post-mortems are documented here when:
- **3+ retry attempts** on the same operation
- **>15 minutes** resolution time
- **Workaround required** for tooling/dependency/environment issues

## Review Cadence

Review this file **monthly** to identify patterns. When patterns emerge, codify them into `@CLAUDE.md` or `@.cursorrules` as preventive rules.

## Entry Format

Each entry follows this structure:

```markdown
## [Date] - [Category] - [Title]

**Task**: [What were you trying to accomplish?]
**Struggle**: [What went wrong? What errors did you encounter?]
**Resolution**: [How did you resolve it? What was the fix/workaround?]
**Time Lost**: [How much time was spent resolving this?]
**Prevention**: [How can this be prevented in the future? What should be added to CLAUDE.md or .cursorrules?]
```

## Categories

- `DEPENDENCY` - Dependency/package management issues
- `TOOLING` - Tool configuration or availability issues
- `CONFIG` - Configuration/environment issues
- `AWS` - AWS-specific issues
- `BUILD` - Build/compilation issues
- `TEST` - Testing framework or test execution issues
- `DEPLOY` - Deployment issues
- `UNDERSTANDING` - Code understanding or pattern recognition issues

## Example Entry

**DELETE AFTER FIRST REAL ENTRY**

## 2025-01-23 - DEPENDENCY - Pandas Lambda size

**Task**: Deploy Python Lambda function with pandas dependency
**Struggle**: Standard pandas package too large for Lambda deployment package size limits. Deployment failed with "Package size exceeds 50MB" error.
**Resolution**: Used pandas layer from AWS or switched to lighter alternative (pandas-lite, pyarrow). Created separate Lambda layer for pandas.
**Time Lost**: 45 minutes
**Prevention**: Check package sizes before Lambda deployment. Document large dependencies that require layers. Add to CLAUDE.md: "For Lambda deployments, check package sizes and use layers for dependencies >10MB."

---

## New Entries

<!-- Add new entries below, newest first -->
