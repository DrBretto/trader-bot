# Plan Deviations

This file logs meaningful pivots and deviations from the original plan in `PLAN.md`.

## Entry Template

```markdown
## [Date] - [Brief Description]

**PLAN Reference**: Section/requirement from PLAN.md
**What Changed**: Brief description of the deviation
**Why**: Rationale for the change
**Impact**: Effect on goals, timeline, or architecture
**Follow-up**: Any required actions or decisions
```

---

## 2026-01-28 - Directory Renamed from lambda/ to src/

**PLAN Reference**: Project File Structure section specifying `lambda/` directory
**What Changed**: Renamed the `lambda/` directory to `src/` throughout the project
**Why**: Python's `lambda` is a reserved keyword, causing syntax errors when importing modules. Example error: `from lambda.models import ... SyntaxError: invalid syntax`
**Impact**:
- All import statements use `src.` prefix instead of `lambda.`
- Lambda handler path changed to `src.handler.lambda_handler`
- Infrastructure scripts updated to copy `src/` instead of `lambda/`
- No functional impact - all tests pass (44/44)
**Follow-up**: Update spec documentation if future phases reference `lambda/` directory
