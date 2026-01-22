# Contributing

This project follows a disciplined workflow focused on correctness, minimal scope drift, and fast iteration.

## Quick Reference

For detailed workflow rules, see:
- **`CLAUDE.md`** - Complete development workflow and AI assistant instructions
- **`.cursorrules`** - Cursor IDE rules (also configured in Cursor settings)

## Key Principles

1. **Work on feature branches** - Never develop directly on `main`
2. **Run fast checks** - Typecheck, lint, tests, build before committing
3. **Commit granularity** - One commit per coherent feature/fix
4. **No dependency drift** - Never upgrade versions "just because"
5. **Security defaults** - Never widen auth/CORS/permissions silently; never commit secrets
6. **Track plan pivots** - Log meaningful deviations in `docs/DEVIATIONS.md`

## Before You Start

1. Read `CLAUDE.md` for the full workflow
2. Ensure `.cursorrules` is configured in your Cursor IDE (or use the file version)
3. Review `docs/PLAN.md` to understand project goals and constraints

## Getting Help

If you need to deviate from the plan or workflow:
- Document architectural decisions in `docs/ADR/`
- Log plan pivots in `docs/DEVIATIONS.md`
- Include rationale and impact in commit messages
