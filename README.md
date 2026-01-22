# Project Template

This is a reusable project template for bootstrapping new repositories with consistent workflow rules, documentation structure, and automation.

## Quick Start

1. Copy this template folder into your new project repository
2. Review and customize the following files:
   - `CLAUDE.md` - Project driver instructions for AI assistants
   - `.cursorrules` - Cursor IDE rules (also configure in Cursor settings for portability)
   - `docs/PLAN.md` - Fill in your project goals, constraints, and architecture
   - `.env.example` - Add your environment variable placeholders
3. Run `make setup` to initialize your environment (or `make check` to verify setup)

## Key Files

### Workflow & Rules
- **`CLAUDE.md`** - Instructions for AI-driven development workflow
- **`.cursorrules`** - Cursor IDE rules (project-agnostic workflow constraints)
- **`CONTRIBUTING.md`** - Contribution guidelines (references CLAUDE.md and .cursorrules)

### Documentation
- **`docs/PLAN.md`** - Project plan template (goals, constraints, architecture, milestones)
- **`docs/DEVIATIONS.md`** - Log of plan pivots and deviations from the original plan
- **`docs/ADR/`** - Architecture Decision Records (optional, for major decisions)

### Configuration
- **`.gitignore`** - General ignore patterns (OS, editors, build artifacts, Node/Python)
- **`.editorconfig`** - Cross-editor formatting defaults
- **`.env.example`** - Environment variable template

### Automation
- **`Makefile`** - Common task shortcuts (`make check`, `make setup`, `make help`)
- **`scripts/check.sh`** - Fast-check runner (detects and runs typecheck, lint, tests, build)
- **`.github/workflows/ci.yml`** - CI workflow that runs fast checks

### Optional
- **`LICENSE`** - Choose and specify your license
- **`CODEOWNERS`** - Define code ownership (GitHub feature)

## Workflow Principles

This template enforces:
- **Dependency discipline**: No random version upgrades; minimal, justified changes only
- **Fast checks**: Run typecheck, lint, tests, build before moving on
- **Feature branches**: Never push directly to `main`; commit + push after each coherent feature/fix
- **Security defaults**: Never widen auth/CORS/permissions silently; never commit secrets
- **Documentation sync**: Track meaningful plan pivots in `docs/DEVIATIONS.md`

See `CLAUDE.md` and `.cursorrules` for detailed workflow rules.
# solo-template
