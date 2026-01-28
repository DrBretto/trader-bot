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

## Claude Code Commands

This template includes 10 Claude Code commands (markdown instruction files in `.claude/commands/`) that become slash commands in Claude Code CLI:

- **`/deploy`** - Complete deployment pipeline with retry logic and health checks
  - *When to use*: Deploying to staging/production, verifying deployments
  - *Usage*: `/deploy [environment]`

- **`/review`** - Comprehensive code review of changes
  - *When to use*: Before committing, reviewing PRs, checking code quality
  - *Usage*: `/review [file1] [file2] ...`

- **`/audit`** - Dependency audit and vulnerability scanning
  - *When to use*: Checking for security vulnerabilities, outdated packages
  - *Usage*: `/audit [--update]`

- **`/test-coverage`** - Analyze test coverage and identify gaps
  - *When to use*: Assessing test coverage, finding untested code paths
  - *Usage*: `/test-coverage [directory]`

- **`/security-scan`** - Security vulnerability scanner
  - *When to use*: Scanning for secrets, insecure patterns, dependency vulnerabilities
  - *Usage*: `/security-scan [--git-history]`

- **`/refactor`** - Code smell detection and refactoring assistant
  - *When to use*: Identifying code smells, suggesting safe refactorings
  - *Usage*: `/refactor [file1] [file2] ... [directory]`

- **`/docs`** - Auto-documentation generator
  - *When to use*: Generating API docs, updating README, creating ADRs
  - *Usage*: `/docs [target] [format]` (target: README/API/ADR)

- **`/migrate`** - Migration helper for dependency/framework upgrades
  - *When to use*: Upgrading dependencies, framework migrations
  - *Usage*: `/migrate [target] [--dry-run]` (e.g., `/migrate react@18`)

- **`/performance`** - Performance profiling and optimization
  - *When to use*: Analyzing bundle sizes, profiling runtime performance
  - *Usage*: `/performance [target]` (target: bundle/runtime/build)

- **`/postmortem`** - Manually add post-mortem entry
  - *When to use*: Documenting significant struggles after the fact
  - *Usage*: `/postmortem [category] [title] [description]`
  - *Example*: `/postmortem DEPENDENCY "Pandas Lambda size" "Standard pandas too large for Lambda"`

### Post-Mortem System

The post-mortem system tracks significant struggles to identify patterns and improve workflows:

- **Purpose**: Document struggles that meet criteria (3+ retries, >15min resolution, workarounds needed)
- **Location**: `docs/POSTMORTEMS.md`
- **When entries are added**: Automatically by commands when struggles are detected, or manually via `/postmortem`
- **Review process**: Monthly review to identify patterns and codify into `CLAUDE.md` or `.cursorrules`

See [`docs/POSTMORTEMS.md`](docs/POSTMORTEMS.md) for documented struggles and entry format.

## Workflow Principles

This template enforces:
- **Dependency discipline**: No random version upgrades; minimal, justified changes only
- **Fast checks**: Run typecheck, lint, tests, build before moving on
- **Feature branches**: Never push directly to `main`; commit + push after each coherent feature/fix
- **Security defaults**: Never widen auth/CORS/permissions silently; never commit secrets
- **Documentation sync**: Track meaningful plan pivots in `docs/DEVIATIONS.md`

See `CLAUDE.md` and `.cursorrules` for detailed workflow rules.
# solo-template
# trader-bot
