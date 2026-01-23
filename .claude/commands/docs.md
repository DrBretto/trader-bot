---
description: Auto-documentation generator
---

# Docs Command

## Usage

```
/docs [target] [format]
```

- `target`: Documentation target type
  - `README` - Update README.md based on code structure
  - `API` - Extract API documentation from code comments
  - `ADR` - Create Architecture Decision Record template
  - If omitted, prompt user for target
- `format`: Output format (default: `markdown`)
  - `markdown` - Markdown format (default)
  - `html` - HTML format

## Inputs / Assumptions

- For README: Analyze project structure, dependencies, scripts
- For API docs: Extract from code comments (JSDoc, docstrings, etc.)
- For ADR: Use template from `@docs/ADR/000-template.md`
- Detect project type to apply appropriate documentation patterns

## Steps

1. **Detect documentation target**
   - If `$ARGUMENTS` provided: Use first argument as target (README/API/ADR)
   - If omitted: Prompt user: "What documentation do you want to generate? (README/API/ADR)"
   - Detect format: Use second argument if provided, default to `markdown`

2. **Generate README documentation** (if target is README)
   - Analyze project structure:
     - Read `@package.json`, `@pyproject.toml`, `@Cargo.toml` for project metadata
     - Detect project type and dependencies
     - List key directories and files
   - Extract from existing `@README.md`:
     - Preserve existing sections if present
     - Update outdated information
   - Generate/update sections:
     - Project description (from package manifest)
     - Installation instructions (detect package manager)
     - Usage examples (from code or existing README)
     - Configuration (from `.env.example` or config files)
     - Project structure overview
     - Contributing guidelines (link to `@CONTRIBUTING.md` if exists)
   - Output to `@README.md` (backup existing first if updating)

3. **Generate API documentation** (if target is API)
   - Detect code language and comment style:
     - JavaScript/TypeScript: JSDoc comments (`/** ... */`)
     - Python: Docstrings (triple quotes)
     - Rust: Doc comments (`/// ...` or `//! ...`)
     - Go: Comments before exported functions
   - Extract API documentation:
     - Scan source files for documentation comments
     - Parse function/class signatures
     - Extract parameters, return types, descriptions, examples
   - Generate API docs:
     - Group by module/namespace
     - List functions/classes with signatures
     - Include parameter descriptions
     - Include return type descriptions
     - Include code examples if present in comments
   - Output format:
     - Markdown: Create `docs/API.md` or update existing
     - HTML: Generate HTML file with navigation

4. **Create ADR template** (if target is ADR)
   - Read `@docs/ADR/000-template.md` to get template structure
   - Prompt user for:
     - ADR number (next sequential number, e.g., 001, 002)
     - Title
     - Status (Proposed/Accepted/Deprecated/Superseded)
   - Create new ADR file: `docs/ADR/[number]-[title-slug].md`
   - Fill template with:
     - User-provided title and status
     - Placeholder sections: Context, Decision, Consequences, Alternatives Considered
   - Open file for user to fill in details

5. **Extract and format code examples**
   - For API docs: Extract code examples from comments
   - For README: Extract usage examples from code or tests
   - Format examples with syntax highlighting markers
   - Ensure examples are runnable (if possible)

6. **Output documentation**
   - Save to appropriate location:
     - README: `@README.md`
     - API: `docs/API.md` (or `docs/api.html` if HTML)
     - ADR: `docs/ADR/[number]-[title].md`
   - Display summary of generated documentation
   - Offer to preview or edit

## Outputs

- Generated documentation file(s):
  - README.md (updated)
  - docs/API.md (or HTML)
  - docs/ADR/[number]-[title].md (new ADR)
- Summary of what was generated
- Preview of key sections

## Struggle Tracking (When to Document)

Document in `@docs/POSTMORTEMS.md` when encountering:

- **Insufficient code comments**: Code lacks documentation comments making API extraction difficult
- **Unclear API structure**: Code structure ambiguous making API documentation generation challenging
- **Ambiguous documentation requirements**: User requirements unclear requiring multiple iterations
- **Documentation tool failures**: Documentation generation tools fail or produce incorrect output

**Criteria**: 3+ retries OR >15min resolution time OR workaround required

**Category**: UNDERSTANDING

## Post-Mortem Write Path

Before documenting a struggle:

1. Check `@docs/POSTMORTEMS.md` for similar entries using:
   - `!rg -n -i "docs\|documentation\|api" docs/POSTMORTEMS.md`
   - Search by category: `!rg -n "UNDERSTANDING" docs/POSTMORTEMS.md`

2. If duplicate found:
   - Ask user: "Similar entry found: [entry]. Update existing or add new?"
   - If update: Modify existing entry with additional context
   - If new: Add new entry (newest first)

3. If new entry:
   - Append to `@docs/POSTMORTEMS.md` in the "New Entries" section
   - Use standard format: date, category (UNDERSTANDING), title, task, struggle, resolution, time lost, prevention
   - Commit with message: `Document UNDERSTANDING struggle in POSTMORTEMS.md`
