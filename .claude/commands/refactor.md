---
description: Code smell detection and refactoring assistant
---

# Refactor Command

## Usage

```
/refactor [file1] [file2] ... [directory]
```

Review specified files or directories for code smells and suggest refactorings. **By default, only suggests changes. Does not apply refactorings unless user explicitly confirms.**

## Inputs / Assumptions

- Analyze specified files using `@file1`, `@file2`, etc.
- If directory provided, analyze all files in directory recursively
- Respect "No Drift" principle from `@CLAUDE.md`: Refactor only when it directly reduces bugs, improves clarity, or enables requested change
- Reference `@CLAUDE.md` and `@.cursorrules` for refactoring guidelines

## Steps

1. **Analyze specified files or directories**
   - If files provided: Read each file using `@` syntax
   - If directory provided: List files recursively and analyze each
   - If no arguments: Ask user which files/directories to analyze

2. **Identify code smells**
   - **Duplication**:
     - Repeated code blocks (similar logic in multiple places)
     - Copy-paste patterns
     - Similar function implementations
   - **Complexity**:
     - High cyclomatic complexity functions
     - Deeply nested conditionals
     - Long functions (>50 lines, configurable)
     - Long parameter lists (>5 parameters)
   - **Anti-patterns**:
     - God objects (classes doing too much)
     - Feature envy (methods accessing other objects excessively)
     - Long method chains
     - Magic numbers/strings
     - Dead code (unused functions, variables)
   - **Maintainability**:
     - Poor naming
     - Missing documentation
     - Tight coupling
     - Low cohesion

3. **Suggest safe refactorings**
   - For each code smell:
     - **Location**: File and line numbers
     - **Issue**: Description of the code smell
     - **Refactoring suggestion**: Specific refactoring technique
     - **Rationale**: Why this refactoring improves code
     - **Risk level**: Low/Medium/High (based on scope and test coverage)
     - **Before/After**: Code examples showing current vs proposed
   - Prioritize suggestions by:
     - Impact on code quality
     - Risk level (prefer low-risk refactorings)
     - Test coverage (safer to refactor well-tested code)

4. **Present suggestions to user**
   - Group by file
   - Show summary: Total suggestions, grouped by risk level
   - For each suggestion: Show before/after code diff
   - **Ask user**: "Apply these refactorings? (yes/no/all/select)"
   - If user says "select": Allow choosing specific refactorings

5. **Apply refactorings** (only if user explicitly confirms)
   - Apply selected refactorings one at a time
   - After each refactoring:
     - Show diff of changes
     - Run fast checks: `!make check` or equivalent
     - If checks fail:
       - Revert the refactoring
       - Report failure to user
       - Document in post-mortem if 3+ refactorings fail (category: BUILD or UNDERSTANDING)

6. **Generate before/after comparison**
   - Summary of changes:
     - Files modified
     - Refactorings applied
     - Metrics improved (if measurable): complexity reduction, duplication reduction
   - Show git diff: `!git diff` to display all changes
   - Confirm tests still pass: `!npm test` or equivalent

## Outputs

- Code smell analysis report:
  - Summary statistics (duplication, complexity, anti-patterns counts)
  - Detailed findings with locations
  - Refactoring suggestions with before/after examples
- If applied: Refactoring summary with before/after comparison and test results

## Struggle Tracking (When to Document)

Document in `@docs/POSTMORTEMS.md` when encountering:

- **Unclear safe refactoring paths**: Multiple valid refactoring approaches with unclear trade-offs requiring investigation
- **Test failures after refactoring**: Refactorings break tests requiring rollback or additional fixes
- **Ambiguous code patterns**: Code patterns that are unclear whether they represent code smells or acceptable patterns
- **Refactoring tool failures**: Automated refactoring tools fail or produce incorrect results

**Criteria**: 3+ retries OR >15min resolution time OR workaround required

**Category**: UNDERSTANDING (or BUILD if test failures)

## Post-Mortem Write Path

Before documenting a struggle:

1. Check `@docs/POSTMORTEMS.md` for similar entries using:
   - `!rg -n -i "refactor\|code smell" docs/POSTMORTEMS.md`
   - Search by category: `!rg -n "UNDERSTANDING\|BUILD" docs/POSTMORTEMS.md`

2. If duplicate found:
   - Ask user: "Similar entry found: [entry]. Update existing or add new?"
   - If update: Modify existing entry with additional context
   - If new: Add new entry (newest first)

3. If new entry:
   - Append to `@docs/POSTMORTEMS.md` in the "New Entries" section
   - Use standard format: date, category (UNDERSTANDING or BUILD), title, task, struggle, resolution, time lost, prevention
   - Commit with message: `Document [category] struggle in POSTMORTEMS.md`
