---
description: Manually add post-mortem entry
---

# Postmortem Command

## Usage

```
/postmortem [category] [title] [description]
```

- `category`: One of: `DEPENDENCY`, `TOOLING`, `CONFIG`, `AWS`, `BUILD`, `TEST`, `DEPLOY`, `UNDERSTANDING`
- `title`: Brief title for the struggle (use quotes if contains spaces)
- `description`: Brief description (use quotes if contains spaces)

Example:
```
/postmortem DEPENDENCY "Pandas Lambda size" "Standard pandas too large for Lambda"
```

## Inputs / Assumptions

- Parse arguments: `$1` = category, `$2` = title, `$3` = description
- Reference `@docs/POSTMORTEMS.md` for entry format and existing entries
- Check for duplicate entries before adding

## Steps

1. **Parse arguments**
   - Extract category from `$1` (validate against allowed categories)
   - Extract title from `$2`
   - Extract description from `$3`
   - If any missing: Prompt user for missing information

2. **Prompt for additional details**
   - **Task context**: What were you trying to accomplish?
   - **Struggle details**: What went wrong? What errors did you encounter?
   - **Resolution**: How did you resolve it? What was the fix/workaround?
   - **Time lost**: How much time was spent resolving this? (e.g., "30 minutes", "2 hours")
   - **Prevention**: How can this be prevented in the future? What should be added to `@CLAUDE.md` or `@.cursorrules`?

3. **Check for duplicates**
   - Search `@docs/POSTMORTEMS.md` for similar entries:
     - By category: `!rg -n "^##.*\[$1\]" docs/POSTMORTEMS.md`
     - By keywords from title: `!rg -n -i "$(echo $2 | tr ' ' '|')" docs/POSTMORTEMS.md`
   - If similar entry found:
     - Display existing entry to user
     - Ask: "Similar entry found: [entry]. Update existing or add new?"
     - If "update": Modify existing entry with additional context
     - If "new": Continue to add new entry

4. **Format entry**
   - Use standard format from `@docs/POSTMORTEMS.md`:
     ```markdown
     ## [Date] - [Category] - [Title]
     
     **Task**: [Task context]
     **Struggle**: [Struggle details]
     **Resolution**: [Resolution]
     **Time Lost**: [Time lost]
     **Prevention**: [Prevention ideas]
     ```
   - Date format: `YYYY-MM-DD` (use current date)
   - Category: Use provided category (uppercase)

5. **Append entry**
   - Read `@docs/POSTMORTEMS.md`
   - Find "New Entries" section
   - Insert new entry at the top (newest first)
   - Write updated file

6. **Commit changes**
   - Stage file: `!git add docs/POSTMORTEMS.md`
   - Commit: `!git commit -m "Document [category] struggle in POSTMORTEMS.md"`
   - Display commit hash and summary

## Outputs

- Confirmation message with:
  - Entry added/updated
  - Entry location in `@docs/POSTMORTEMS.md`
  - Commit hash (if committed)
- Link to view entry in `@docs/POSTMORTEMS.md`

## Categories

Valid categories:
- `DEPENDENCY` - Dependency/package management issues
- `TOOLING` - Tool configuration or availability issues
- `CONFIG` - Configuration/environment issues
- `AWS` - AWS-specific issues
- `BUILD` - Build/compilation issues
- `TEST` - Testing framework or test execution issues
- `DEPLOY` - Deployment issues
- `UNDERSTANDING` - Code understanding or pattern recognition issues

## Notes

- This command is for manually documenting struggles that meet the criteria:
  - 3+ retry attempts on same operation
  - >15 minutes resolution time
  - Workaround required
- Entries are reviewed monthly to identify patterns and codify into `@CLAUDE.md` or `@.cursorrules`
- See `@docs/POSTMORTEMS.md` for full documentation of the post-mortem system
