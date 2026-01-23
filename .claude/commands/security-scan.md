---
description: Security vulnerability scanner
---

# Security Scan Command

## Usage

```
/security-scan [--git-history]
```

Optional `--git-history` flag: If provided, scan git history for secrets (limited to last 100 commits).

## Inputs / Assumptions

- Scan current codebase for security issues
- Optionally scan git history (last 100 commits) if flag provided
- Reference `@CLAUDE.md` security defaults: no secrets in code, least privilege, etc.

## Steps

1. **Scan for hardcoded secrets**
   - Search for common secret patterns:
     - API keys: `!rg -i "api[_-]?key\s*[:=]\s*['\"][^'\"]{10,}" . --type-not git`
     - Passwords: `!rg -i "password\s*[:=]\s*['\"][^'\"]{6,}" . --type-not git`
     - Tokens: `!rg -i "(token|secret|private[_-]?key)\s*[:=]\s*['\"][^'\"]{10,}" . --type-not git`
     - AWS keys: `!rg -i "AKIA[0-9A-Z]{16}" . --type-not git`
     - Private keys: `!rg -E "-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----" . --type-not git`
   - Check `.env` files for committed secrets (should be in `.gitignore`)
   - Flag findings with file location and line number

2. **Check for insecure coding patterns**
   - **SQL Injection**:
     - String concatenation in SQL queries: `!rg -i "SELECT.*\+|INSERT.*\+|UPDATE.*\+|DELETE.*\+" . --type-not git`
     - Missing parameterized queries
   - **XSS vulnerabilities**:
     - Unsanitized user input in HTML: `!rg -i "innerHTML|dangerouslySetInnerHTML|document\.write" . --type-not git`
     - Missing output encoding
   - **CSRF vulnerabilities**:
     - Missing CSRF tokens in forms
     - Missing SameSite cookie attributes
   - **Authentication/Authorization**:
     - Weak password requirements
     - Missing rate limiting
     - Insecure session management

3. **Run dependency security audits**
   - Execute `/audit` command logic (reuse audit.md steps)
   - Focus on critical and high severity vulnerabilities
   - Include in security report

4. **Optional: Check git history for secrets** (if `--git-history` flag provided)
   - Limit to last 100 commits: `!git log --oneline -100`
   - Scan commit messages and diffs:
     - `!git log -100 -p | rg -i "(api[_-]?key|password|secret|token|private[_-]?key)"`
     - `!git log -100 -p | rg -E "AKIA[0-9A-Z]{16}"`
     - `!git log -100 -p | rg -E "-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"`
   - Flag commits containing potential secrets
   - **Warning**: If secrets found in history, recommend:
     - Rotate exposed credentials immediately
     - Consider rewriting git history if repository is private (with user confirmation)

5. **Generate security report**
   - **Immediate action required**:
     - Hardcoded secrets found
     - Critical vulnerabilities in dependencies
     - SQL injection vulnerabilities
   - **High priority**:
     - XSS vulnerabilities
     - Authentication weaknesses
     - High severity dependency vulnerabilities
   - **Medium priority**:
     - CSRF vulnerabilities
     - Missing security headers
     - Medium severity dependency vulnerabilities
   - **Low priority**:
     - Best practice violations
     - Low severity dependency vulnerabilities
   - For each finding:
     - Severity level
     - File location and line number
     - Description of issue
     - Remediation steps
     - References to security best practices

## Outputs

- Security report with:
  - Summary: Counts by severity (immediate/high/medium/low)
  - Detailed findings: Issue type, location, description, remediation
  - Dependency vulnerabilities (from audit)
  - Git history findings (if scanned)
  - Immediate action items prioritized
  - Remediation roadmap

## Struggle Tracking (When to Document)

Document in `@docs/POSTMORTEMS.md` when encountering:

- **False positives requiring investigation**: Security patterns flagged but require manual verification to confirm if real issues
- **Tool failures**: Security scanning tools fail to run or produce incorrect results
- **Unclear security patterns**: Code patterns that are ambiguous whether they represent security vulnerabilities
- **Git history analysis failures**: Unable to scan git history or parse results requiring manual review

**Criteria**: 3+ retries OR >15min resolution time OR workaround required

**Category**: UNDERSTANDING

## Post-Mortem Write Path

Before documenting a struggle:

1. Check `@docs/POSTMORTEMS.md` for similar entries using:
   - `!rg -n -i "security\|secret\|vulnerability" docs/POSTMORTEMS.md`
   - Search by category: `!rg -n "UNDERSTANDING" docs/POSTMORTEMS.md`

2. If duplicate found:
   - Ask user: "Similar entry found: [entry]. Update existing or add new?"
   - If update: Modify existing entry with additional context
   - If new: Add new entry (newest first)

3. If new entry:
   - Append to `@docs/POSTMORTEMS.md` in the "New Entries" section
   - Use standard format: date, category (UNDERSTANDING), title, task, struggle, resolution, time lost, prevention
   - Commit with message: `Document UNDERSTANDING struggle in POSTMORTEMS.md`
