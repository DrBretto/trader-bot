"""Regression guard: broker/cutover paths must not hardcode AWS default profile."""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]

FILES_TO_SCAN = [
    "scripts/bridge_cutover_continuity.py",
    "docs/OPERATIONS.md",
    "docs/DEPLOY.md",
    "docs/plans/2026-03-12-claude-code-prompt-cutover-continuity-bridge.md",
    "docs/plans/2026-03-12-claude-code-prompt-cutover-holdings-bootstrap.md",
    "docs/plans/2026-03-12-claude-code-prompt-drift-lockdown.md",
    "docs/plans/2026-03-12-claude-code-prompt-continuity-hotfix-deploy.md",
]

FORBIDDEN_PATTERNS = [
    re.compile(r"--profile\s+default\b"),
    re.compile(r"profile_name\s*=\s*['\"]default['\"]"),
    re.compile(r"\bAWS_PROFILE=default\b"),
    re.compile(r"\bAWS_DEFAULT_PROFILE=default\b"),
]


def test_no_hardcoded_default_profile_assumptions():
    violations = []

    for relative_path in FILES_TO_SCAN:
        path = REPO_ROOT / relative_path
        assert path.exists(), f"Expected file missing from guardrail scan: {relative_path}"
        text = path.read_text(encoding="utf-8")

        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                violations.append((relative_path, pattern.pattern))

    assert not violations, (
        "Hardcoded AWS default profile usage found in broker/cutover paths: "
        + ", ".join(f"{p} ({pat})" for p, pat in violations)
    )
