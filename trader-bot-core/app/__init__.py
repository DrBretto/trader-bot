"""app/ — the clean-spine entrypoint.

- ``handler``   — the thin router (dispatch only; no DS compute).
- ``night``     — the night forward pipeline the router dispatches to.
- ``morning`` / ``midday`` — PRESERVED paths (delegate to the prod phases).
- ``ops_probes``— governed, non-destructive diag / canary / watchdog-diag branches.
"""
