"""monitors/ — substrate + line watchdogs (clean spine).

Relocated VERBATIM from ``src/brain/monitors.py`` (KEEP-code per the clean-rebuild
architecture). This packet (P4) ports ONLY ``check_substrate_fresh`` — the
freeze-detection watchdog the replay keystone's acceptance test exercises. The
remaining watchdogs relocate under P5/P8.
"""
from .substrate import LocalReplayStore, check_substrate_fresh  # noqa: F401
