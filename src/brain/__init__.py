"""Native two-stage brain (PKT-TB-008+).

The two-stage engine lives under ``src.brain.engine``. It replaces the
incumbent greedy cash-draining selection loop (``src/steps/decision_engine.py``)
with a strict Select-then-Allocate pipeline in which the held symbol *set* is a
pure function of the forecast and selection parameters and is constant under all
sizing parameters (the symbol-set-invariance-under-sizing property).
"""
